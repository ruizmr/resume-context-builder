import os
import sys
from pathlib import Path
from typing import Iterable, List

from markitdown import MarkItDown


def find_input_files(root: Path) -> List[Path]:
	"""Recursively find input files under the given root directory.

	Scans for all files except archives like .zip; conversion will be attempted
	with MarkItDown and skipped on failure.
	"""
	files: List[Path] = []
	for p in root.rglob("**/*"):
		if not p.is_file():
			continue
		# Skip archives to avoid extracting here
		if p.suffix.lower() in {".zip"}:
			continue
		files.append(p)
	return files


def ensure_directory(path: Path) -> None:
	"""Create parent directories for the given path if they do not exist."""
	path.parent.mkdir(parents=True, exist_ok=True)


def convert_file_to_markdown(src_path: Path, output_path: Path, markitdown_client: MarkItDown) -> None:
	"""Convert a single file to Markdown using MarkItDown and write to output_path."""
	result = markitdown_client.convert_local(str(src_path))
	markdown_text = result.markdown
	ensure_directory(output_path)
	output_path.write_text(markdown_text, encoding="utf-8")


def convert_documents_to_markdown(
	input_dir: str | Path,
	output_dir: str | Path,
	*,
	write_index: bool = True,
) -> List[Path]:
	"""Convert all supported files under input_dir to Markdown under output_dir, mirroring structure.

	Returns a list of generated Markdown file paths.
	"""
	input_root = Path(input_dir).expanduser().resolve()
	output_root = Path(output_dir).expanduser().resolve()
	output_root.mkdir(parents=True, exist_ok=True)

	# Prefer builtins only; avoid optional heavy backends that may fail to install
	md_client = MarkItDown(enable_builtins=True, enable_plugins=False)
	generated: List[Path] = []

	for src in find_input_files(input_root):
		rel = src.relative_to(input_root)
		md_path = output_root / rel.with_suffix(".md")
		try:
			convert_file_to_markdown(src, md_path, md_client)
			generated.append(md_path)
		except Exception as e:
			print(f"[warn] Failed to convert {src}: {e}", file=sys.stderr)

	if write_index:
		index_path = output_root / "INDEX.md"
		lines = ["# Converted Documents\n"]
		for md in sorted(generated):
			rel_md = md.relative_to(output_root)
			lines.append(f"- [{rel_md}]({rel_md.as_posix()})")
		index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
		if index_path not in generated:
			generated.append(index_path)

	return generated


def convert_pdfs_to_markdown(
	input_dir: str | Path,
	output_dir: str | Path,
	*,
	write_index: bool = True,
) -> List[Path]:
	"""Backward compatible wrapper: converts any supported files under input_dir to Markdown."""
	return convert_documents_to_markdown(input_dir, output_dir, write_index=write_index)


if __name__ == "__main__":
	# Simple CLI for ad-hoc use
	if len(sys.argv) < 3:
		print("Usage: python -m hr_tools.pdf_to_md <input_pdf_dir> <output_md_dir>")
		sys.exit(1)
	convert_pdfs_to_markdown(sys.argv[1], sys.argv[2])
	print("Done.")


