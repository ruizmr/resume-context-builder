import os
import sys
from pathlib import Path
from typing import Iterable, List, Callable, Optional

# Optional dependency: prefer MarkItDown when available
try:
	from markitdown import MarkItDown  # type: ignore
	HAS_MARKITDOWN = True
except Exception:  # pragma: no cover
	MarkItDown = None  # type: ignore
	HAS_MARKITDOWN = False

# Fallback for PDF-only conversion when MarkItDown is not installed
try:
	from pdfminer.high_level import extract_text as pdf_extract_text  # type: ignore
	HAS_PDFMINER = True
except Exception:  # pragma: no cover
	HAS_PDFMINER = False


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


def convert_file_to_markdown(src_path: Path, output_path: Path, markitdown_client: "MarkItDown | None") -> None:
	"""Convert a single file to Markdown; prefer MarkItDown, fallback to pdfminer for PDFs."""
	markdown_text: str | None = None
	if HAS_MARKITDOWN and markitdown_client is not None:
		result = markitdown_client.convert_local(str(src_path))
		markdown_text = result.markdown
	elif src_path.suffix.lower() == ".pdf" and HAS_PDFMINER:
		text = pdf_extract_text(str(src_path)) or ""
		markdown_text = f"# {src_path.name}\n\n" + text
	else:
		raise RuntimeError("No converter available for this file type without MarkItDown installed.")

	ensure_directory(output_path)
	output_path.write_text(markdown_text, encoding="utf-8")


def convert_documents_to_markdown(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    write_index: bool = True,
    progress_cb: Optional[Callable[[int, int, Path], None]] = None,
    timing_cb: Optional[Callable[[Path, float], None]] = None,
    cancel_cb: Optional[Callable[[], bool]] = None,
) -> List[Path]:
	"""Convert all supported files under input_dir to Markdown under output_dir, mirroring structure.

	Returns a list of generated Markdown file paths.
	"""
	input_root = Path(input_dir).expanduser().resolve()
	output_root = Path(output_dir).expanduser().resolve()
	output_root.mkdir(parents=True, exist_ok=True)

	# Prefer builtins only; avoid optional heavy backends that may fail to install
	md_client = MarkItDown(enable_builtins=True, enable_plugins=False) if HAS_MARKITDOWN else None
	generated: List[Path] = []

	files = find_input_files(input_root)
	total = len(files)
	for idx, src in enumerate(files):
		if cancel_cb is not None:
			try:
				if cancel_cb():
					break
			except Exception:
				# ignore cancel errors
				pass
		rel = src.relative_to(input_root)
		md_path = output_root / rel.with_suffix(".md")
		try:
			if progress_cb is not None:
				try:
					progress_cb(idx + 1, total, src)
				except Exception:
					pass
			import time as _t
			_t0 = _t.perf_counter()
			convert_file_to_markdown(src, md_path, md_client)
			_dt = _t.perf_counter() - _t0
			if timing_cb is not None:
				try:
					timing_cb(src, _dt)
				except Exception:
					pass
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
    progress_cb: Optional[Callable[[int, int, Path], None]] = None,
    timing_cb: Optional[Callable[[Path, float], None]] = None,
    cancel_cb: Optional[Callable[[], bool]] = None,
) -> List[Path]:
	"""Backward compatible wrapper: converts any supported files under input_dir to Markdown."""
	return convert_documents_to_markdown(
		input_dir,
		output_dir,
		write_index=write_index,
		progress_cb=progress_cb,
		timing_cb=timing_cb,
		cancel_cb=cancel_cb,
	)


if __name__ == "__main__":
	# Simple CLI for ad-hoc use
	if len(sys.argv) < 3:
		print("Usage: python -m hr_tools.pdf_to_md <input_pdf_dir> <output_md_dir>")
		sys.exit(1)
	convert_pdfs_to_markdown(sys.argv[1], sys.argv[2])
	print("Done.")


