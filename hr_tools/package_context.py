from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from repomix import (
	RepoProcessor,
	RepomixConfig,
	RepomixConfigIgnore,
	RepomixConfigOutput,
	RepomixConfigSecurity,
	RepomixOutputStyle,
)

import tiktoken


def create_hr_config(
	output_file: Path,
	*,
	header_text: str = "HR Candidate Resume Pack",
	instruction_file_path: Optional[Path] = None,
) -> RepomixConfig:
	"""Create a Repomix configuration optimized for HR resume packaging."""
	output_cfg = RepomixConfigOutput(
		file_path=str(output_file),
		style=RepomixOutputStyle.MARKDOWN,
		header_text=header_text,
		instruction_file_path=str(instruction_file_path) if instruction_file_path else "",
		remove_comments=False,
		remove_empty_lines=False,
		top_files_length=0,
		show_line_numbers=False,
		copy_to_clipboard=False,
		include_empty_directories=False,
		calculate_tokens=False,
		show_file_stats=False,
		show_directory_structure=False,
		parsable_style=False,
		truncate_base64=False,
		stdout=False,
		include_diffs=False,
	)

	ignore_cfg = RepomixConfigIgnore(
		custom_patterns=[
			"**/*.pdf",
			"**/*.jpg",
			"**/*.jpeg",
			"**/*.png",
			"**/*.gif",
			"**/*.csv",
			"**/*.json",
			"**/*.xlsx",
			"**/*.zip",
		],
		use_gitignore=False,
		use_default_ignore=False,
	)

	security_cfg = RepomixConfigSecurity(
		enable_security_check=False,
		exclude_suspicious_files=False,
	)

	config = RepomixConfig(
		output=output_cfg,
		ignore=ignore_cfg,
		security=security_cfg,
		include=["**/*.md"],
	)
	return config


def _strip_repomix_noise(output_content: str) -> str:
	"""Remove Repomix preamble sections like 'Notes:' and 'Additional Information:'
	and the informational footer lines, while preserving actual file content and
	any user-provided instructions/header text.

	Heuristic: drop blocks that start with 'Notes:' or 'Additional Information:'
	until a blank line or a '## File:' header is encountered. Also drop lines
	containing the Repomix link or 'User Provided Header:' metadata line.
	"""
	lines = output_content.splitlines(True)
	out: list[str] = []
	skipping = False
	for ln in lines:
		s = ln.strip()
		if s.startswith("Notes:") or s.startswith("Additional Information:"):
			skipping = True
			continue
		if "For more information about Repomix" in ln:
			continue
		if s.startswith("User Provided Header:"):
			continue
		if skipping:
			# end skip at blank line or next file header
			if s == "" or ln.startswith("## File:"):
				skipping = False
				# include the current line if it is a file header or the blank separator
				out.append(ln)
				continue
			# otherwise, keep skipping lines in the preamble block
			continue
		out.append(ln)
	return "".join(out)


def _split_output_by_tokens(
	output_content: str,
	*,
	max_tokens: int,
	encoding_name: str = "o200k_base",
	include_preamble_once: bool = True,
) -> List[str]:
	"""Split a large Repomix output into token-limited chunks at file boundaries.

	- Splits at lines starting with '## File:'.
	- Includes the preamble (everything before the first '## File:') only in the first chunk if include_preamble_once.
	"""
	enc = tiktoken.get_encoding(encoding_name)
	lines = output_content.splitlines(True)
	file_header_idxs: List[int] = [i for i, ln in enumerate(lines) if ln.startswith("## File: ")]

	# Build segments: preamble then each file block
	segments: List[str] = []
	if file_header_idxs:
		first_idx = file_header_idxs[0]
		preamble = "".join(lines[:first_idx])
		if preamble:
			segments.append(preamble)
		for j, idx in enumerate(file_header_idxs):
			end = file_header_idxs[j + 1] if j + 1 < len(file_header_idxs) else len(lines)
			segments.append("".join(lines[idx:end]))
	else:
		# No file headers; split by lines conservatively
		segments = ["".join(lines)]

	chunks: List[str] = []
	current_parts: List[str] = []
	current_tokens = 0

	def flush_chunk():
		if current_parts:
			chunks.append("".join(current_parts))

	for k, seg in enumerate(segments):
		seg_tokens = len(enc.encode(seg))
		is_preamble = (k == 0 and file_header_idxs)
		if is_preamble and include_preamble_once:
			# Always try to place preamble in first chunk; if it alone exceeds max, still place it
			if current_tokens == 0:
				current_parts.append(seg)
				current_tokens += seg_tokens
				continue
			# If preamble comes later (should not), flush first
			flush_chunk()
			current_parts = [seg]
			current_tokens = seg_tokens
			continue

		# If this segment alone is larger than max, split by lines
		if seg_tokens > max_tokens:
			seg_lines = seg.splitlines(True)
			for ln in seg_lines:
				ln_tokens = len(enc.encode(ln))
				if current_tokens + ln_tokens > max_tokens and current_tokens > 0:
					flush_chunk()
					current_parts = []
					current_tokens = 0
				current_parts.append(ln)
				current_tokens += ln_tokens
			continue

		# Normal case: add whole segment if it fits, otherwise flush and start new
		if current_tokens + seg_tokens > max_tokens and current_tokens > 0:
			flush_chunk()
			current_parts = []
			current_tokens = 0
		current_parts.append(seg)
		current_tokens += seg_tokens

	flush_chunk()
	return chunks


def package_markdown_directory(
	source_dir: str | Path,
	output_file: str | Path,
	*,
	instruction_file: Optional[str | Path] = None,
	header_text: str = "HR Candidate Resume Pack",
	max_tokens: Optional[int] = None,
	encoding_name: str = "o200k_base",
    predefined_md_files: Optional[List[str | Path]] = None,
) -> List[Path]:
	"""Package Markdown files under source_dir into a single output file using Repomix.

	Returns the output file path.
	"""
	directory = Path(source_dir).expanduser().resolve()
	output_path = Path(output_file).expanduser().resolve()
	output_path.parent.mkdir(parents=True, exist_ok=True)

	# Default HR config; instruction file path is optional
	config = create_hr_config(
		output_file=output_path,
		header_text=header_text,
		instruction_file_path=Path(instruction_file) if instruction_file else None,
	)

	processor = RepoProcessor(directory=str(directory), config=config)
	# Explicitly set file list to avoid ignore/include mismatches
	if predefined_md_files:
		md_files = [str(Path(p)) for p in predefined_md_files]
	else:
		md_files = [str(p) for p in directory.rglob("*.md") if p.is_file()]
	processor.set_predefined_file_paths(md_files)
	result = processor.process(write_output=True)

	# Clean informational preamble/footer from the output while retaining content
	cleaned = _strip_repomix_noise(result.output_content)

	# Ensure the file is written and sanitized
	if not output_path.exists():
		output_path.write_text(cleaned, encoding="utf-8")
	else:
		try:
			output_path.write_text(cleaned, encoding="utf-8")
		except Exception:
			# Best-effort overwrite; if it fails, existing file remains
			pass

	# If no max token limit requested, return single file
	if not max_tokens or max_tokens <= 0:
		return [output_path]

	# Split by tokens; include preamble only once (use cleaned content)
	chunks = _split_output_by_tokens(cleaned, max_tokens=max_tokens, encoding_name=encoding_name, include_preamble_once=True)
	if len(chunks) <= 1:
		return [output_path]

	# Write chunk files alongside the target
	base = output_path.with_suffix("")
	chunk_paths: List[Path] = []
	for idx, chunk_text in enumerate(chunks, start=1):
		chunk_path = Path(f"{base}.part{idx}.md")
		chunk_path.write_text(chunk_text, encoding="utf-8")
		chunk_paths.append(chunk_path)

	return chunk_paths


