from pathlib import Path
from typing import List, Tuple, Optional

import re
import tiktoken

from kb.db import get_engine, upsert_chunks, get_file_index, upsert_file_index, delete_chunks_by_path


def _find_prev_boundary(text: str, approx_char_idx: int, window: int = 500) -> int:
	"""Find a reasonable previous boundary near approx_char_idx.

	Prefers blank line or markdown heading. Falls back to the provided index.
	"""
	start = max(0, approx_char_idx - window)
	snippet = text[start:approx_char_idx]
	# Prefer double newline boundary
	bl = snippet.rfind("\n\n")
	if bl != -1:
		return start + bl + 2
	# Prefer previous heading
	for pat in (r"\n# ", r"\n## ", r"\n### "):
		m = list(re.finditer(pat, snippet))
		if m:
			return start + m[-1].start() + 1
	return approx_char_idx


def slice_text_tokens(
	text: str,
	max_tokens: int = 1500,
	overlap_tokens: int = 150,
	encoding_name: str = "o200k_base",
) -> List[str]:
	"""Token-based chunking with overlap; aligns boundaries to paragraphs/headings when possible."""
	if not text:
		return []
	if max_tokens <= 0:
		return [text]
	enc = tiktoken.get_encoding(encoding_name)
	toks = enc.encode(text)
	chunks: List[str] = []
	step = max(1, max_tokens - max(0, overlap_tokens))
	for i in range(0, len(toks), step):
		chunk_toks = toks[i : i + max_tokens]
		if not chunk_toks:
			break
		candidate = enc.decode(chunk_toks)
		# Try to align the end to a nearby boundary
		if i + max_tokens < len(toks):
			# estimate char idx of the end
			approx_end = len(enc.decode(toks[: i + max_tokens]))
			adj = _find_prev_boundary(text, approx_end)
			candidate = text[len(enc.decode(toks[:i])) : adj]
		chunks.append(candidate)
	return chunks


def upsert_markdown_files(
	md_files: List[Path],
	*,
	max_tokens_per_chunk: Optional[int] = 1500,
	overlap_tokens: int = 150,
	encoding_name: str = "o200k_base",
) -> int:
	engine = get_engine()
	records: List[Tuple[str, str, str]] = []
	for md in md_files:
		try:
            content = md.read_text(encoding="utf-8")
            # Compute file-level hash and params signature to skip unchanged
            import hashlib
            file_sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
            params_sig = f"t={max_tokens_per_chunk}|o={overlap_tokens}|e={encoding_name}"
            prev = get_file_index(engine, str(md))
            if prev and prev[0] == file_sha and prev[1] == params_sig:
                # Unchanged for current settings: skip recompute/upsert
                continue
            else:
                # Content changed or settings changed: remove old chunks for this file (idempotent) and re-upsert
                delete_chunks_by_path(engine, str(md))
			if max_tokens_per_chunk and max_tokens_per_chunk > 0:
				chunks = slice_text_tokens(
					content,
					max_tokens=max_tokens_per_chunk,
					overlap_tokens=overlap_tokens,
					encoding_name=encoding_name,
				)
			else:
				# Fallback single chunk
				chunks = [content]
			for i, ch in enumerate(chunks):
				records.append((str(md), f"part{i+1}", ch))
            # Record file hash and params
            upsert_file_index(engine, str(md), file_sha, params_sig)
		except Exception:
			continue
	if records:
		upsert_chunks(engine, records)
	return len(records)


