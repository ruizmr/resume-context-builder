from pathlib import Path
from typing import List, Tuple

from kb.db import get_engine, upsert_chunks


def slice_text(text: str, max_chars: int = 8000) -> List[str]:
	# crude slicer to ensure relatively even chunking
	return [text[i : i + max_chars] for i in range(0, len(text), max_chars)] if text else []


def upsert_markdown_files(md_files: List[Path]) -> int:
	engine = get_engine()
	records: List[Tuple[str, str, str]] = []
	for md in md_files:
		try:
			content = md.read_text(encoding="utf-8")
			chunks = slice_text(content)
			for i, ch in enumerate(chunks):
				records.append((str(md), f"part{i+1}", ch))
		except Exception:
			continue
	if records:
		upsert_chunks(engine, records)
	return len(records)


