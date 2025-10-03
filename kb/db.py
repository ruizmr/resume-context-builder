import os
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Tuple

from sqlalchemy import (
	create_engine,
	text,
	Engine,
)


STATE_DIR = Path.home() / ".context-packager-state"
STATE_DIR.mkdir(parents=True, exist_ok=True)


def get_database_url() -> str:
	# Prefer explicit env vars; fallback to stateful SQLite file
	url = (
		os.getenv("CONTEXT_DB_URL")
		or os.getenv("DATABASE_URL")
		or os.getenv("PGSERVER_URL")
		or os.getenv("PGSERVER")
	)
	if url:
		return url
	return f"sqlite:///{(STATE_DIR / 'context.db').as_posix()}"


def get_engine(echo: bool = False) -> Engine:
	engine = create_engine(get_database_url(), echo=echo, future=True, pool_pre_ping=True)
	init_schema(engine)
	return engine


def init_schema(engine: Engine) -> None:
	# Split into single statements for SQLite compatibility
	with engine.begin() as conn:
		conn.execute(
			text(
				"""
				CREATE TABLE IF NOT EXISTS chunks (
					id INTEGER PRIMARY KEY AUTOINCREMENT,
					path TEXT,
					chunk_name TEXT,
					hash TEXT UNIQUE,
					content TEXT,
					created_at TEXT,
					updated_at TEXT
				)
				"""
			)
		)
		conn.execute(
			text(
				"""
				CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(hash)
				"""
			)
		)


def compute_hash(content: str) -> str:
	return hashlib.sha256(content.encode("utf-8")).hexdigest()


def upsert_chunks(engine: Engine, records: List[Tuple[str, str, str]]):
	"""Upsert chunk records.

	records items: (path, chunk_name, content)
	"""
	now = datetime.utcnow().isoformat()
	with engine.begin() as conn:
		for path, chunk_name, content in records:
			h = compute_hash(content)
			conn.execute(
				text(
					"""
					INSERT INTO chunks(path, chunk_name, hash, content, created_at, updated_at)
					VALUES(:path, :chunk_name, :hash, :content, :now, :now)
					ON CONFLICT(hash) DO UPDATE SET
						path=excluded.path,
						chunk_name=excluded.chunk_name,
						content=excluded.content,
						updated_at=excluded.updated_at
					"""
				),
				{"path": path, "chunk_name": chunk_name, "hash": h, "content": content, "now": now},
			)


def fetch_all_chunks(engine: Engine) -> List[Tuple[int, str, str, str]]:
	"""Return list of (id, path, chunk_name, content)."""
	with engine.begin() as conn:
		rows = conn.execute(text("SELECT id, path, chunk_name, content FROM chunks"))
		return [(r[0], r[1], r[2], r[3]) for r in rows]


def count_chunks(engine: Engine, like: str | None = None) -> int:
	"""Return total number of chunks, optionally filtered by LIKE pattern."""
	query = "SELECT COUNT(1) FROM chunks"
	params = {}
	if like:
		query += " WHERE path LIKE :like OR chunk_name LIKE :like OR content LIKE :like"
		params["like"] = like
	with engine.begin() as conn:
		row = conn.execute(text(query), params).fetchone()
		return int(row[0]) if row else 0


def fetch_chunks(engine: Engine, limit: int = 50, offset: int = 0, like: str | None = None) -> List[Tuple[int, str, str, str]]:
	"""Paginated fetch of (id, path, chunk_name, content) with optional LIKE filter."""
	base = "SELECT id, path, chunk_name, content FROM chunks"
	params = {"limit": int(limit), "offset": int(offset)}
	if like:
		base += " WHERE path LIKE :like OR chunk_name LIKE :like OR content LIKE :like"
		params["like"] = like
	base += " ORDER BY id DESC LIMIT :limit OFFSET :offset"
	with engine.begin() as conn:
		rows = conn.execute(text(base), params)
		return [(r[0], r[1], r[2], r[3]) for r in rows]


def delete_chunks_by_ids(engine: Engine, ids: List[int]) -> int:
	"""Delete chunks by id list. Returns number of rows deleted."""
	if not ids:
		return 0
	# sanitize to ints and unique
	safe_ids = []
	for i in ids:
		try:
			safe_ids.append(int(i))
		except Exception:
			continue
	if not safe_ids:
		return 0
	placeholders = ", ".join(str(i) for i in sorted(set(safe_ids)))
	with engine.begin() as conn:
		res = conn.execute(text(f"DELETE FROM chunks WHERE id IN ({placeholders})"))
		try:
			return int(res.rowcount)  # type: ignore[attr-defined]
		except Exception:
			return 0


