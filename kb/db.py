import os
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Tuple, Dict

from sqlalchemy import (
	create_engine,
	text,
	Engine,
)
import uuid
from datetime import datetime


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
		conn.execute(
			text(
				"""
				CREATE TABLE IF NOT EXISTS file_index (
					path TEXT PRIMARY KEY,
					sha256 TEXT,
					params_sig TEXT,
					updated_at TEXT
				)
				"""
			)
		)
		# Jobs metadata for scheduler visibility
		conn.execute(
			text(
				"""
				CREATE TABLE IF NOT EXISTS jobs (
					id TEXT PRIMARY KEY,
					name TEXT,
					input_dir TEXT,
					md_out_dir TEXT,
					sla_minutes INTEGER,
					created_at TEXT,
					updated_at TEXT
				)
				"""
			)
		)
		# Backfill migration: ensure sla_minutes column exists on older DBs
		try:
			if engine.dialect.name == "sqlite":
				rows = conn.execute(text("PRAGMA table_info(jobs)")).fetchall()
				cols = [r[1] for r in rows]
				if "sla_minutes" not in cols:
					conn.execute(text("ALTER TABLE jobs ADD COLUMN sla_minutes INTEGER"))
			else:
				# Best-effort for other dialects
				conn.execute(text("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS sla_minutes INTEGER"))
		except Exception:
			pass
		# Per-run tracking for progress/logs/history
		conn.execute(
			text(
				"""
				CREATE TABLE IF NOT EXISTS job_runs (
					id TEXT PRIMARY KEY,
					job_id TEXT,
					input_dir TEXT,
					md_out_dir TEXT,
					status TEXT,
					progress INTEGER,
					processed_files INTEGER,
					total_files INTEGER,
					chunks_upserted INTEGER,
					started_at TEXT,
					finished_at TEXT,
					last_message TEXT,
					log TEXT,
					cancel_requested INTEGER DEFAULT 0,
					error TEXT
				)
				"""
			)
		)
		conn.execute(
			text(
				"""
				CREATE INDEX IF NOT EXISTS idx_job_runs_started_at ON job_runs(started_at)
				"""
			)
		)
		# Prepare pgvector schema if available (best-effort, Postgres only)
		try:
			if engine.dialect.name == "postgresql":
				conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
				conn.execute(
					text(
						"""
						CREATE TABLE IF NOT EXISTS chunk_vectors (
							chunk_id BIGINT PRIMARY KEY,
							svd vector,
							updated_at TIMESTAMP
						)
						"""
					)
				)
		except Exception:
			pass


def ensure_job(job_id: str, name: str | None, input_dir: str, md_out_dir: str | None, sla_minutes: int | None = None) -> None:
	"""Upsert a job metadata row for visibility in UI.

	This is independent from APScheduler's job store and serves only for display.
	"""
	engine = get_engine()
	now = datetime.utcnow().isoformat()
	with engine.begin() as conn:
		conn.execute(
			text(
				"""
				INSERT INTO jobs(id, name, input_dir, md_out_dir, sla_minutes, created_at, updated_at)
				VALUES(:id, :name, :in_dir, :out_dir, :sla, :now, :now)
				ON CONFLICT(id) DO UPDATE SET
					name=excluded.name,
					input_dir=excluded.input_dir,
					md_out_dir=excluded.md_out_dir,
					sla_minutes=excluded.sla_minutes,
					updated_at=excluded.updated_at
				"""
			),
			{"id": job_id, "name": name, "in_dir": input_dir, "out_dir": md_out_dir, "sla": (int(sla_minutes) if sla_minutes is not None else None), "now": now},
		)


def start_job_run(job_id: str | None, input_dir: str, md_out_dir: str | None) -> str:
	"""Create a job_run row and return its id."""
	engine = get_engine()
	run_id = uuid.uuid4().hex
	now = datetime.utcnow().isoformat()
	with engine.begin() as conn:
		conn.execute(
			text(
				"""
				INSERT INTO job_runs(id, job_id, input_dir, md_out_dir, status, progress, processed_files, total_files, chunks_upserted, started_at, finished_at, last_message, log, cancel_requested, error)
				VALUES(:id, :job_id, :in_dir, :out_dir, 'running', 0, 0, 0, 0, :now, NULL, '', '', 0, NULL)
				"""
			),
			{"id": run_id, "job_id": job_id, "in_dir": input_dir, "out_dir": md_out_dir, "now": now},
		)
	return run_id


def update_job_run(
	run_id: str,
	*,
	status: str | None = None,
	progress: int | None = None,
	processed_files: int | None = None,
	total_files: int | None = None,
	chunks_upserted: int | None = None,
	last_message: str | None = None,
	append_log: str | None = None,
	error: str | None = None,
) -> None:
	"""Partial update of a job_run row."""
	engine = get_engine()
	# Build dynamic SET clause
	sets: list[str] = []
	params: dict = {"id": run_id}
	if status is not None:
		sets.append("status=:status")
		params["status"] = status
	if progress is not None:
		sets.append("progress=:progress")
		params["progress"] = int(max(0, min(100, progress)))
	if processed_files is not None:
		sets.append("processed_files=:pf")
		params["pf"] = int(processed_files)
	if total_files is not None:
		sets.append("total_files=:tf")
		params["tf"] = int(total_files)
	if chunks_upserted is not None:
		sets.append("chunks_upserted=:cu")
		params["cu"] = int(chunks_upserted)
	if last_message is not None:
		sets.append("last_message=:lm")
		params["lm"] = str(last_message)
	if error is not None:
		sets.append("error=:err")
		params["err"] = str(error)
	stmt = "UPDATE job_runs SET " + ", ".join(sets) + " WHERE id=:id"
	if sets:
		with engine.begin() as conn:
			conn.execute(text(stmt), params)
	if append_log:
		with engine.begin() as conn:
			conn.execute(
				text(
					"""
					UPDATE job_runs
					SET log=COALESCE(log,'') || :chunk
					WHERE id=:id
					"""
				),
				{"id": run_id, "chunk": (append_log or "")},
			)


def finish_job_run(run_id: str, *, status: str, error: str | None = None) -> None:
	"""Mark a run finished and set status and optional error."""
	engine = get_engine()
	now = datetime.utcnow().isoformat()
	with engine.begin() as conn:
		conn.execute(
			text(
				"""
				UPDATE job_runs
				SET status=:status, finished_at=:now, error=:error
				WHERE id=:id
				"""
			),
			{"id": run_id, "status": status, "now": now, "error": error},
		)


def request_cancel_run(run_id: str) -> None:
	"""Signal a running job to cancel."""
	engine = get_engine()
	with engine.begin() as conn:
		conn.execute(text("UPDATE job_runs SET cancel_requested=1 WHERE id=:id"), {"id": run_id})


def is_cancel_requested(run_id: str) -> bool:
	engine = get_engine()
	with engine.begin() as conn:
		row = conn.execute(text("SELECT cancel_requested FROM job_runs WHERE id=:id"), {"id": run_id}).fetchone()
		try:
			return bool(row[0]) if row else False
		except Exception:
			return False


def fetch_recent_runs(limit: int = 50, job_id: str | None = None) -> List[Tuple]:
	"""Return recent job runs, newest first."""
	engine = get_engine()
	params: dict = {"limit": int(limit)}
	query = "SELECT id, job_id, input_dir, md_out_dir, status, progress, processed_files, total_files, chunks_upserted, started_at, finished_at, last_message, substr(log, -5000), cancel_requested, error FROM job_runs"
	if job_id:
		query += " WHERE job_id=:job_id"
		params["job_id"] = job_id
	query += " ORDER BY started_at DESC LIMIT :limit"
	with engine.begin() as conn:
		rows = conn.execute(text(query), params)
		return [tuple(r) for r in rows]


def fetch_jobs() -> List[Tuple[str, str, str, str, str, str]]:
	"""Return (id, name, input_dir, md_out_dir, created_at, updated_at, sla_minutes)."""
	engine = get_engine()
	with engine.begin() as conn:
		rows = conn.execute(text("SELECT id, name, input_dir, md_out_dir, created_at, updated_at, sla_minutes FROM jobs ORDER BY updated_at DESC"))
		return [tuple(r) for r in rows]


def fetch_last_success(job_id: str) -> str | None:
	"""Return ISO timestamp of last successful run for a job, if any."""
	engine = get_engine()
	with engine.begin() as conn:
		row = conn.execute(text("SELECT started_at FROM job_runs WHERE job_id=:jid AND status='success' ORDER BY started_at DESC LIMIT 1"), {"jid": job_id}).fetchone()
		return str(row[0]) if row else None


def persist_chunk_vectors(engine: Engine, rows: List[Tuple[int, List[float]]]) -> int:
	"""Persist SVD vectors to Postgres pgvector table.

	- No-op for non-Postgres engines.
	- Creates extension/table if needed (best-effort).
	- Upserts by chunk_id.
	"""
	if not rows:
		return 0
	if engine.dialect.name != "postgresql":
		return 0
	# Ensure schema exists
	with engine.begin() as conn:
		try:
			conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
			conn.execute(
				text(
					"""
					CREATE TABLE IF NOT EXISTS chunk_vectors (
						chunk_id BIGINT PRIMARY KEY,
						svd vector,
						updated_at TIMESTAMP
					)
					"""
				)
			)
		except Exception:
			pass
	# Upsert in batches
	from math import ceil
	BATCH = 500
	updated = 0
	for i in range(0, len(rows), BATCH):
		batch = rows[i : i + BATCH]
		payload = []
		ts = datetime.utcnow()
		for cid, vec in batch:
			# pgvector input literal format: '[v1,v2,...]'
			literal = "[" + ",".join(f"{float(x):.6f}" for x in (vec or [])) + "]"
			payload.append({"id": int(cid), "svd": literal, "ts": ts})
		with engine.begin() as conn:
			try:
				conn.execute(
					text(
						"""
						INSERT INTO chunk_vectors(chunk_id, svd, updated_at)
						VALUES(:id, :svd::vector, :ts)
						ON CONFLICT (chunk_id) DO UPDATE SET
							svd = excluded.svd,
							updated_at = excluded.updated_at
						"""
					),
					payload,
				)
				updated += len(batch)
			except Exception:
				# continue with best-effort
				pass
	return updated


def fetch_chunk_vectors(engine: Engine, ids: List[int]) -> Dict[int, List[float]]:
	"""Fetch SVD vectors from pgvector table for given chunk IDs.

	Returns a mapping chunk_id -> vector (list of floats). No-op for non-Postgres.
	"""
	result: Dict[int, List[float]] = {}
	if not ids:
		return result
	if engine.dialect.name != "postgresql":
		return result
	# Sanitize and build IN clause
	safe_ids: List[int] = []
	for x in ids:
		try:
			safe_ids.append(int(x))
		except Exception:
			continue
	if not safe_ids:
		return result
	placeholders = ", ".join(str(i) for i in sorted(set(safe_ids)))
	with engine.begin() as conn:
		try:
			rows = conn.execute(text(f"SELECT chunk_id, svd::text FROM chunk_vectors WHERE chunk_id IN ({placeholders})"))
			for cid, vec_txt in rows:
				try:
					s = str(vec_txt or "").strip()
					if s.startswith("[") and s.endswith("]"):
						parts = s[1:-1].split(",")
						vec = [float(p) for p in parts if p.strip()]
						result[int(cid)] = vec
				except Exception:
					continue
		except Exception:
			return result
	return result


def fetch_job_sla(job_id: str) -> int | None:
	engine = get_engine()
	with engine.begin() as conn:
		row = conn.execute(text("SELECT sla_minutes FROM jobs WHERE id=:id"), {"id": job_id}).fetchone()
		try:
			return int(row[0]) if row and row[0] is not None else None
		except Exception:
			return None


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


def get_file_index(engine: Engine, path: str) -> Tuple[str, str] | None:
    """Return (sha256, params_sig) for a path if recorded."""
    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT sha256, params_sig FROM file_index WHERE path=:p"), {"p": path}
        ).fetchone()
        return (row[0], row[1]) if row else None


def upsert_file_index(engine: Engine, path: str, sha256: str, params_sig: str | None) -> None:
    from datetime import datetime
    now = datetime.utcnow().isoformat()
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO file_index(path, sha256, params_sig, updated_at)
                VALUES(:path, :sha, :psig, :now)
                ON CONFLICT(path) DO UPDATE SET
                    sha256=excluded.sha256,
                    params_sig=excluded.params_sig,
                    updated_at=excluded.updated_at
                """
            ),
            {"path": path, "sha": sha256, "psig": params_sig, "now": now},
        )


def delete_chunks_by_path(engine: Engine, path: str) -> int:
    """Delete all chunks that belong to a specific file path."""
    with engine.begin() as conn:
        res = conn.execute(text("DELETE FROM chunks WHERE path=:p"), {"p": path})
        try:
            return int(res.rowcount)  # type: ignore[attr-defined]
        except Exception:
            return 0

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

def fetch_chunk_by_id(engine: Engine, chunk_id: int) -> Tuple[int, str, str, str] | None:
	"""Fetch a single chunk by id."""
	with engine.begin() as conn:
		row = conn.execute(text("SELECT id, path, chunk_name, content FROM chunks WHERE id=:id"), {"id": int(chunk_id)}).fetchone()
		return (row[0], row[1], row[2], row[3]) if row else None



def factory_reset_db() -> None:
	"""Dangerous: wipe all KB and scheduler tables and reinitialize schema.

	- For SQLite: delete the DB file under STATE_DIR
	- For other DBs: DROP known tables and re-create KB schema
	"""
	url = get_database_url()
	try:
		if url.startswith("sqlite///") or url.startswith("sqlite:///"):
			# Remove SQLite file and exit early; schema will be re-created lazily on next engine init
			try:
				# Expected default path
				p = (STATE_DIR / 'context.db')
				if p.exists():
					p.unlink()
			except Exception:
				pass
			return
		# Non-SQLite: drop known tables and reinit schema
		engine = create_engine(url, future=True, pool_pre_ping=True)
		with engine.begin() as conn:
			for tbl in ("apscheduler_jobs", "chunks", "file_index", "jobs", "job_runs"):
				try:
					conn.execute(text(f"DROP TABLE IF EXISTS {tbl}"))
				except Exception:
					pass
		# Recreate KB schema
		init_schema(engine)
	except Exception:
		# Best-effort reset; swallow exceptions to avoid breaking UI
		return

