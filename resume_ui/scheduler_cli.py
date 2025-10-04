import argparse
import os
from pathlib import Path
from typing import Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore

from kb.db import get_database_url
from resume_ui.ingest_cli import main as ingest_main
from kb.db import (
    ensure_job,
    start_job_run,
    update_job_run,
    finish_job_run,
    is_cancel_requested,
    fetch_last_success,
    fetch_job_sla,
)
from hr_tools.pdf_to_md import convert_pdfs_to_markdown
from kb.upsert import upsert_markdown_files


def _scheduler() -> BackgroundScheduler:
    jobstores = {
        'default': SQLAlchemyJobStore(url=get_database_url()),
    }
    executors = {
        'default': ThreadPoolExecutor(10),
    }
    job_defaults = {
        'coalesce': True,  # run once for overdue triggers
        'max_instances': 1,
        'misfire_grace_time': 300,
    }
    return BackgroundScheduler(jobstores=jobstores, executors=executors, job_defaults=job_defaults)


def _job_ingest(input_dir: str, markdown_out_dir: Optional[str] = None, job_id: Optional[str] = None, job_name: Optional[str] = None) -> None:
    # Instrumented ingest with progress tracking and cancel support
    try:
        if job_id:
            ensure_job(job_id, job_name, input_dir, markdown_out_dir)
    except Exception:
        pass

    run_id = start_job_run(job_id or None, input_dir, markdown_out_dir or None)
    try:
        def _cancel() -> bool:
            return is_cancel_requested(run_id)

        def _progress_files(i: int, total: int, p: Path) -> None:
            pct = int(min(100, max(0, (i * 100) / max(1, total))))
            try:
                update_job_run(run_id, progress=pct, processed_files=i, total_files=total, last_message=f"Converting {p.name}")
            except Exception:
                pass

        def _progress_upsert(i: int, total: int, p: Path) -> None:
            pct = int(min(100, max(0, (i * 100) / max(1, total))))
            try:
                update_job_run(run_id, progress=pct, last_message=f"Indexing {p.name}")
            except Exception:
                pass

        # Convert to markdown
        out_dir = Path(markdown_out_dir or (Path.home() / "context-packager-md")).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        md_files = convert_pdfs_to_markdown(input_dir, str(out_dir), progress_cb=_progress_files, cancel_cb=_cancel)
        update_job_run(run_id, last_message=f"Converted {len(md_files)} file(s)")

        # Upsert into KB
        count = upsert_markdown_files([Path(p) for p in md_files], progress_cb=_progress_upsert, cancel_cb=_cancel)
        update_job_run(run_id, chunks_upserted=count, last_message=f"Upserted {count} chunk(s)")
        # Finalize status depending on cancellation
        if is_cancel_requested(run_id):
            finish_job_run(run_id, status="cancelled")
        else:
            finish_job_run(run_id, status="success")
    except Exception as e:
        try:
            update_job_run(run_id, last_message=f"Error: {e}", error=str(e))
        except Exception:
            pass
        finish_job_run(run_id, status="failed", error=str(e))


def main() -> None:
    parser = argparse.ArgumentParser("context-scheduler", description="Schedule periodic ingest+upsert tasks")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_add = sub.add_parser("add", help="Add a new job")
    p_add.add_argument("job_id", help="Unique job id")
    p_add.add_argument("input_dir", help="Folder to ingest")
    p_add.add_argument("--md-out", dest="md_out", help="Markdown output dir (optional)")
    p_add.add_argument("--cron", dest="cron", help="Cron expression (e.g. '0 * * * *')")
    p_add.add_argument("--interval", dest="interval", type=int, help="Interval minutes (alternative to cron)")
    p_add.add_argument("--name", dest="name", help="Human-friendly job name")
    p_add.add_argument("--sla-minutes", dest="sla", type=int, help="SLA window in minutes for catch-up on startup")

    p_list = sub.add_parser("list", help="List jobs")

    p_remove = sub.add_parser("remove", help="Remove a job")
    p_remove.add_argument("job_id")

    p_start = sub.add_parser("start", help="Run scheduler (blocking)")

    args = parser.parse_args()
    sched = _scheduler()

    if args.cmd == "add":
        sched.start(paused=True)
        trigger = None
        if args.cron:
            from apscheduler.triggers.cron import CronTrigger
            trigger = CronTrigger.from_crontab(args.cron)
        elif args.interval:
            from apscheduler.triggers.interval import IntervalTrigger
            trigger = IntervalTrigger(minutes=int(args.interval))
        else:
            raise SystemExit("Specify --cron or --interval")

        # validate dirs
        in_dir = Path(args.input_dir).expanduser().resolve()
        if not in_dir.is_dir():
            raise SystemExit(f"Input directory not found: {in_dir}")
        md_out = Path(args.md_out).expanduser().resolve() if args.md_out else None

        ensure_job(args.job_id, args.name or None, str(in_dir), str(md_out) if md_out else None, args.sla if args.sla is not None else None)
        sched.add_job(
            _job_ingest,
            id=args.job_id,
            name=args.name or None,
            trigger=trigger,
            args=[str(in_dir), str(md_out) if md_out else None, args.job_id, args.name or None],
            replace_existing=True,
        )
        sched.shutdown(wait=False)
        print(f"Added job {args.job_id}")
        return

    if args.cmd == "list":
        sched.start(paused=True)
        for job in sched.get_jobs():
            print(f"{job.id}: {job.trigger} next={job.next_run_time}")
        sched.shutdown(wait=False)
        return

    if args.cmd == "remove":
        sched.start(paused=True)
        try:
            sched.remove_job(args.job_id)
            print(f"Removed job {args.job_id}")
        except Exception as e:
            print(f"Failed to remove: {e}")
        finally:
            sched.shutdown(wait=False)
        return

    if args.cmd == "start":
        # Blocking scheduler that runs due jobs on startup (coalesce)
        try:
            sched.start()
            # SLA catch-up: queue one-off runs for jobs out of SLA
            from apscheduler.triggers.date import DateTrigger
            from datetime import datetime, timezone, timedelta
            for job in sched.get_jobs():
                jid = job.id
                sla = fetch_job_sla(jid)
                if sla is None:
                    continue
                last = fetch_last_success(jid)
                overdue = False
                if last:
                    try:
                        last_dt = datetime.fromisoformat(last)
                        now = datetime.now(timezone.utc)
                        if (now - last_dt).total_seconds() > sla * 60:
                            overdue = True
                    except Exception:
                        overdue = True
                else:
                    overdue = True
                if overdue:
                    run_id = f"catchup::{jid}"
                    try:
                        sched.add_job(
                            _job_ingest,
                            id=run_id,
                            args=[job.args[0] if job.args else '', job.args[1] if (job.args and len(job.args)>1) else None, jid, f"Catch-up for {jid}"],
                            trigger=DateTrigger(run_date=datetime.now(timezone.utc) + timedelta(seconds=1)),
                            replace_existing=True,
                        )
                    except Exception:
                        pass
            print("Scheduler started. Press Ctrl+C to stop.")
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            sched.shutdown(wait=True)


if __name__ == "__main__":
    main()


