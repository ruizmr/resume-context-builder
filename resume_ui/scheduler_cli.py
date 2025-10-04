import argparse
import os
from pathlib import Path
from typing import Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore

from kb.db import get_database_url
from resume_ui.ingest_cli import main as ingest_main


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


def _job_ingest(input_dir: str, markdown_out_dir: Optional[str] = None) -> None:
    # Reuse existing CLI behavior
    args = [input_dir] + ([markdown_out_dir] if markdown_out_dir else [])
    import sys
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *args]
        ingest_main()
    finally:
        sys.argv = old_argv


def main() -> None:
    parser = argparse.ArgumentParser("context-scheduler", description="Schedule periodic ingest+upsert tasks")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_add = sub.add_parser("add", help="Add a new job")
    p_add.add_argument("job_id", help="Unique job id")
    p_add.add_argument("input_dir", help="Folder to ingest")
    p_add.add_argument("--md-out", dest="md_out", help="Markdown output dir (optional)")
    p_add.add_argument("--cron", dest="cron", help="Cron expression (e.g. '0 * * * *')")
    p_add.add_argument("--interval", dest="interval", type=int, help="Interval minutes (alternative to cron)")

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

        sched.add_job(
            _job_ingest,
            id=args.job_id,
            trigger=trigger,
            args=[str(in_dir), str(md_out) if md_out else None],
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


