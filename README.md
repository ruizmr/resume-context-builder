# Context Packager

Convert documents to Markdown (MarkItDown), package into a single LLM-ready context (Repomix), and use a sleek Streamlit UI to copy/download.

## 1. Quick start (uv)



### macOS/Linux (full extras)
```bash
sh -c 'command -v uv >/dev/null 2>&1 || (curl -LsSf https://astral.sh/uv/install.sh | sh); PATH="$HOME/.local/bin:$PATH" uvx --python 3.12 --refresh --from "git+https://github.com/ruizmr/resume-context-builder.git@main?extra=full" resume-ui'
```


### Windows (full extras)
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -Command "if (-not (Get-Command uv -EA SilentlyContinue)) { iwr https://astral.sh/uv/install.ps1 -UseBasicParsing | iex }; $env:Path = \"$env:USERPROFILE\.local\bin;$env:Path\"; uvx --python 3.12 --refresh --from 'git+https://github.com/ruizmr/resume-context-builder.git@main?extra=full' resume-ui"
```

Use the sidebar to upload files or point to a folder. Download/copy packaged chunks. Check "Include in knowledge base" to persist chunks and use the built-in search.

For smoother live reload and file watching, the `full` extras include Watchdog.

KB persistence defaults to a local SQLite file at `~/.context-packager-state/context.db`. To use a shared DB, set `CONTEXT_DB_URL` before launching, for example on Linux/macOS:
```bash
export CONTEXT_DB_URL="sqlite:////absolute/path/to/context.db"
resume-ui
```

## 2. Install a shortcut (one-time)
Creates a persistent `resume-ui` command on your PATH.



### macOS/Linux (full extras)
```bash
sh -c 'command -v uv >/dev/null 2>&1 || (curl -LsSf https://astral.sh/uv/install.sh | sh); PATH="$HOME/.local/bin:$PATH" uv tool install --python 3.12 --force "git+https://github.com/ruizmr/resume-context-builder.git?extra=full" && echo "Installed. Next time just run: resume-ui"'
```


### Windows (full extras)
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -Command "if (-not (Get-Command uv -EA SilentlyContinue)) { iwr https://astral.sh/uv/install.ps1 -UseBasicParsing | iex }; $env:Path = \"$env:USERPROFILE\.local\bin;$env:Path\"; uv tool install --python 3.12 --force \"git+https://github.com/ruizmr/resume-context-builder.git?extra=full\"; Write-Host 'Installed. Next time just run: resume-ui'"
```

Next runs after install:
```bash
resume-ui
```

Available commands after install:
```bash
# Launch UI (Home + Manage knowledge)
resume-ui

# Convert to Markdown then upsert to KB (cron-friendly)
context-ingest "/path/to/input_dir" "/path/to/markdown_out_dir"

# Upsert existing Markdown into KB
context-upsert "/path/to/markdown_out_dir"

# Scheduler (persistent)
# Add a job that ingests every hour
context-scheduler add hourly_ingest "/path/to/input_dir" --md-out "/path/to/markdown_out_dir" --interval 60
# Or cron syntax
context-scheduler add nightly_ingest "/path/to/input_dir" --md-out "/path/to/markdown_out_dir" --cron "0 2 * * *"
# List jobs
context-scheduler list
# Remove job
context-scheduler remove hourly_ingest
# Start scheduler (blocking)
context-scheduler start
```

One-shot (no install):
```bash
# UI
sh -c 'command -v uv >/dev/null 2>&1 || (curl -LsSf https://astral.sh/uv/install.sh | sh); PATH="$HOME/.local/bin:$PATH" uvx --python 3.12 --refresh --from "git+https://github.com/ruizmr/resume-context-builder.git@main?extra=full" resume-ui'

# Ingest (convert + upsert)
sh -c 'command -v uv >/dev/null 2>&1 || (curl -LsSf https://astral.sh/uv/install.sh | sh); PATH="$HOME/.local/bin:$PATH" uvx --python 3.12 --refresh --from "git+https://github.com/ruizmr/resume-context-builder.git@main?extra=full" context-ingest "/path/to/input_dir" "/path/to/markdown_out_dir"'
```

Database configuration:
```bash
# SQLite (default)
# ~/.context-packager-state/context.db

# Postgres
export CONTEXT_DB_URL='postgresql+psycopg://user:pass@host:5432/dbname'
resume-ui
```

## 3. Troubleshooting PATH
- macOS/Linux: ensure `~/.local/bin` is on PATH (e.g., add `export PATH="$HOME/.local/bin:$PATH"` to your shell rc).
- Windows: ensure `%USERPROFILE%\.local\bin` is on PATH, then open a new terminal.

If `resume-ui` is not found right after install
- Quick fix (macOS/Linux):
```bash
source "$HOME/.local/bin/env" || { echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc && source ~/.zshrc; }
```
- Run via absolute path once:
```bash
"$HOME/.local/bin/resume-ui"
```
- Windows PowerShell:
```powershell
& "$env:USERPROFILE\.local\bin\resume-ui.exe"
```

## 4. Reset/nuke cached install (uv)

If you’re not seeing the latest UI changes:

```bash
# Remove cached venvs and archives
uv cache prune

# Remove previously installed tool shim (optional)
uv tool uninstall resume-ui || true

# Reinstall and pin to main
sh -c 'command -v uv >/dev/null 2>&1 || (curl -LsSf https://astral.sh/uv/install.sh | sh); PATH="$HOME/.local/bin:$PATH" uvx --python 3.12 --refresh --from "git+https://github.com/ruizmr/resume-context-builder.git@main?extra=full" resume-ui'
```

## 5. Scheduler visibility and continuous sync

- Jobs added via UI or `context-scheduler add` are mirrored in a lightweight `jobs` table for display.
- Every execution (scheduled or one-off) creates a `job_runs` record with:
  - status: queued, running, success, failed, cancelled
  - progress: 0-100, processed/total files, chunks upserted
  - recent log tail and last message
- In the UI Manage knowledge tab:
  - Add job with Daily/Weekly/Monthly/Interval/Cron.
  - Run now for ad-hoc runs.
  - View recent runs with progress and cancel running jobs.

Environment notes:
- Set `CONTEXT_DB_URL` to persist job/run history in a shared DB.
- APScheduler jobs are still stored in `apscheduler_jobs` table via `SQLAlchemyJobStore` for durability.
