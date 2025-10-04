import os
import base64
import zipfile
import json
import uuid
from pathlib import Path
import importlib.resources as resources
from datetime import time as dt_time

import streamlit as st
import tiktoken
import streamlit.components.v1 as components

from hr_tools.pdf_to_md import convert_pdfs_to_markdown
from hr_tools.package_context import package_markdown_directory
from kb.upsert import upsert_markdown_files
from kb.db import get_engine, fetch_all_chunks, count_chunks, fetch_chunks, delete_chunks_by_ids, fetch_chunk_by_id, STATE_DIR, get_database_url
from kb.search import HybridSearcher


st.set_page_config(page_title="Context Packager", layout="wide", initial_sidebar_state="collapsed")
st.title("Context Packager")

# Defaults designed to be portable on any machine
default_pdf_dir = ""
default_md_dir = str(Path.home() / "context-packager-md")
default_out_file = str(Path.home() / "context-packager-output" / "context.md")

try:
    # Try to resolve packaged instructions file
    _instr = resources.files("hr_tools").joinpath("instruction.md")
    default_instr = str(_instr) if _instr and _instr.is_file() else ""
except Exception:
    default_instr = ""

uploads_root = str(Path.home() / ".context-packager-uploads")
Path(uploads_root).mkdir(parents=True, exist_ok=True)


# Persistent settings
SETTINGS_FILE = STATE_DIR / "settings.json"
PERSIST_KEYS = [
    "md_dir",
    "output_file",
    "attach_instructions",
    "instruction_file",
    "header_text",
    "max_tokens",
    "encoding_name",
    "kb_top_k",
    "kb_min_score",
        "kb_neighbors",
        "kb_sequence",
]


def _load_saved_settings() -> dict:
    try:
        if SETTINGS_FILE.exists():
            return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save_settings(payload: dict) -> None:
    try:
        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS_FILE.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        raise e


# Load persisted settings into session (first run only)
_saved = _load_saved_settings()
for _k in PERSIST_KEYS:
    if _k in _saved and _k not in st.session_state:
        st.session_state[_k] = _saved[_k]

if "instruction_file" not in st.session_state:
    st.session_state["instruction_file"] = default_instr

# Reusable copy widget

with st.sidebar:
    st.header("Settings")
    md_dir = st.text_input(
        "Markdown output directory",
        value=st.session_state.get("md_dir", default_md_dir),
        key="md_dir",
    )
    output_file = st.text_input(
        "Context output file",
        value=st.session_state.get("output_file", default_out_file),
        key="output_file",
    )
    attach_instructions = st.checkbox(
        "Attach instructions",
        value=bool(st.session_state.get("attach_instructions", True)),
        key="attach_instructions",
    )
    instruction_file = st.text_input(
        "Instructions file (Markdown)",
        value=st.session_state.get("instruction_file", default_instr),
        key="instruction_file",
    )
    header_text = st.text_input(
        "Document header",
        value=st.session_state.get("header_text", "Context Pack"),
        key="header_text",
    )
    max_tokens = st.number_input(
        "Max tokens per chunk (0 = no split)",
        value=int(st.session_state.get("max_tokens", 120000)),
        min_value=0,
        step=1000,
        key="max_tokens",
    )
    encoding_name = st.selectbox(
        "Tokenizer",
        options=["o200k_base", "cl100k_base"],
        index=(0 if st.session_state.get("encoding_name", "o200k_base") == "o200k_base" else 1),
        key="encoding_name",
    )
    # persist settings so search uses the same caps/encoding
    st.session_state["max_tokens_config"] = int(max_tokens or 0)
    # moved include_kb next to build controls below
    st.divider()
    st.subheader("Edit and delete knowledge")
    st.caption("KB search configuration")
    kb_top_k = st.number_input(
        "Max results",
        value=int(st.session_state.get("kb_top_k", 5)),
        min_value=1,
        max_value=50,
        step=1,
        key="kb_top_k",
    )
    kb_min_score = st.slider(
        "Minimum score",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state.get("kb_min_score", 0.005)),
        step=0.005,
        key="kb_min_score",
    )
    kb_neighbors = st.number_input(
        "Neighbors to include (per match)",
        value=int(st.session_state.get("kb_neighbors", 0)),
        min_value=0,
        max_value=10,
        step=1,
        key="kb_neighbors",
    )
    kb_sequence = st.checkbox(
        "Preserve document sequence (group by file, in order)",
        value=bool(st.session_state.get("kb_sequence", True)),
        key="kb_sequence",
    )

    st.divider()
    st.caption("Instructions (optional)")

    @st.dialog("Edit instructions")
    def edit_instructions_dialog():
        current_path = st.session_state.get("instruction_file") or default_instr or str(Path.home() / "instruction.md")
        try:
            default_text = Path(current_path).read_text(encoding="utf-8") if current_path and os.path.isfile(current_path) else ""
        except Exception:
            default_text = ""
        new_path = st.text_input("Save as", value=str(current_path))
        text = st.text_area("Instructions (Markdown)", value=default_text, height=300)
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Save", use_container_width=True):
                try:
                    p = Path(new_path).expanduser().resolve()
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text(text or "", encoding="utf-8")
                    st.session_state["instruction_file"] = str(p)
                    st.success(f"Saved to {p}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to save: {e}")
        with col_b:
            st.button("Cancel", use_container_width=True)

    if st.button("Edit instructions", use_container_width=True):
        edit_instructions_dialog()

    st.divider()
    if st.button("Save settings", use_container_width=True):
        try:
            payload = {k: st.session_state.get(k) for k in PERSIST_KEYS}
            _save_settings(payload)
            st.success("Settings saved")
        except Exception as e:
            st.error(f"Failed to save settings: {e}")

def render_copy_button(label: str, text: str, height: int = 110):
    b64 = base64.b64encode(text.encode("utf-8")).decode("ascii")
    tmpl = """
    <style>
    .copy-wrap { position: relative; width: 100%; margin: 0.2rem 0; }
    .copy-btn {
      all: unset; display: inline-block; width: 100%; text-align: center;
      padding: 0.6rem 1rem; border-radius: 0.25rem; background-color: #F63366;
      color: #fff; cursor: pointer; font-weight: 600; box-shadow: rgba(0,0,0,0.1) 0 1px 2px;
      transition: filter .15s ease-in-out, background-color .15s ease-in-out;
    }
    .copy-btn:hover { filter: brightness(0.95); }
    .copy-btn.copied { background-color: #22c55e; }
    .toast {
      position: absolute; top: -2.2rem; right: 0.25rem; background: #22c55e; color: #fff;
      padding: 4px 10px; border-radius: 6px; font-size: 0.85rem; display: inline-flex; align-items: center;
      gap: 6px; opacity: 0; transform: translateY(4px); transition: opacity .18s, transform .18s;
    }
    .toast.show { opacity: 1; transform: translateY(0); }
    .toast .check { font-weight: 900; }
    </style>
    <script>
    async function copyGeneric(){
      const text = atob('{{B64}}');
      try { await navigator.clipboard.writeText(text); } catch(e) {
        const ta = document.createElement('textarea'); ta.value = text; document.body.appendChild(ta);
        ta.focus(); ta.select(); try { document.execCommand('copy'); } catch(e2) {} document.body.removeChild(ta);
      }
      const toast = document.getElementById('copy-toast');
      const btn = document.getElementById('copy-btn');
      if (btn) {
        const original = btn.textContent;
        btn.classList.add('copied'); btn.textContent = 'Copied!';
        setTimeout(()=>{ btn.classList.remove('copied'); btn.textContent = original; }, 1200);
      }
      if (toast) { toast.classList.add('show'); setTimeout(()=> toast.classList.remove('show'), 1200); }
    }
    </script>
    <div class="copy-wrap">
      <button id="copy-btn" class="copy-btn" onclick="copyGeneric()">{{LABEL}}</button>
      <span id="copy-toast" class="toast"><span class="check">✓</span> Copied to clipboard</span>
    </div>
    """
    html = tmpl.replace("{{B64}}", b64).replace("{{LABEL}}", label)
    components.html(html, height=height)

# Tabs
home_tab, manage_tab = st.tabs(["Home", "Manage knowledge"])

with home_tab:
    # Global search bar pinned at top (independent form)
    with st.form("kb_search_form", clear_on_submit=False):
        top_cols = st.columns([8, 1])
        with top_cols[0]:
            st.text_input(
                "Search knowledge base",
                value=st.session_state.get("q_top", ""),
                placeholder="Search...",
                label_visibility="collapsed",
                key="q_top",
            )
        with top_cols[1]:
            submitted_search = st.form_submit_button("Search")

        if submitted_search and st.session_state.get("q_top", "").strip():
            try:
                engine = get_engine()
                # cache searcher in session to avoid refit on every search
                if "kb_searcher" not in st.session_state or (st.session_state.get("kb_searcher_docs_len") is not None and st.session_state.get("kb_searcher_docs_len") != len(fetch_all_chunks(engine))):
                    docs = fetch_all_chunks(engine)
                    searcher = HybridSearcher()
                    searcher.fit(docs)
                    st.session_state["kb_searcher"] = searcher
                    st.session_state["kb_searcher_docs_len"] = len(docs)
                else:
                    searcher = st.session_state["kb_searcher"]
                top_k = int(st.session_state.get("kb_top_k", 5))
                min_score = float(st.session_state.get("kb_min_score", 0.0))
                results = searcher.search(
                    st.session_state["q_top"],
                    top_k=top_k,
                    neighbors=int(st.session_state.get("kb_neighbors", 0)),
                    sequence=bool(st.session_state.get("kb_sequence", True)),
                )
                # filter by minimum score
                results = [r for r in results if r[1] >= min_score]

                if not results:
                    st.session_state["kb_results_agg"] = ""
                    st.session_state["kb_results_list"] = []
                else:
                    sections = []
                    for cid, score, path, cname, snippet, full_text in results:
                        header = f"{path} :: {cname} — score {score:.3f}"
                        sections.append(f"{header}\n\n{full_text.strip()}")
                    aggregated = "\n\n---\n\n".join(sections)

                    # Enforce overall token cap using the same token limit as packaging
                    max_tok = int(st.session_state.get("max_tokens_config") or 0)
                    enc_name = st.session_state.get("encoding_name", "o200k_base")
                    if max_tok and max_tok > 0:
                        try:
                            enc = tiktoken.get_encoding(enc_name)
                            toks = enc.encode(aggregated)
                            if len(toks) > max_tok:
                                aggregated = enc.decode(toks[:max_tok])
                        except Exception:
                            pass

                    st.session_state["kb_results_agg"] = aggregated
                    st.session_state["kb_results_list"] = results
            except Exception as e:
                st.session_state["kb_results_agg"] = ""
                st.session_state["kb_results_list"] = []
                st.error(f"Search failed: {e}")

    # Always render last search results if present (no refresh required)
    if "kb_results_agg" in st.session_state and st.session_state["kb_results_agg"] is not None:
        if st.session_state["kb_results_agg"]:
            st.subheader("Search results")
            render_copy_button("Copy all results", st.session_state["kb_results_agg"], height=80)
            st.text_area("Aggregated results", st.session_state["kb_results_agg"], height=400)
        else:
            st.info("No results")

    # Input and packaging live only on Home tab
    st.subheader("Input")
    uploaded_files = st.file_uploader("Upload files or ZIP", type=None, accept_multiple_files=True, key="home_file_uploader")
    fallback_dir_main = st.text_input("Or enter a directory path (optional)", value="", key="home_fallback_dir")

    st.write("")
    left, mid, right = st.columns([1,2,1])
    with mid:
        include_kb = st.checkbox("Include in knowledge base (stateful)", value=True)
        start = st.button("Build context", type="primary", use_container_width=True)
        clear = st.button("Reset form", use_container_width=True)

    if clear:
        st.session_state.pop("context_content", None)
        st.session_state.pop("out_paths", None)
        st.session_state.pop("selected_chunk", None)

    if start:
        effective_input_dir = None
        if uploaded_files:
            temp_root = Path(uploads_root) / f"session-{uuid.uuid4().hex[:8]}"
            temp_root.mkdir(parents=True, exist_ok=True)
            combined_dir = temp_root / "combined"
            combined_dir.mkdir(parents=True, exist_ok=True)
            for f in uploaded_files:
                name = Path(f.name).name
                if name.lower().endswith(".zip"):
                    zip_path = temp_root / name
                    zip_path.write_bytes(f.read())
                    with zipfile.ZipFile(zip_path, "r") as zf:
                        zf.extractall(combined_dir)
                else:
                    (combined_dir / name).write_bytes(f.read())
            effective_input_dir = str(combined_dir.resolve())
        elif fallback_dir_main and os.path.isdir(fallback_dir_main):
            effective_input_dir = fallback_dir_main
        else:
            st.error("Please upload files/ZIP or provide a valid directory path.")
            st.stop()

        Path(md_dir).mkdir(parents=True, exist_ok=True)
        Path(Path(output_file).parent).mkdir(parents=True, exist_ok=True)

        with st.spinner("Converting to Markdown..."):
            prog = st.progress(0, text="Scanning files…")
            last_file = st.empty()
            slow_log: list[tuple[str, float]] = []
            def _progress(i: int, total: int, p: Path):
                prog.progress(min(100, int(i * 100 / max(1, total))), text=f"{i}/{total}: {p.name}")
                last_file.write(f"Processing: {p}")
            def _timing(p: Path, dt: float):
                slow_log.append((str(p), dt))
            generated_md = convert_pdfs_to_markdown(
                effective_input_dir,
                md_dir,
                progress_cb=_progress,
                timing_cb=_timing,
            )
            if slow_log:
                slow_log.sort(key=lambda x: x[1], reverse=True)
                top = "\n".join([f"{Path(p).name} — {t:.2f}s" for p, t in slow_log[:5]])
                st.caption("Slowest files:")
                st.text(top)
            st.success(f"Converted {len(generated_md)} markdown files.")

        with st.spinner("Packaging context with Repomix..."):
            instr_path = instruction_file if (attach_instructions and instruction_file and os.path.isfile(instruction_file)) else None
            from time import strftime
            run_tag = strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
            base = Path(output_file)
            unique_output = base.with_name(f"{base.stem}_{run_tag}{base.suffix}")
            out_paths = package_markdown_directory(
                md_dir,
                str(unique_output),
                instruction_file=instr_path,
                header_text=header_text,
                max_tokens=max_tokens or None,
                encoding_name=encoding_name,
                predefined_md_files=[str(p) for p in generated_md],
            )

            if include_kb:
                try:
                    count = upsert_markdown_files([Path(p) for p in generated_md])
                    st.success(f"Upserted {count} chunk(s) into knowledge base.")
                    # Invalidate cached searcher so new docs are included next search
                    st.session_state.pop("kb_searcher", None)
                    st.session_state.pop("kb_searcher_docs_len", None)
                except Exception as e:
                    st.error(f"KB upsert failed: {e}")

            st.success(f"Packaged {len(out_paths)} file(s)")
            try:
                content = Path(out_paths[0]).read_text(encoding="utf-8")
                st.session_state["context_content"] = content
            except Exception as e:
                st.error(f"Failed to read output file: {e}")
            st.session_state["out_paths"] = [str(p) for p in out_paths]
            enc = tiktoken.get_encoding(encoding_name)
            rows = []
            for p in out_paths:
                text = Path(p).read_text(encoding="utf-8")
                tok = len(enc.encode(text))
                rows.append((str(p), tok))
            st.write("Generated chunks (tokens):")
            for p, tok in rows:
                st.write(f"{p} — tokens: {tok}")

    if "out_paths" in st.session_state and st.session_state["out_paths"]:
        st.subheader("Context preview")
        options = [f"Part {i+1}: {Path(p).name}" for i, p in enumerate(st.session_state["out_paths"])]
        idx = st.session_state.get("selected_chunk", 0)
        idx = st.selectbox("Select chunk", options=list(range(len(options))), format_func=lambda i: options[i], index=min(idx, len(options)-1))
        st.session_state["selected_chunk"] = idx
        selected_path = st.session_state["out_paths"][idx]
        selected_content = Path(selected_path).read_text(encoding="utf-8")

        col_dl, col_cp = st.columns(2)
        with col_dl:
            st.download_button(
                label=f"Download",
                data=selected_content,
                file_name=Path(selected_path).name,
                mime="text/markdown",
                use_container_width=True,
            )
        with col_cp:
            render_copy_button("Copy to clipboard", selected_content, height=110)

        st.text_area("Preview", selected_content, height=400)

with manage_tab:
    st.subheader("Manage knowledge base")
    engine = get_engine()
    st.divider()
    st.subheader("Enable continuous folder sync")

    # Scheduler controls (uses same DB via SQLAlchemy job store)
    @st.dialog("Choose folder to sync")
    def _choose_sync_folder_dialog():
        try:
            cwd_str = st.session_state.get("sync_browser_cwd", str(Path.home()))
            cwd = Path(cwd_str).expanduser().resolve()
        except Exception:
            cwd = Path.home()
            st.session_state["sync_browser_cwd"] = str(cwd)
        st.caption(f"Current: {cwd}")
        cols = st.columns([1, 1, 1])
        with cols[0]:
            if st.button("Select this folder", use_container_width=True):
                st.session_state["sync_folder"] = str(cwd)
                st.success(f"Selected: {cwd}")
                st.rerun()
        with cols[1]:
            if st.button("Up", use_container_width=True, disabled=(cwd.parent == cwd)):
                st.session_state["sync_browser_cwd"] = str(cwd.parent)
                st.rerun()
        st.divider()
        try:
            dirs = [p for p in cwd.iterdir() if p.is_dir()]
            dirs.sort(key=lambda p: p.name.lower())
        except Exception:
            dirs = []
        for d in dirs:
            if st.button(d.name + "/", key=f"sync_browse_{d}"):
                st.session_state["sync_browser_cwd"] = str(d)
                st.rerun()

    @st.dialog("Choose Markdown output directory")
    def _choose_md_out_dialog():
        try:
            cwd_str = st.session_state.get("sync_md_browser_cwd", str(Path.home()))
            cwd = Path(cwd_str).expanduser().resolve()
        except Exception:
            cwd = Path.home()
            st.session_state["sync_md_browser_cwd"] = str(cwd)
        st.caption(f"Current: {cwd}")
        cols = st.columns([1, 1, 1])
        with cols[0]:
            if st.button("Select this folder", use_container_width=True):
                st.session_state["sync_md_out"] = str(cwd)
                st.success(f"Selected: {cwd}")
                st.rerun()
        with cols[1]:
            if st.button("Up", use_container_width=True, disabled=(cwd.parent == cwd)):
                st.session_state["sync_md_browser_cwd"] = str(cwd.parent)
                st.rerun()
        st.divider()
        try:
            dirs = [p for p in cwd.iterdir() if p.is_dir()]
            dirs.sort(key=lambda p: p.name.lower())
        except Exception:
            dirs = []
        for d in dirs:
            if st.button(d.name + "/", key=f"md_browse_{d}"):
                st.session_state["sync_md_browser_cwd"] = str(d)
                st.rerun()

    col_a, col_b = st.columns([3, 2])
    with col_a:
        sync_folder = st.text_input("Folder to sync", value=st.session_state.get("sync_folder", ""), key="sync_folder", placeholder="/path/to/folder")
        if st.button("Browse…", key="browse_sync_folder"):
            _choose_sync_folder_dialog()
        md_out_target = st.text_input("Markdown output dir (optional)", value=st.session_state.get("sync_md_out", st.session_state.get("md_dir", default_md_dir)), key="sync_md_out")
        if st.button("Browse…", key="browse_md_out"):
            _choose_md_out_dialog()
    with col_b:
        schedule_mode = st.selectbox("Schedule", options=["Daily", "Every X minutes", "Weekly", "Monthly", "Cron (advanced)"], index=0, key="sync_mode")
        interval_minutes = int(st.number_input("Interval (minutes)", value=int(st.session_state.get("sync_interval", 60)), min_value=1, step=1, key="sync_interval")) if schedule_mode == "Every X minutes" else 1440
        cron_expr = st.text_input("Cron (min hr dom mon dow)", value=st.session_state.get("sync_cron", "0 2 * * *"), key="sync_cron") if schedule_mode == "Cron (advanced)" else st.session_state.get("sync_cron", "0 2 * * *")
        if schedule_mode == "Daily":
            st.session_state["sync_time"] = st.time_input("Time of day", value=st.session_state.get("sync_time", dt_time(2, 0)), key="sync_time")
        elif schedule_mode == "Weekly":
            days_options = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            st.session_state["sync_weekdays"] = st.multiselect("Days of week", options=days_options, default=st.session_state.get("sync_weekdays", ["Mon"]), key="sync_weekdays")
            st.session_state["sync_time_weekly"] = st.time_input("Time of day", value=st.session_state.get("sync_time_weekly", dt_time(2, 0)), key="sync_time_weekly")
        elif schedule_mode == "Monthly":
            st.session_state["sync_dom"] = int(st.number_input("Day of month", min_value=1, max_value=31, value=int(st.session_state.get("sync_dom", 1)), step=1, key="sync_dom"))
            st.session_state["sync_time_monthly"] = st.time_input("Time of day", value=st.session_state.get("sync_time_monthly", dt_time(2, 0)), key="sync_time_monthly")

    col_btn1, col_btn2, col_btn3, col_btn4 = st.columns([1,1,1,1])
    with col_btn1:
        start_sync = st.button("Start", use_container_width=True)
    with col_btn2:
        stop_sync = st.button("Stop", use_container_width=True)
    with col_btn3:
        list_sync = st.button("List jobs", use_container_width=True)
    with col_btn4:
        remove_sync = st.button("Remove job", use_container_width=True)

    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.executors.pool import ThreadPoolExecutor
    from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
    from apscheduler.triggers.interval import IntervalTrigger
    from apscheduler.triggers.cron import CronTrigger
    from resume_ui.scheduler_cli import _job_ingest as _job_ingest_fn

    def _get_scheduler():
        jobstores = {'default': SQLAlchemyJobStore(url=get_database_url())}
        executors = {'default': ThreadPoolExecutor(4)}
        job_defaults = {'coalesce': True, 'max_instances': 1, 'misfire_grace_time': 300}
        return BackgroundScheduler(jobstores=jobstores, executors=executors, job_defaults=job_defaults)

    def _ensure_sched_started():
        if "ui_scheduler" not in st.session_state:
            st.session_state["ui_scheduler"] = _get_scheduler()
            st.session_state["ui_scheduler"].start()

    job_id = "ui_continuous_sync"
    if start_sync:
        try:
            _ensure_sched_started()
            sched = st.session_state["ui_scheduler"]
            trigger = None
            if schedule_mode == "Daily":
                t = st.session_state.get("sync_time", dt_time(2, 0))
                trigger = CronTrigger(hour=int(getattr(t, 'hour', 2)), minute=int(getattr(t, 'minute', 0)))
            elif schedule_mode == "Weekly":
                t = st.session_state.get("sync_time_weekly", dt_time(2, 0))
                days = st.session_state.get("sync_weekdays", ["Mon"]) or ["Mon"]
                dow_map = {"Mon": "mon", "Tue": "tue", "Wed": "wed", "Thu": "thu", "Fri": "fri", "Sat": "sat", "Sun": "sun"}
                day_str = ",".join([dow_map.get(d, "mon") for d in days])
                trigger = CronTrigger(day_of_week=day_str, hour=int(getattr(t, 'hour', 2)), minute=int(getattr(t, 'minute', 0)))
            elif schedule_mode == "Monthly":
                t = st.session_state.get("sync_time_monthly", dt_time(2, 0))
                dom = int(st.session_state.get("sync_dom", 1))
                trigger = CronTrigger(day=dom, hour=int(getattr(t, 'hour', 2)), minute=int(getattr(t, 'minute', 0)))
            elif schedule_mode == "Every X minutes":
                trigger = IntervalTrigger(minutes=int(interval_minutes))
            else:
                trigger = CronTrigger.from_crontab(str(cron_expr))
            in_dir = sync_folder.strip()
            if not in_dir:
                st.error("Please provide a folder to sync")
            else:
                sched.add_job(_job_ingest_fn, id=job_id, args=[in_dir, md_out_target or None], trigger=trigger, replace_existing=True)
                st.success("Continuous sync enabled")
        except Exception as e:
            st.error(f"Failed to start: {e}")

    if stop_sync:
        try:
            if "ui_scheduler" in st.session_state:
                sched = st.session_state["ui_scheduler"]
                try:
                    sched.remove_job(job_id)
                except Exception:
                    pass
                st.success("Continuous sync stopped")
        except Exception as e:
            st.error(f"Failed to stop: {e}")

    if list_sync:
        try:
            _ensure_sched_started()
            sched = st.session_state["ui_scheduler"]
            jobs = sched.get_jobs()
            if not jobs:
                st.info("No scheduled jobs")
            else:
                for j in jobs:
                    st.caption(f"{j.id}: {j.trigger} next={j.next_run_time}")
        except Exception as e:
            st.error(f"Failed to list jobs: {e}")

    if remove_sync:
        try:
            _ensure_sched_started()
            sched = st.session_state["ui_scheduler"]
            try:
                sched.remove_job(job_id)
                st.success("Removed scheduled job")
            except Exception as e:
                st.error(f"Failed to remove: {e}")
        except Exception as e:
            st.error(f"Failed to remove: {e}")

    st.divider()

    def _friendly_title(path: str, chunk_name: str, content: str) -> str:
        # Prefer first Markdown heading if present
        try:
            for line in content.splitlines():
                s = line.strip()
                if s.startswith("# "):
                    return s[2:].strip()
                if s.startswith("## "):
                    return s[3:].strip()
                if s and not s.startswith("#") and len(s) > 8:
                    # fallback: first non-empty line with some length
                    return s[:80]
        except Exception:
            pass
        # Fallback to filename without noisy tokens
        name = Path(path).name
        base = name.rsplit(".", 1)[0]
        cleaned = base.replace("_", " ").replace("-", " ")
        cleaned = " ".join(part for part in cleaned.split() if not part.isupper() or len(part) <= 5)
        return cleaned.strip() or chunk_name or name

    # Controls
    colf, colp, colr = st.columns([4, 1, 1])
    with colf:
        filter_text = st.text_input("Filter (path, name, or content)", value=st.session_state.get("kb_m_filter", ""), key="kb_m_filter")
    with colp:
        rows_per_page = st.number_input("Rows/page", min_value=5, max_value=200, value=int(st.session_state.get("kb_m_rpp", 25)), step=5, key="kb_m_rpp")
    with colr:
        if "kb_m_page" not in st.session_state:
            st.session_state["kb_m_page"] = 1
        page = st.number_input("Page", min_value=1, value=int(st.session_state["kb_m_page"]), step=1)
        st.session_state["kb_m_page"] = int(page)

    like = f"%{filter_text}%" if filter_text else None
    total = count_chunks(engine, like)
    max_page = max(1, (total + int(rows_per_page) - 1) // int(rows_per_page))
    st.caption(f"{total} item(s) — page {st.session_state['kb_m_page']} of {max_page}")
    if st.session_state["kb_m_page"] > max_page:
        st.session_state["kb_m_page"] = max_page

    offset = (st.session_state["kb_m_page"] - 1) * int(rows_per_page)
    rows = fetch_chunks(engine, limit=int(rows_per_page), offset=int(offset), like=like)

    # Selection state
    if "kb_m_selected" not in st.session_state:
        st.session_state["kb_m_selected"] = []

    # Build dropdown options for this page (id -> label)
    page_options = []
    for rid, path, cname, content in rows:
        title = _friendly_title(path, cname, content)
        label = f"{title} — {Path(path).name} :: {cname}"
        page_options.append((rid, label))
    option_labels = {rid: label for rid, label in page_options}

    # Searchable multiselect (limited to current page for performance)
    selected_ids = st.multiselect(
        "Select items (current page)",
        options=[rid for rid, _ in page_options],
        default=[rid for rid in st.session_state["kb_m_selected"] if any(rid == rid2 for rid2, _ in page_options)],
        format_func=lambda rid: option_labels.get(rid, str(rid)),
        key="kb_m_multi",
    )
    st.session_state["kb_m_selected"] = list(selected_ids)

    # Auto preview first selected, with prev/next controls if multiple
    if st.session_state["kb_m_selected"]:
        sel_ids = st.session_state["kb_m_selected"]
        idx = st.session_state.get("kb_m_prev_idx", 0)
        idx = max(0, min(idx, len(sel_ids) - 1))
        cols_nav = st.columns([1, 3, 1])
        with cols_nav[0]:
            if st.button("◀ Prev", disabled=(idx <= 0)):
                st.session_state["kb_m_prev_idx"] = max(0, idx - 1)
                st.rerun()
        with cols_nav[2]:
            if st.button("Next ▶", disabled=(idx >= len(sel_ids) - 1)):
                st.session_state["kb_m_prev_idx"] = min(len(sel_ids) - 1, idx + 1)
                st.rerun()

        current_id = sel_ids[idx]
        row = fetch_chunk_by_id(engine, int(current_id))
        if row:
            _rid, path, cname, content = row
            st.caption(f"{path} :: {cname}")
            st.text_area("Content", content, height=360)

    del_count = len(st.session_state["kb_m_selected"])
    if st.button(f"Delete selected ({del_count})", type="primary", disabled=(del_count == 0)):
        try:
            n = delete_chunks_by_ids(engine, list(st.session_state["kb_m_selected"]))
            st.success(f"Deleted {n} item(s)")
            st.session_state["kb_m_selected"] = []
            st.session_state.pop("kb_m_prev_idx", None)
            # Invalidate search cache
            st.session_state.pop("kb_searcher", None)
            st.session_state.pop("kb_searcher_docs_len", None)
            st.rerun()
        except Exception as e:
            st.error(f"Delete failed: {e}")

 # legacy bottom search removed in favor of the top search bar

 # legacy bottom search removed in favor of the top search bar
