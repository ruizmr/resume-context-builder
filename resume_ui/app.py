import os
import hashlib
import base64
import zipfile
import shutil
import json
import uuid
from pathlib import Path
import importlib.resources as resources
from datetime import time as dt_time, datetime, timedelta, timezone

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


# Upload safety limits
MAX_UPLOAD_FILE_BYTES = 512 * 1024 * 1024  # 512MB per file
MAX_ZIP_FILES = 2000
MAX_ZIP_TOTAL_BYTES = 200 * 1024 * 1024  # 200MB cumulative uncompressed


def _is_within_directory(base: Path, target: Path) -> bool:
    try:
        base = base.resolve()
        target = target.resolve()
        return os.path.commonpath([str(base), str(target)]) == str(base)
    except Exception:
        return False


def _safe_extract_zip(zip_path: Path, target_dir: Path) -> int:
    extracted = 0
    total_bytes = 0
    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            for info in zf.infolist():
                # Skip directories
                if getattr(info, "is_dir", None) and info.is_dir():
                    continue
                # Basic filename sanitation
                member = info.filename or ""
                if not member or member.endswith("/"):
                    continue
                # Per-file size cap
                if getattr(info, "file_size", 0) > MAX_UPLOAD_FILE_BYTES:
                    continue
                # Cumulative size cap
                nxt = total_bytes + int(getattr(info, "file_size", 0) or 0)
                if nxt > MAX_ZIP_TOTAL_BYTES:
                    break
                # Path traversal guard
                dest = (target_dir / member)
                if not _is_within_directory(target_dir, dest):
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                # Extract safely
                with zf.open(info, "r") as src, open(dest, "wb") as out:
                    while True:
                        chunk = src.read(1024 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)
                total_bytes = nxt
                extracted += 1
                if extracted >= MAX_ZIP_FILES:
                    break
    except Exception:
        # Best-effort: partial extraction allowed, caller handles missing files
        pass
    return extracted


# Text normalization helpers
def _normalize_mojibake(text: str) -> str:
    """Best-effort fix for common PDF→MD mojibake (e.g., â, â) and stray NBSP.

    Strategy:
    - If suspicious Windows-1252/Latin-1 artifacts are present, attempt a
      single latin1→utf-8 round-trip which repairs sequences like â → —.
    - Then normalize non-breaking spaces and zero-width chars.
    """
    s = text or ""
    # Only attempt re-decode when artifacts commonly appear
    if ("Â" in s) or ("Ã" in s) or ("â" in s):
        try:
            s = s.encode("latin1").decode("utf-8")
        except Exception:
            # Keep original on failure
            s = s
    # Replace non-breaking spaces with regular spaces
    try:
        s = s.replace("\u00A0", " ")
    except Exception:
        pass
    # Strip zero-width and BOM if present
    for zw in ("\u200b", "\ufeff"):
        try:
            s = s.replace(zw, "")
        except Exception:
            pass
    return s

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
    "kb_use_ann",
    "kb_cand_mult",
    "kb_bm25_weight",
    "kb_lsa_weight",
    "kb_tfidf_metric",
    "kb_ann_weight",
    "kb_mmr_diversify",
    "kb_mmr_lambda",
    "kb_phrase_boost",
    "kb_enable_rare_filter",
    "kb_rare_idf_threshold",
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

# Ensure defaults exist in session state for sidebar widgets to avoid value/key conflicts
_defaults_map = {
    "md_dir": default_md_dir,
    "output_file": default_out_file,
    "attach_instructions": True,
    "instruction_file": default_instr,
    "header_text": "Context Pack",
    "max_tokens": 120000,
    "encoding_name": "o200k_base",
    "kb_top_k": 5,
    "kb_min_score": 0.005,
	"kb_neighbors": 0,
    "kb_sequence": True,
    "kb_use_ann": True,
    "kb_cand_mult": 5,
    "kb_bm25_weight": 0.45,
    "kb_lsa_weight": 0.2,
    "kb_tfidf_metric": "cosine",
    "kb_ann_weight": 0.15,
    "kb_mmr_diversify": True,
    "kb_mmr_lambda": 0.2,
    "kb_phrase_boost": 0.1,
    "kb_enable_rare_filter": True,
    "kb_rare_idf_threshold": 3.0,
}
for _k, _v in _defaults_map.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# Reusable copy widget

with st.sidebar:
    st.header("Settings")
    md_dir = st.text_input("Markdown output directory", key="md_dir")
    output_file = st.text_input("Context output file", key="output_file")
    attach_instructions = st.checkbox("Attach instructions", key="attach_instructions")
    instruction_file = st.text_input("Instructions file (Markdown)", key="instruction_file")
    header_text = st.text_input("Document header", key="header_text")
    max_tokens = st.number_input("Max tokens per chunk (0 = no split)", min_value=0, step=1000, key="max_tokens")
    encoding_name = st.selectbox("Tokenizer", options=["o200k_base", "cl100k_base"], key="encoding_name")
    # persist settings so search uses the same caps/encoding
    st.session_state["max_tokens_config"] = int(max_tokens or 0)
    # moved include_kb next to build controls below
    st.divider()
    if st.button("Save settings", use_container_width=True):
        try:
            payload = {k: st.session_state.get(k) for k in PERSIST_KEYS}
            _save_settings(payload)
            st.success("Settings saved")
        except Exception as e:
            st.error(f"Failed to save settings: {e}")
    st.caption("KB search configuration")
    kb_top_k = st.number_input("Max results", min_value=1, max_value=50, step=1, key="kb_top_k")
    kb_min_score = st.slider("Minimum score", min_value=0.0, max_value=1.0, step=0.005, key="kb_min_score")
    kb_neighbors = st.number_input("Neighbors to include (per match)", min_value=0, max_value=10, step=1, key="kb_neighbors")
    kb_sequence = st.checkbox("Preserve document sequence (group by file, in order)", key="kb_sequence")
    # Advanced retrieval tuning
    kb_use_ann = st.checkbox("Use ANN (NN-Descent) for candidates", key="kb_use_ann")
    kb_cand_mult = st.number_input("Candidate multiplier", min_value=1, max_value=20, step=1, key="kb_cand_mult")
    kb_bm25_weight = st.slider("BM25 weight (0=TF-IDF, 1=BM25)", min_value=0.0, max_value=1.0, step=0.05, key="kb_bm25_weight")
    kb_lsa_weight = st.slider("LSA weight (SVD cosine)", min_value=0.0, max_value=1.0, step=0.05, key="kb_lsa_weight")
    kb_tfidf_metric = st.selectbox("TF-IDF similarity metric", options=["cosine", "l2"], key="kb_tfidf_metric")
    kb_ann_weight = st.slider("ANN (NN-Descent) weight", min_value=0.0, max_value=1.0, step=0.05, key="kb_ann_weight")
    mmr_div = st.checkbox("Diversify results (MMR)", key="kb_mmr_diversify")
    mmr_lambda = st.slider("MMR lambda (relevance vs diversity)", min_value=0.0, max_value=1.0, step=0.05, key="kb_mmr_lambda")
    phrase_boost = st.slider("Quoted phrase boost", min_value=0.0, max_value=0.5, step=0.05, key="kb_phrase_boost")
    enable_rare = st.checkbox("Filter candidates by rare terms", key="kb_enable_rare_filter")
    rare_idf_th = st.number_input("Rare term IDF threshold", min_value=0.0, max_value=20.0, step=0.5, key="kb_rare_idf_threshold")

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

def render_copy_button(label: str, text: str, height: int = 110, disabled: bool = False, help_text: str | None = None):
    b64 = base64.b64encode(text.encode("utf-8")).decode("ascii")
    uid = uuid.uuid4().hex[:8]
    title = help_text or ""
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
    .copy-btn:disabled { background-color: #e5e7eb; color: #9ca3af; cursor: not-allowed; }
    .toast {
      position: absolute; top: -2.2rem; right: 0.25rem; background: #22c55e; color: #fff;
      padding: 4px 10px; border-radius: 6px; font-size: 0.85rem; display: inline-flex; align-items: center;
      gap: 6px; opacity: 0; transform: translateY(4px); transition: opacity .18s, transform .18s;
    }
    .toast.show { opacity: 1; transform: translateY(0); }
    .toast .check { font-weight: 900; }
    </style>
    <script>
    async function copyGeneric_{{UID}}(){
      const text = atob('{{B64}}');
      try { await navigator.clipboard.writeText(text); } catch(e) {
        const ta = document.createElement('textarea'); ta.value = text; document.body.appendChild(ta);
        ta.focus(); ta.select(); try { document.execCommand('copy'); } catch(e2) {} document.body.removeChild(ta);
      }
      const toast = document.getElementById('copy-toast-{{UID}}');
      const btn = document.getElementById('copy-btn-{{UID}}');
      if (btn) {
        const original = btn.textContent;
        btn.classList.add('copied'); btn.textContent = 'Copied!';
        setTimeout(()=>{ btn.classList.remove('copied'); btn.textContent = original; }, 1200);
      }
      if (toast) { toast.classList.add('show'); setTimeout(()=> toast.classList.remove('show'), 1200); }
    }
    </script>
    <div class="copy-wrap">
      <button id="copy-btn-{{UID}}" class="copy-btn" {{DISABLED}} onclick="copyGeneric_{{UID}}()" title="{{TITLE}}" aria-label="{{LABEL}}">{{LABEL}}</button>
      <span id="copy-toast-{{UID}}" class="toast"><span class="check">✓</span> Copied to clipboard</span>
    </div>
    """
    html = (
        tmpl
        .replace("{{B64}}", b64)
        .replace("{{LABEL}}", label)
        .replace("{{UID}}", uid)
        .replace("{{TITLE}}", title)
        .replace("{{DISABLED}}", "disabled" if disabled else "")
    )
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
                # Effective threshold with base floor (prevents persisted 0.0 from leaking)
                _saved_ms = float(st.session_state.get("kb_min_score", 0.005))
                min_score = max(0.005, _saved_ms)
                results = searcher.search(
                    st.session_state["q_top"],
                    top_k=top_k,
                    neighbors=int(st.session_state.get("kb_neighbors", 0)),
                    sequence=bool(st.session_state.get("kb_sequence", True)),
                    use_ann=bool(st.session_state.get("kb_use_ann", True)),
                    bm25_weight=float(st.session_state.get("kb_bm25_weight", 0.6)),
                    cand_multiplier=int(st.session_state.get("kb_cand_mult", 3)),
                    lsa_weight=float(st.session_state.get("kb_lsa_weight", 0.2)),
                    tfidf_metric=str(st.session_state.get("kb_tfidf_metric", "cosine")),
                    ann_weight=float(st.session_state.get("kb_ann_weight", 0.0)),
                    min_score=float(min_score),
                    mmr_diversify=bool(st.session_state.get("kb_mmr_diversify", True)),
                    mmr_lambda=float(st.session_state.get("kb_mmr_lambda", 0.2)),
                    phrase_boost=float(st.session_state.get("kb_phrase_boost", 0.1)),
                    enable_rare_term_filter=bool(st.session_state.get("kb_enable_rare_filter", True)),
                    rare_idf_threshold=float(st.session_state.get("kb_rare_idf_threshold", 3.0)),
                )
                # filter by minimum score (defensive)
                results = [r for r in results if r[1] >= min_score]
                # Treat as empty if best remaining score is below threshold
                if results:
                    try:
                        best = max(r[1] for r in results)
                        if best < float(min_score):
                            results = []
                    except Exception:
                        pass

                if not results:
                    st.session_state["kb_results_agg"] = ""
                    st.session_state["kb_results_list"] = []
                else:
                    sections = []
                    for cid, score, path, cname, snippet, full_text in results:
                        header = f"{path} :: {cname} — relevancy score {score:.3f}"
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
            # Collapsible, scrollable panels per result with selection checkboxes
            results_list = st.session_state.get("kb_results_list") or []
            selected_map = st.session_state.get("kb_results_selected") or {}
            # Build aggregated text from only selected items BEFORE rendering list so the button sits above
            if results_list:
                selected_results = []
                for cid, score, path, cname, snippet, full_text in results_list:
                    sel_key = f"kb_sel_{cid}"
                    sel_val = bool(st.session_state.get(sel_key, selected_map.get(cid, True)))
                    if sel_val:
                        selected_results.append((cid, score, path, cname, snippet, full_text))
                if selected_results:
                    sections = []
                    for cid, score, path, cname, snippet, full_text in selected_results:
                        header = f"{path} :: {cname} — relevancy score {score:.3f}"
                        sections.append(f"{header}\n\n{full_text.strip()}")
                    aggregated_sel = "\n\n---\n\n".join(sections)
                    # Enforce token cap similar to packaging
                    try:
                        max_tok = int(st.session_state.get("max_tokens_config") or 0)
                        enc_name = st.session_state.get("encoding_name", "o200k_base")
                        if max_tok and max_tok > 0:
                            enc = tiktoken.get_encoding(enc_name)
                            toks = enc.encode(aggregated_sel)
                            if len(toks) > max_tok:
                                aggregated_sel = enc.decode(toks[:max_tok])
                    except Exception:
                        aggregated_sel = aggregated_sel
                else:
                    aggregated_sel = ""
                # Copy selected button ABOVE the results list with selected count and tooltip
                sel_count = len(selected_results)
                total_count = len(results_list)
                copy_cols = st.columns([2, 1])
                with copy_cols[0]:
                    render_copy_button(
                        "Copy all selected results",
                        aggregated_sel,
                        height=80,
                        disabled=(sel_count == 0),
                        help_text="Copies only the selected results",
                    )
                with copy_cols[1]:
                    st.caption(f"Selected {sel_count} of {total_count}")
                st.caption("Select which results to include below.")

                # Now render list with checkboxes and update selection map (revert to previous full expander per item)
                for i, (cid, score, path, cname, snippet, full_text) in enumerate(results_list):
                    cols = st.columns([1, 24])
                    with cols[0]:
                        sel = st.checkbox("", value=bool(selected_map.get(cid, True)), key=f"kb_sel_{cid}")
                        selected_map[cid] = bool(sel)
                    with cols[1]:
                        header = f"{path} :: {cname} — relevancy score {score:.3f}"
                        with st.expander(header, expanded=(i == 0)):
                            st.markdown(f"**{path} :: {cname} — relevancy score {score:.3f}**\n\n{full_text}")
                # Persist selection state
                st.session_state["kb_results_selected"] = dict(selected_map)
            else:
                # Fallback to aggregated rendering
                st.markdown(st.session_state["kb_results_agg"])
        else:
            st.info("No search results found")

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
        _temp_root: Path | None = None
        try:
            if uploaded_files:
                _temp_root = Path(uploads_root) / f"session-{uuid.uuid4().hex[:8]}"
                _temp_root.mkdir(parents=True, exist_ok=True)
                combined_dir = _temp_root / "combined"
                combined_dir.mkdir(parents=True, exist_ok=True)
                total_uploaded = 0
                files_written = 0
                for f in uploaded_files:
                    name = Path(f.name).name
                    raw = f.read() if not name.lower().endswith(".zip") else None
                    if raw is not None:
                        # Per-file size limit
                        if len(raw) > MAX_UPLOAD_FILE_BYTES:
                            st.warning(f"Skipping {name}: file too large")
                            continue
                        total_uploaded += len(raw)
                        if total_uploaded > MAX_ZIP_TOTAL_BYTES:
                            st.warning("Upload total size exceeded limit; skipping remaining files")
                            break
                    if name.lower().endswith(".zip"):
                        zip_path = _temp_root / name
                        zip_path.write_bytes(f.read())
                        _ = _safe_extract_zip(zip_path, combined_dir)
                    else:
                        (combined_dir / name).write_bytes(raw or b"")
                    files_written += 1
                if files_written == 0:
                    st.error("No acceptable files were uploaded (size/type limits)")
                    st.stop()
                effective_input_dir = str(combined_dir.resolve())
            elif fallback_dir_main and os.path.isdir(fallback_dir_main):
                effective_input_dir = fallback_dir_main
            else:
                st.error("Please upload files/ZIP or provide a valid directory path.")
                st.stop()

        except Exception as e:
            # Ensure the try block is properly closed and surface a friendly error
            try:
                st.error(f"Upload processing failed: {e}")
            except Exception:
                pass
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
            # One-time post-processing: normalize mojibake in generated Markdown before storage/packaging
            try:
                normalized_files = 0
                for p in list(generated_md):
                    try:
                        txt = Path(p).read_text(encoding="utf-8")
                        fixed = _normalize_mojibake(txt)
                        if fixed != txt:
                            Path(p).write_text(fixed, encoding="utf-8")
                            normalized_files += 1
                    except Exception:
                        # Continue best-effort on individual file errors
                        pass
                if normalized_files > 0:
                    st.caption(f"Normalized text encoding in {normalized_files} file(s)")
            except Exception:
                # Non-fatal: proceed even if normalization step fails
                pass
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
        # Cleanup temporary upload directory
        try:
            if _temp_root and _temp_root.exists():
                shutil.rmtree(_temp_root, ignore_errors=True)
        except Exception:
            pass

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
            render_copy_button("Copy to clipboard", selected_content, height=110, help_text="Copies the current previewed chunk")

        # Render packaged context as Markdown in a collapsible panel
        with st.expander("Preview (Markdown)", expanded=True):
            st.markdown(selected_content)

with manage_tab:
    st.subheader("Manage knowledge base")
    engine = get_engine()
    st.divider()
    st.subheader("Enable continuous folder sync")

    # Redesigned scheduler: single add-job form + run-now
    with st.form("add_job_form", clear_on_submit=False):
        col_a, col_b = st.columns([3, 2])
        with col_a:
            sync_folder = st.text_input("Folder to sync", key="sync_folder", placeholder="/path/to/folder")
            if sync_folder and not os.path.isdir(sync_folder):
                st.caption("Invalid folder path")
            md_out_target = st.text_input("Markdown output dir (optional)", key="sync_md_out")
            if md_out_target and not os.path.isdir(md_out_target):
                st.caption("Invalid folder path")
        with col_b:
            schedule_mode = st.selectbox("Schedule", options=["Daily", "Every X minutes", "Weekly", "Monthly", "Cron (advanced)"], key="sync_mode")
            interval_minutes = int(st.number_input("Interval (minutes)", min_value=1, step=1, key="sync_interval")) if schedule_mode == "Every X minutes" else 1440
            cron_expr = st.text_input("Cron (min hr dom mon dow)", key="sync_cron") if schedule_mode == "Cron (advanced)" else st.session_state.get("sync_cron", "0 2 * * *")
            if schedule_mode == "Daily":
                st.time_input("Time of day", key="sync_time")
            elif schedule_mode == "Weekly":
                days_options = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
                st.multiselect("Days of week", options=days_options, key="sync_weekdays")
                st.time_input("Time of day", key="sync_time_weekly")
            elif schedule_mode == "Monthly":
                st.number_input("Day of month", min_value=1, max_value=31, step=1, key="sync_dom")
                st.time_input("Time of day", key="sync_time_monthly")
        col1, col2 = st.columns([1,1])
        add_job_submit = col1.form_submit_button("Add job", use_container_width=True)
        run_now_submit = col2.form_submit_button("Run now", use_container_width=True)

    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.executors.pool import ThreadPoolExecutor
    from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
    from apscheduler.triggers.interval import IntervalTrigger
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.date import DateTrigger
    from resume_ui.scheduler_cli import _job_ingest as _job_ingest_fn
    from kb.db import ensure_job, fetch_recent_runs, request_cancel_run

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
    if add_job_submit:
        try:
            _ensure_sched_started()
            sched = st.session_state["ui_scheduler"]
            trigger = None
            if schedule_mode == "Daily":
                t = st.session_state.get("sync_time", dt_time(2, 0))
                trigger = CronTrigger(hour=int(getattr(t, 'hour', 2)), minute=int(getattr(t, 'minute', 0)))
                human_sched = f"Daily at {int(getattr(t,'hour',2)):02d}:{int(getattr(t,'minute',0)):02d}"
            elif schedule_mode == "Weekly":
                t = st.session_state.get("sync_time_weekly", dt_time(2, 0))
                days = st.session_state.get("sync_weekdays", ["Mon"]) or ["Mon"]
                dow_map = {"Mon": "mon", "Tue": "tue", "Wed": "wed", "Thu": "thu", "Fri": "fri", "Sat": "sat", "Sun": "sun"}
                day_str = ",".join([dow_map.get(d, "mon") for d in days])
                trigger = CronTrigger(day_of_week=day_str, hour=int(getattr(t, 'hour', 2)), minute=int(getattr(t, 'minute', 0)))
                human_sched = f"Weekly on {', '.join(days)} at {int(getattr(t,'hour',2)):02d}:{int(getattr(t,'minute',0)):02d}"
            elif schedule_mode == "Monthly":
                t = st.session_state.get("sync_time_monthly", dt_time(2, 0))
                dom = int(st.session_state.get("sync_dom", 1))
                trigger = CronTrigger(day=dom, hour=int(getattr(t, 'hour', 2)), minute=int(getattr(t, 'minute', 0)))
                human_sched = f"Monthly on day {dom} at {int(getattr(t,'hour',2)):02d}:{int(getattr(t,'minute',0)):02d}"
            elif schedule_mode == "Every X minutes":
                trigger = IntervalTrigger(minutes=int(interval_minutes))
                human_sched = f"Every {int(interval_minutes)} minute(s)"
            else:
                trigger = CronTrigger.from_crontab(str(cron_expr))
                human_sched = f"Cron: {cron_expr}"
            # Implied SLA fixed to 5 minutes
            implied_sla_min = 5

            in_dir = sync_folder.strip()
            if not in_dir:
                st.error("Please provide a folder to sync")
            else:
                base = Path(in_dir).name or "folder"
                jid = f"sync::{hashlib.sha1(in_dir.encode('utf-8')).hexdigest()[:8]}::{base}"
                jname = f"{base} — {human_sched}"
                ensure_job(jid, jname, in_dir, md_out_target or None, implied_sla_min)
                sched.add_job(_job_ingest_fn, id=jid, name=jname, args=[in_dir, md_out_target or None, jid, jname], trigger=trigger, replace_existing=True)
                st.success("Continuous sync enabled")
        except Exception as e:
            st.error(f"Failed to start: {e}")

    if run_now_submit:
        try:
            in_dir = sync_folder.strip()
            if not in_dir or not os.path.isdir(in_dir):
                st.error("Please provide a valid folder to sync")
            else:
                _ensure_sched_started()
                sched = st.session_state["ui_scheduler"]
                jid = f"run::{hashlib.sha1((in_dir + str(uuid.uuid4())).encode('utf-8')).hexdigest()[:8]}"
                # For one-off runs, SLA is not applicable; store None
                ensure_job(jid, f"One-off run — {Path(in_dir).name}", in_dir, md_out_target or None, None)
                sched.add_job(_job_ingest_fn, id=jid, args=[in_dir, md_out_target or None, jid, f"One-off run — {Path(in_dir).name}"], trigger=DateTrigger(run_date=datetime.now(timezone.utc) + timedelta(seconds=1)))
                st.success("Queued run in background")
        except Exception as e:
            st.error(f"Run failed: {e}")

    # Always show current jobs with human-friendly labels and selection, wrapped in a form to prevent flicker
    try:
        _ensure_sched_started()
        sched = st.session_state["ui_scheduler"]
        jobs = sched.get_jobs()
        if not jobs:
            st.info("No scheduled jobs")
        else:
            labels = {}
            options = []
            for j in jobs:
                folder = None
                try:
                    if j.args and len(j.args) >= 1 and isinstance(j.args[0], str):
                        folder = Path(j.args[0]).name
                except Exception:
                    folder = None
                label = j.name or f"{folder or j.id} — {str(j.trigger)}"
                if j.next_run_time:
                    label = f"{label} — next {j.next_run_time}"
                labels[j.id] = label
                options.append(j.id)
            st.caption("Scheduled jobs")
            with st.form("jobs_form", clear_on_submit=False):
                selected_jobs = st.multiselect("Select jobs", options=options, format_func=lambda jid: labels.get(jid, jid), key="ui_jobs_multi")
                col_rm, col_run_sel = st.columns([1,1])
                rm_pressed = col_rm.form_submit_button("Remove selected", disabled=(not selected_jobs), use_container_width=True)
                run_pressed = col_run_sel.form_submit_button("Run selected now", disabled=(not selected_jobs), use_container_width=True)
                purge_pressed = st.form_submit_button("Purge all jobs", use_container_width=True)
                if rm_pressed:
                    removed = 0
                    for jid in list(selected_jobs):
                        try:
                            sched.remove_job(jid)
                            removed += 1
                        except Exception:
                            pass
                    if removed > 0:
                        st.success(f"Removed {removed} job(s)")
                    else:
                        st.info("No jobs removed")
                if run_pressed:
                    queued = 0
                    for jid in list(selected_jobs):
                        try:
                            j = sched.get_job(jid)
                            if j and j.args and isinstance(j.args[0], str):
                                in_dir = j.args[0]
                                out_dir = j.args[1] if len(j.args) > 1 else None
                                rqid = f"run::{hashlib.sha1((in_dir + str(uuid.uuid4())).encode('utf-8')).hexdigest()[:8]}"
                                ensure_job(rqid, f"One-off run — {Path(in_dir).name}", in_dir, out_dir)
                                sched.add_job(_job_ingest_fn, id=rqid, args=[in_dir, out_dir, rqid, f"One-off run — {Path(in_dir).name}"], trigger=DateTrigger(run_date=datetime.now(timezone.utc) + timedelta(seconds=1)))
                                queued += 1
                        except Exception:
                            pass
                    if queued > 0:
                        st.success(f"Queued {queued} job(s)")
                    else:
                        st.info("No jobs queued")
                if purge_pressed:
                    try:
                        for jid in list(options):
                            try:
                                sched.remove_job(jid)
                            except Exception:
                                pass
                        st.success("Purged all jobs")
                    except Exception as e:
                        st.error(f"Failed to purge: {e}")
    except Exception as e:
        st.error(f"Failed to list/remove jobs: {e}")

    # Removed single remove button; handled via multiselect above

    st.divider()
    st.subheader("Recent runs")
    if st.button("Refresh", use_container_width=False):
        st.rerun()
    try:
        runs = fetch_recent_runs(limit=20)
        if not runs:
            st.caption("No runs yet")
        else:
            for rid, jid, in_dir, out_dir, status, progress, processed, total, chunks, started_at, finished_at, last_msg, log_tail, cancel_req, err in runs:
                with st.container():
                    st.markdown(f"**{status.upper()}** — {progress}% — {Path(in_dir).name}  ")
                    st.caption(f"Run {rid} — job {jid or '-'} — started {started_at}{' — finished ' + str(finished_at) if finished_at else ''}")
                    cols = st.columns([3,1,1])
                    with cols[0]:
                        st.progress(int(progress or 0), text=last_msg or "")
                        if log_tail:
                            with st.expander("View recent log"):
                                st.code(str(log_tail))
                    with cols[1]:
                        if status in ("running", "queued") and not cancel_req:
                            if st.button("Cancel", key=f"cancel_{rid}", use_container_width=True):
                                try:
                                    request_cancel_run(rid)
                                    st.success("Cancellation requested")
                                except Exception as e:
                                    st.error(f"Cancel failed: {e}")
                    with cols[2]:
                        st.caption(f"Files {processed}/{total}")
                        if chunks is not None:
                            st.caption(f"Chunks {int(chunks)}")
                        if err:
                            st.caption(f"Error: {err}")
    except Exception as e:
        st.error(f"Failed to load runs: {e}")
    st.subheader("Edit and delete knowledge")

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
        filter_text = st.text_input("Filter (path, name, or content)", key="kb_m_filter")
    with colp:
        rows_per_page = st.number_input("Rows/page", min_value=5, max_value=200, step=5, key="kb_m_rpp")
    with colr:
        if "kb_m_page" not in st.session_state:
            st.session_state["kb_m_page"] = 1
        page = st.number_input("Page", min_value=1, value=int(st.session_state["kb_m_page"]), step=1)
        st.session_state["kb_m_page"] = int(page)

    like = f"%{st.session_state.get('kb_m_filter','')}%" if st.session_state.get('kb_m_filter') else None
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

    # Searchable multiselect (limited to current page for performance), wrapped in a form to reduce reruns
    with st.form("kb_manage_form", clear_on_submit=False):
        selected_ids = st.multiselect(
            "Select items (current page)",
            options=[rid for rid, _ in page_options],
            default=[rid for rid in st.session_state.get("kb_m_selected", []) if any(rid == rid2 for rid2, _ in page_options)],
            format_func=lambda rid: option_labels.get(rid, str(rid)),
            key="kb_m_multi",
        )
        submit_del = st.form_submit_button("Apply selection")
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
    with st.form("kb_delete_form", clear_on_submit=False):
        btn_del = st.form_submit_button(f"Delete selected ({del_count})", disabled=(del_count == 0))
    if btn_del:
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

    st.divider()
    st.subheader("Danger zone")
    st.caption("Factory reset DB: wipes jobs and KB tables. This cannot be undone.")
    with st.form("factory_reset_form", clear_on_submit=False):
        confirm = st.text_input("Type RESET to confirm", key="factory_reset_confirm")
        do_reset = st.form_submit_button("Factory reset DB")
    if do_reset:
        if (st.session_state.get("factory_reset_confirm", "").strip().upper() != "RESET"):
            st.error("Please type RESET to confirm.")
        else:
            try:
                from kb.db import factory_reset_db
                factory_reset_db()
                st.success("Database reset complete")
            except Exception as e:
                st.error(f"Reset failed: {e}")
