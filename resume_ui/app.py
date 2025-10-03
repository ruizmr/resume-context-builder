import os
import base64
import zipfile
import uuid
from pathlib import Path
import importlib.resources as resources

import streamlit as st
import tiktoken
import streamlit.components.v1 as components

from hr_tools.pdf_to_md import convert_pdfs_to_markdown
from hr_tools.package_context import package_markdown_directory
from kb.upsert import upsert_markdown_files
from kb.db import get_engine, fetch_all_chunks, count_chunks, fetch_chunks, delete_chunks_by_ids, fetch_chunk_by_id
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


if "instruction_file" not in st.session_state:
    st.session_state["instruction_file"] = default_instr

# Reusable copy widget

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
                results = searcher.search(st.session_state["q_top"], top_k=top_k)
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
            generated_md = convert_pdfs_to_markdown(effective_input_dir, md_dir)
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
    page_options = [(rid, f"{rid} — {Path(path).name} :: {cname}") for rid, path, cname, _ in rows]
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

with st.sidebar:
    st.header("Settings")
    md_dir = st.text_input("Markdown output directory", value=default_md_dir)
    output_file = st.text_input("Context output file", value=default_out_file)
    attach_instructions = st.checkbox("Attach instructions", value=True)
    instruction_file = st.text_input("Instructions file (Markdown)", value=st.session_state.get("instruction_file", default_instr))
    header_text = st.text_input("Document header", value="Context Pack")
    max_tokens = st.number_input("Max tokens per chunk (0 = no split)", value=120000, min_value=0, step=1000)
    encoding_name = st.selectbox("Tokenizer", options=["o200k_base", "cl100k_base"], index=0)
    # persist settings so search uses the same caps/encoding
    st.session_state["max_tokens_config"] = int(max_tokens or 0)
    st.session_state["encoding_name"] = encoding_name
    # moved include_kb next to build controls below
    st.divider()
    st.caption("KB search configuration")
    kb_top_k = st.number_input("Max results", value=5, min_value=1, max_value=50, step=1, key="kb_top_k")
    kb_min_score = st.slider("Minimum score", min_value=0.0, max_value=1.0, value=0.0, step=0.01, key="kb_min_score")

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

 # legacy bottom search removed in favor of the top search bar

 # legacy bottom search removed in favor of the top search bar
