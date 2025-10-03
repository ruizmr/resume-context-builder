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
from kb.db import get_engine, fetch_all_chunks
from kb.search import HybridSearcher


st.set_page_config(page_title="Context Packager", layout="wide", initial_sidebar_state="collapsed")
st.title("Context Packager")

# Global search bar pinned at top
top_cols = st.columns([6, 1])
with top_cols[0]:
    q_top = st.text_input("Search knowledge base", value="", placeholder="Search...", label_visibility="collapsed")
with top_cols[1]:
    if st.button("Search", key="top_search_btn"):
        try:
            engine = get_engine()
            docs = fetch_all_chunks(engine)
            searcher = HybridSearcher()
            searcher.fit(docs)
            results = searcher.search(q_top, top_k=5)
            if not results:
                st.info("No results")
            else:
                for cid, score, path, cname, snippet, full_text in results:
                    st.markdown(f"**{path} :: {cname}** — score {score:.3f}")
                    st.caption("Lineage: original file path and chunk name shown above.")
                    st.code(snippet[: 400])
                    render_copy_button("Copy this chunk", full_text, height=80)
        except Exception as e:
            st.error(f"Search failed: {e}")

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

with st.sidebar:
    st.header("Settings")
    md_dir = st.text_input("Markdown output directory", value=default_md_dir)
    output_file = st.text_input("Context output file", value=default_out_file)
    attach_instructions = st.checkbox("Attach instructions", value=True)
    instruction_file = st.text_input("Instructions file (Markdown)", value=st.session_state.get("instruction_file", default_instr))
    header_text = st.text_input("Document header", value="Context Pack")
    max_tokens = st.number_input("Max tokens per chunk (0 = no split)", value=120000, min_value=0, step=1000)
    encoding_name = st.selectbox("Tokenizer", options=["o200k_base", "cl100k_base"], index=0)
    # moved include_kb next to build controls below
    st.divider()
    st.caption("KB search configuration")
    kb_top_k = st.number_input("Results to return", value=5, min_value=1, max_value=50, step=1)
    kb_snippet_len = st.number_input("Snippet length (chars)", value=400, min_value=50, max_value=4000, step=50)

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

st.subheader("Input")
uploaded_files = st.file_uploader("Upload files or ZIP", type=None, accept_multiple_files=True)
fallback_dir_main = st.text_input("Or enter a directory path (optional)", value="")

st.write("")
left, mid, right = st.columns([1,2,1])
with mid:
    include_kb = st.checkbox("Include in knowledge base (stateful)", value=False)
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

st.subheader("Knowledge base search")
q = st.text_input("Search query", value="")
if st.button("Search KB"):
    try:
        engine = get_engine()
        docs = fetch_all_chunks(engine)
        searcher = HybridSearcher()
        searcher.fit(docs)
        results = searcher.search(q, top_k=int(kb_top_k))
        if not results:
            st.info("No results")
        else:
            for cid, score, path, cname, snippet, full_text in results:
                st.markdown(f"**{path} :: {cname}** — score {score:.3f}")
                st.caption("Lineage: original file path and chunk name shown above.")
                st.code(snippet[: int(kb_snippet_len)])
                render_copy_button("Copy this chunk", full_text, height=80)
    except Exception as e:
        st.error(f"Search failed: {e}")


