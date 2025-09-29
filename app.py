import os
import base64
import zipfile
import tempfile
import uuid
import shutil
from pathlib import Path

import streamlit as st
import tiktoken
import streamlit.components.v1 as components
import time

from hr_tools.pdf_to_md import convert_pdfs_to_markdown
from hr_tools.package_context import package_markdown_directory


st.set_page_config(page_title="Resume Context Builder", layout="wide", initial_sidebar_state="collapsed")
st.title("Resume Context Builder")

default_pdf_dir = "/home/overseer1/hr2/resume-dataset/data/data/CONSULTANT"
default_md_dir = "/home/overseer1/hr2/resume-dataset-md"
default_out_file = "/home/overseer1/hr2/output/consultant_context.md"
default_instr = "/home/overseer1/hr2/hr_tools/hr_instructions.md"
uploads_root = "/home/overseer1/hr2/uploads"
Path(uploads_root).mkdir(parents=True, exist_ok=True)



with st.sidebar:
	st.header("Settings")
	md_dir = st.text_input("Output Markdown directory", value=default_md_dir)
	output_file = st.text_input("Packaged context output file", value=default_out_file)
	use_hr_instructions = st.checkbox("Include HR instructions", value=True)
	instruction_file = st.text_input("Instruction file path", value=default_instr)
	header_text = st.text_input("Header text", value="HR Candidate Resume Pack")
	max_tokens = st.number_input("Max tokens per file (0 = no split)", value=120000, min_value=0, step=1000)
	encoding_name = st.selectbox("Tokenizer/encoding", options=["o200k_base", "cl100k_base"], index=0)

st.subheader("Input")
uploaded_files = st.file_uploader("Upload ZIP or supported files", type=None, accept_multiple_files=True)
fallback_dir_main = st.text_input("Or enter a directory path (optional)", value="")

st.write("")
left, mid, right = st.columns([1,2,1])
with mid:
	start = st.button("Process and Package", type="primary", use_container_width=True)
	clear = st.button("Reset", use_container_width=True)

if clear:
	st.session_state.pop("context_content", None)
	st.session_state.pop("out_paths", None)
	st.session_state.pop("selected_chunk", None)

if start:
	effective_pdf_dir = None
	# Preferred: uploaded files
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
				# Assume regular file; write as-is (conversion handles supported types)
				(combined_dir / name).write_bytes(f.read())
		effective_pdf_dir = str(combined_dir.resolve())
	elif fallback_dir_main and os.path.isdir(fallback_dir_main):
		effective_pdf_dir = fallback_dir_main
	else:
		st.error("Please upload a ZIP/PDFs or provide a valid directory path.")
		st.stop()

	Path(md_dir).mkdir(parents=True, exist_ok=True)
	Path(Path(output_file).parent).mkdir(parents=True, exist_ok=True)

	with st.spinner("Converting PDFs to Markdown..."):
		generated_md = convert_pdfs_to_markdown(effective_pdf_dir, md_dir)
		st.success(f"Converted {len(generated_md)} markdown files.")

	with st.spinner("Packaging context with Repomix..."):
		instr_path = instruction_file if (use_hr_instructions and os.path.isfile(instruction_file)) else None
		# Unique output base to avoid overwriting previous runs
		run_tag = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
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
		st.success(f"Packaged {len(out_paths)} file(s)")
		# Load first for preview
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
		st.write("Generated files (with token counts):")
		for p, tok in rows:
			st.write(f"{p} — tokens: {tok}")

if "out_paths" in st.session_state and st.session_state["out_paths"]:
	st.subheader("Packaged Context")
	# Chunk selector
	options = [f"Part {i+1}: {Path(p).name}" for i, p in enumerate(st.session_state["out_paths"])]
	idx = st.session_state.get("selected_chunk", 0)
	idx = st.selectbox("Select chunk", options=list(range(len(options))), format_func=lambda i: options[i], index=min(idx, len(options)-1))
	st.session_state["selected_chunk"] = idx
	selected_path = st.session_state["out_paths"][idx]
	selected_content = Path(selected_path).read_text(encoding="utf-8")

	# Download and Copy buttons
	col_dl, col_cp = st.columns(2)
	with col_dl:
		st.download_button(
			label=f"Download {Path(selected_path).name}",
			data=selected_content,
			file_name=Path(selected_path).name,
			mime="text/markdown",
			use_container_width=True,
		)
	with col_cp:
		b64 = base64.b64encode(selected_content.encode("utf-8")).decode("ascii")
		html = f"""
		<style>
		.copy-wrap {{ position: relative; width: 100%; }}
		.copy-btn {{
		  all: unset; display: inline-block; width: 100%; text-align: center;
		  padding: 0.6rem 1rem; border-radius: 0.25rem; background-color: #F63366;
		  color: #fff; cursor: pointer; font-weight: 600; box-shadow: rgba(0,0,0,0.1) 0 1px 2px;
		  transition: filter .15s ease-in-out, background-color .15s ease-in-out;
		}}
		.copy-btn:hover {{ filter: brightness(0.95); }}
		.copy-btn.copied {{ background-color: #22c55e; }} /* green */
		.toast {{
		  position: absolute; top: -2.2rem; right: 0.25rem; background: #22c55e; color: #fff;
		  padding: 4px 10px; border-radius: 6px; font-size: 0.85rem; display: inline-flex; align-items: center;
		  gap: 6px; opacity: 0; transform: translateY(4px); transition: opacity .18s, transform .18s;
		}}
		.toast.show {{ opacity: 1; transform: translateY(0); }}
		.toast .check {{ font-weight: 900; }}
		</style>
		<script>
		async function copyChunk(){{
		  const text = atob('{b64}');
		  try {{ await navigator.clipboard.writeText(text); }} catch(e) {{
		    const ta = document.createElement('textarea'); ta.value = text; document.body.appendChild(ta);
		    ta.focus(); ta.select(); try {{ document.execCommand('copy'); }} catch(e2) {{}} document.body.removeChild(ta);
		  }}
		  const toast = document.getElementById('copy-toast');
		  const btn = document.getElementById('copy-btn');
		  if (btn) {{
		    const original = btn.textContent;
		    btn.classList.add('copied'); btn.textContent = 'Copied!';
		    setTimeout(()=>{{ btn.classList.remove('copied'); btn.textContent = original; }}, 1200);
		  }}
		  if (toast) {{ toast.classList.add('show'); setTimeout(()=> toast.classList.remove('show'), 1200); }}
		}}
		</script>
		<div class=\"copy-wrap\">
		  <button id=\"copy-btn\" class=\"copy-btn\" onclick=\"copyChunk()\">Copy to clipboard</button>
		  <span id=\"copy-toast\" class=\"toast\"><span class=\"check\">✓</span> Copied to clipboard</span>
		</div>
		"""
		components.html(html, height=110)

	st.text_area("Preview", selected_content, height=400)


