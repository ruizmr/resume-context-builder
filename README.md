# Resume Context Builder

Pipeline to convert resume PDFs to Markdown (MarkItDown), package into a single LLM-ready context (Repomix), and provide a Streamlit UI to copy/download the result.

## Setup

```bash
python3 -m venv /home/overseer1/hr2/.venv
source /home/overseer1/hr2/.venv/bin/activate
pip install -r /home/overseer1/hr2/requirements.txt
```

## Usage (CLI snippets)

Convert PDFs to Markdown:

```bash
python -m hr_tools.pdf_to_md /home/overseer1/hr2/resume-dataset/data/data/CONSULTANT /home/overseer1/hr2/resume-dataset-md
```

Package Markdown with HR instructions:

```bash
python - << 'PY'
from hr_tools.package_context import package_markdown_directory
out = package_markdown_directory(
    "/home/overseer1/hr2/resume-dataset-md",
    "/home/overseer1/hr2/output/consultant_context.md",
    instruction_file="/home/overseer1/hr2/hr_tools/hr_instructions.md",
    header_text="HR Candidate Resume Pack",
)
print(out)
PY
```

## Streamlit UI

```bash
streamlit run /home/overseer1/hr2/app.py
```

## Install/Run via uv (minimal)

Minimal install (PDF and common formats) avoids `onnxruntime` on macOS:

```bash
uvx --from git+https://github.com/ruizmr/resume-context-builder.git resume-ui
```

To enable all MarkItDown converters (may require platform-specific deps), use the `full` extra:

```bash
uvx --from git+https://github.com/ruizmr/resume-context-builder.git?extra=full resume-ui
```

Use the sidebar to set directories and generate the packaged context. Download the final Markdown for copy-paste into your agent.
