# Resume Context Builder

Pipeline to convert resume PDFs to Markdown (MarkItDown), package into a single LLM-ready context (Repomix), and provide a Streamlit UI to copy/download the result.

## Install/Run via uv (recommended)

### macOS/Linux (minimal)
```bash
sh -c 'command -v uv >/dev/null 2>&1 || (curl -LsSf https://astral.sh/uv/install.sh | sh); PATH="$HOME/.local/bin:$PATH" uvx --python 3.12 --refresh --from git+https://github.com/ruizmr/resume-context-builder.git resume-ui'
```

### macOS/Linux (full MarkItDown extras)
```bash
sh -c 'command -v uv >/dev/null 2>&1 || (curl -LsSf https://astral.sh/uv/install.sh | sh); PATH="$HOME/.local/bin:$PATH" uvx --python 3.12 --refresh --from git+https://github.com/ruizmr/resume-context-builder.git?extra=full resume-ui'
```

### Windows PowerShell (minimal)
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -Command "if (-not (Get-Command uv -EA SilentlyContinue)) { iwr https://astral.sh/uv/install.ps1 -UseBasicParsing | iex }; $env:Path = \"$env:USERPROFILE\.local\bin;$env:Path\"; uvx --python 3.12 --refresh --from git+https://github.com/ruizmr/resume-context-builder.git resume-ui"
```

### Windows PowerShell (full MarkItDown extras)
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -Command "if (-not (Get-Command uv -EA SilentlyContinue)) { iwr https://astral.sh/uv/install.ps1 -UseBasicParsing | iex }; $env:Path = \"$env:USERPROFILE\.local\bin;$env:Path\"; uvx --python 3.12 --refresh --from git+https://github.com/ruizmr/resume-context-builder.git?extra=full resume-ui"
```

Use the sidebar to set directories and generate the packaged context. Download the final Markdown for copy-paste into your agent.

## Install a persistent shortcut (recommended)
Create a command shim so subsequent runs are just `resume-ui`.

### macOS/Linux (minimal)
```bash
sh -c 'command -v uv >/dev/null 2>&1 || (curl -LsSf https://astral.sh/uv/install.sh | sh); PATH="$HOME/.local/bin:$PATH" uv tool install --python 3.12 --force git+https://github.com/ruizmr/resume-context-builder.git && echo "Installed. Next time just run: resume-ui"'
```

### macOS/Linux (full MarkItDown extras)
```bash
sh -c 'command -v uv >/dev/null 2>&1 || (curl -LsSf https://astral.sh/uv/install.sh | sh); PATH="$HOME/.local/bin:$PATH" uv tool install --python 3.12 --force "git+https://github.com/ruizmr/resume-context-builder.git?extra=full" && echo "Installed. Next time just run: resume-ui"'
```

### Windows PowerShell (minimal)
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -Command "if (-not (Get-Command uv -EA SilentlyContinue)) { iwr https://astral.sh/uv/install.ps1 -UseBasicParsing | iex }; $env:Path = \"$env:USERPROFILE\.local\bin;$env:Path\"; uv tool install --python 3.12 --force git+https://github.com/ruizmr/resume-context-builder.git; Write-Host 'Installed. Next time just run: resume-ui'"
```

### Windows PowerShell (full MarkItDown extras)
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -Command "if (-not (Get-Command uv -EA SilentlyContinue)) { iwr https://astral.sh/uv/install.ps1 -UseBasicParsing | iex }; $env:Path = \"$env:USERPROFILE\.local\bin;$env:Path\"; uv tool install --python 3.12 --force \"git+https://github.com/ruizmr/resume-context-builder.git?extra=full\"; Write-Host 'Installed. Next time just run: resume-ui'"
```

Next runs after install:
```bash
resume-ui
```

Troubleshooting PATH
- macOS/Linux: ensure `~/.local/bin` is on PATH (e.g., add `export PATH="$HOME/.local/bin:$PATH"` to your shell rc).
- Windows: ensure `%USERPROFILE%\.local\bin` is on PATH, then open a new terminal.
