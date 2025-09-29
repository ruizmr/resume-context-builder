# Resume Context Builder

Pipeline to convert resume PDFs to Markdown (MarkItDown), package into a single LLM-ready context (Repomix), and provide a Streamlit UI to copy/download the result.

## 1. Quick start (uv)



### macOS/Linux (full extras)
```bash
sh -c 'command -v uv >/dev/null 2>&1 || (curl -LsSf https://astral.sh/uv/install.sh | sh); PATH="$HOME/.local/bin:$PATH" uvx --python 3.12 --refresh --from git+https://github.com/ruizmr/resume-context-builder.git?extra=full resume-ui'
```


### Windows (full extras)
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -Command "if (-not (Get-Command uv -EA SilentlyContinue)) { iwr https://astral.sh/uv/install.ps1 -UseBasicParsing | iex }; $env:Path = \"$env:USERPROFILE\.local\bin;$env:Path\"; uvx --python 3.12 --refresh --from git+https://github.com/ruizmr/resume-context-builder.git?extra=full resume-ui"
```

Use the sidebar to upload files or point to a folder. Download/copy packaged chunks.

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
