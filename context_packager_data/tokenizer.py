from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def _ensure_cache_dir() -> str:
    """Resolve and set TIKTOKEN_CACHE_DIR to our bundled data if not already set.

    This prevents runtime network fetches in restricted environments by pointing
    tiktoken at a packaged cache directory containing `.tiktoken` files.
    """
    env_key = "TIKTOKEN_CACHE_DIR"
    if os.getenv(env_key):
        return os.environ[env_key]
    pkg_cache = Path(__file__).parent / "tiktoken_cache"
    # Create directory if missing so callers can drop files at runtime too
    try:
        pkg_cache.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    os.environ[env_key] = str(pkg_cache)
    return os.environ[env_key]


def get_encoding(encoding_name: str = "o200k_base"):
    """Return a tiktoken encoding using the offline cache directory.

    Falls back to `cl100k_base` if the requested encoding is unavailable.
    """
    _ensure_cache_dir()
    import tiktoken  # import after setting env var

    try:
        return tiktoken.get_encoding(encoding_name)
    except Exception:
        # Fallback to a widely available encoding if requested one is missing
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            # As a last resort, raise the original error
            return tiktoken.get_encoding(encoding_name)


def prefer_offline_cache(path: Optional[str] = None) -> str:
    """Optionally override cache dir to a custom path and return the active path."""
    if path:
        os.environ["TIKTOKEN_CACHE_DIR"] = str(Path(path).expanduser().resolve())
    return _ensure_cache_dir()


