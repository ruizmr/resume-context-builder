#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target_dir = repo_root / "context_packager_data" / "tiktoken_cache"
    target_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TIKTOKEN_CACHE_DIR"] = str(target_dir)
    print(f"TIKTOKEN_CACHE_DIR set to: {target_dir}")

    import tiktoken

    encodings = ["o200k_base", "cl100k_base"]
    for name in encodings:
        print(f"Ensuring encoding available: {name}")
        try:
            enc = tiktoken.get_encoding(name)
            # Touch encode/decode to ensure file is fully prepared
            _ = enc.encode("hello world")
            print(f"OK: {name}")
        except Exception as e:
            print(f"FAILED: {name} — {e}")

    print("Done. Commit the files under context_packager_data/tiktoken_cache and rebuild the wheel.")


if __name__ == "__main__":
    main()


