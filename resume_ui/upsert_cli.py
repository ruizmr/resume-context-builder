import sys
from pathlib import Path

from kb.upsert import upsert_markdown_files


def main() -> None:
	if len(sys.argv) < 2:
		print("Usage: context-upsert <markdown_dir>")
		sys.exit(1)
	root = Path(sys.argv[1]).expanduser().resolve()
	md_files = sorted(root.rglob("*.md"))
	count = upsert_markdown_files(md_files)
	print(f"Upserted {count} chunk(s) from {len(md_files)} file(s)")


if __name__ == "__main__":
	main()


