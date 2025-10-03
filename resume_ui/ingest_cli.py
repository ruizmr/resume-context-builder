import sys
from pathlib import Path

from hr_tools.pdf_to_md import convert_pdfs_to_markdown
from kb.upsert import upsert_markdown_files


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: context-ingest <input_dir> [<markdown_out_dir>]")
        sys.exit(1)
    input_dir = Path(sys.argv[1]).expanduser().resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Input directory not found: {input_dir}")
        sys.exit(2)
    md_out = (
        Path(sys.argv[2]).expanduser().resolve()
        if len(sys.argv) >= 3
        else Path.home() / "context-packager-md"
    )
    md_out.mkdir(parents=True, exist_ok=True)

    md_files = convert_pdfs_to_markdown(str(input_dir), str(md_out))
    count = upsert_markdown_files([Path(p) for p in md_files])
    print(f"Converted {len(md_files)} file(s); upserted {count} chunk(s)")


if __name__ == "__main__":
    main()


