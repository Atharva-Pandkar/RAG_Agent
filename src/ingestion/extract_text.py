"""
Extract clean text from SEC 10-K HTML filings.

Phase 0 use: produce readable .txt dumps so we can author golden-set
Q&A pairs grounded in real filing content.

Phase 1+ use: this is the first stage of the ingestion pipeline -
swap `extract_text` for a section-aware variant that tags Item
boundaries and tables separately.
"""
import re
from pathlib import Path
from bs4 import BeautifulSoup

DATASET_DIR = Path(__file__).resolve().parents[2] / "Dataset"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "Documents" / "extracted"


def extract_text(html_path: Path) -> str:
    with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "lxml")

    for tag in soup(["script", "style"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    # collapse excessive blank lines / whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for html_file in sorted(DATASET_DIR.glob("*.html")):
        print(f"Extracting {html_file.name} ...")
        text = extract_text(html_file)
        out_path = OUTPUT_DIR / (html_file.stem + ".txt")
        out_path.write_text(text, encoding="utf-8")
        print(f"  -> {out_path} ({len(text):,} chars)")


if __name__ == "__main__":
    main()
