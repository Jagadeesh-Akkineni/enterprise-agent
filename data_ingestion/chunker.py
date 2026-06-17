"""
PDF Chunker — Hierarchical Section-Based Chunking
--------------------------------------------------
Strategy: detects section headers inside each PDF and creates one chunk per section.
If a section exceeds max_tokens, it is recursively split on paragraph → sentence
boundaries so no chunk ever blows an LLM context window.

Dependencies:
    pip install pdfplumber
"""

import re
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict

import pdfplumber

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DOCS_DIR = Path(__file__).parent.parent / "Sample Comp docs"
OUTPUT_FILE = Path(__file__).parent / "chunks.json"

MAX_TOKENS = 512    # hard ceiling per chunk (estimated: 1 token ≈ 4 chars)
OVERLAP_TOKENS = 64 # token overlap between sibling sub-chunks (context continuity)

# Patterns that signal a new section header inside policy / guide PDFs.
# Order matters: more specific patterns first.
HEADER_PATTERNS = [
    re.compile(r"^\s*(\d+[\.\d]*)\s+[A-Z][^\n]{3,}$"),          # "1. Introduction" / "2.1 Scope"
    re.compile(r"^\s*[A-Z][A-Z\s\-]{4,}$"),                      # "LEAVE ENTITLEMENTS"
    re.compile(r"^\s*(Section|SECTION|Chapter|CHAPTER)\s+\d+"),   # "Section 3"
    re.compile(r"^\s*(Article|ARTICLE)\s+[IVXLCDM\d]+"),         # "Article IV"
    re.compile(r"^\s*Q\d*[\.:]\s"),                               # "Q1:" FAQ questions
]

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    chunk_id: str
    source_file: str
    section_title: str
    page_start: int
    page_end: int
    text: str
    token_count: int
    sub_chunk_index: int = 0  # >0 when a section was split further


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def count_tokens(text: str) -> int:
    """
    Approximates token count without a network-dependent tokeniser.
    Rule of thumb: 1 token ≈ 4 characters for English prose.
    Accurate enough for chunking decisions; swap for tiktoken if network allows.
    """
    return max(1, len(text) // 4)


def is_header(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) < 3:
        return False
    return any(p.match(stripped) for p in HEADER_PATTERNS)


def split_on_paragraphs(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """
    Recursively split text that is too large.
    First tries paragraph boundaries, then sentence boundaries.
    Overlap is preserved by re-including the tail of the previous sub-chunk.
    """
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

    chunks: list[str] = []
    current_parts: list[str] = []
    current_tokens = 0
    overlap_tail = ""

    for para in paragraphs:
        para_tokens = count_tokens(para)

        # Single paragraph already too large → split by sentence
        if para_tokens > max_tokens:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sent in sentences:
                sent_tokens = count_tokens(sent)
                if current_tokens + sent_tokens > max_tokens and current_parts:
                    assembled = (overlap_tail + " " + " ".join(current_parts)).strip()
                    chunks.append(assembled)
                    overlap_tail = _tail_tokens(assembled, overlap_tokens)
                    current_parts = []
                    current_tokens = 0
                current_parts.append(sent)
                current_tokens += sent_tokens
            continue

        if current_tokens + para_tokens > max_tokens and current_parts:
            assembled = (overlap_tail + " " + " ".join(current_parts)).strip()
            chunks.append(assembled)
            overlap_tail = _tail_tokens(assembled, overlap_tokens)
            current_parts = []
            current_tokens = 0

        current_parts.append(para)
        current_tokens += para_tokens

    if current_parts:
        assembled = (overlap_tail + " " + " ".join(current_parts)).strip()
        chunks.append(assembled)

    return chunks or [text]


def _tail_tokens(text: str, n: int) -> str:
    """Return approximately the last n tokens of text (n*4 chars) for overlap."""
    char_count = n * 4
    return text[-char_count:] if len(text) > char_count else text


# ---------------------------------------------------------------------------
# Core: extract sections from a PDF
# ---------------------------------------------------------------------------

def extract_sections(pdf_path: Path) -> list[dict]:
    """
    Returns a list of {'title': str, 'text': str, 'page_start': int, 'page_end': int}
    one entry per detected section.
    """
    sections: list[dict] = []
    current_title = "Introduction"
    current_lines: list[str] = []
    current_page_start = 1

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            raw = page.extract_text() or ""
            for line in raw.splitlines():
                if is_header(line):
                    # Save whatever we've accumulated so far
                    body = "\n".join(current_lines).strip()
                    if body:
                        sections.append({
                            "title": current_title,
                            "text": body,
                            "page_start": current_page_start,
                            "page_end": page_num,
                        })
                    current_title = line.strip()
                    current_lines = []
                    current_page_start = page_num
                else:
                    current_lines.append(line)

    # Flush final section
    body = "\n".join(current_lines).strip()
    if body:
        sections.append({
            "title": current_title,
            "text": body,
            "page_start": current_page_start,
            "page_end": current_page_start,
        })

    # Edge case: no headers found → treat entire doc as one section
    if not sections:
        with pdfplumber.open(pdf_path) as pdf:
            full_text = "\n".join(
                page.extract_text() or "" for page in pdf.pages
            ).strip()
        sections = [{"title": pdf_path.stem, "text": full_text,
                     "page_start": 1, "page_end": len(pdf.pages)}]

    return sections


# ---------------------------------------------------------------------------
# Core: chunk a single PDF
# ---------------------------------------------------------------------------

def chunk_pdf(pdf_path: Path) -> list[Chunk]:
    sections = extract_sections(pdf_path)
    chunks: list[Chunk] = []
    chunk_counter = 0

    for section in sections:
        tokens = count_tokens(section["text"])

        if tokens <= MAX_TOKENS:
            chunk_counter += 1
            chunks.append(Chunk(
                chunk_id=f"{pdf_path.stem}__{chunk_counter:04d}",
                source_file=pdf_path.name,
                section_title=section["title"],
                page_start=section["page_start"],
                page_end=section["page_end"],
                text=section["text"],
                token_count=tokens,
                sub_chunk_index=0,
            ))
        else:
            sub_texts = split_on_paragraphs(section["text"], MAX_TOKENS, OVERLAP_TOKENS)
            for idx, sub_text in enumerate(sub_texts, start=1):
                chunk_counter += 1
                chunks.append(Chunk(
                    chunk_id=f"{pdf_path.stem}__{chunk_counter:04d}",
                    source_file=pdf_path.name,
                    section_title=section["title"],
                    page_start=section["page_start"],
                    page_end=section["page_end"],
                    text=sub_text,
                    token_count=count_tokens(sub_text),
                    sub_chunk_index=idx,
                ))

    return chunks


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    pdf_files = sorted(DOCS_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {DOCS_DIR}")
        return

    all_chunks: list[Chunk] = []

    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")
        file_chunks = chunk_pdf(pdf_path)
        all_chunks.extend(file_chunks)
        token_counts = [c.token_count for c in file_chunks]
        print(f"  -> {len(file_chunks)} chunks | "
              f"min={min(token_counts)} max={max(token_counts)} "
              f"avg={sum(token_counts)//len(token_counts)} tokens")

    OUTPUT_FILE.write_text(
        json.dumps([asdict(c) for c in all_chunks], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nTotal chunks: {len(all_chunks)}")
    print(f"Output saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
