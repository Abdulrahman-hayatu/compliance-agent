"""
Nigerian Fintech Compliance Agent Indexer 
rag/indexer.py
"""

import os
import re
import pickle

import pdfplumber
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# documents paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CBN_PDF  = os.path.join(BASE_DIR, "data",
           "CIRCULAR AND GUIDELINES FOR THE OPERATIONS OF AGENT BANKING IN NIGERIA OCTOBER 6 2025.pdf")
NDPC_PDF = os.path.join(BASE_DIR, "data",
           "Nigeria_Data_Protection_Act_2023.pdf")

INDEX_DIR        = os.path.join(BASE_DIR, "index")
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "compliance_index.faiss")
CHUNKS_PATH      = os.path.join(INDEX_DIR, "chunks.pkl")

# Regex patterns for chunking
SECTION_PATTERN  = re.compile(r'(?=\n\d+\.\d+\s+[A-Z])')   # CBN:  X.X UPPERCASE
NDPC_PATTERN     = re.compile(r'(?=\n\d{1,2}\.(?:—\(|\s+[A-Z]))')  # NDPA: real section numbers, e.g. "24.—(1)" or "66. This Act..."
TOC_PATTERN      = re.compile(r'\.{4,}')                     # dot-leaders in TOC

# Header-capture patterns — used only to attach a citable clause/part
# number to each chunk as metadata. Does NOT affect chunk boundaries.
CBN_HEADER_RE  = re.compile(r'^(\d+\.\d+)\s+[A-Z]|^(APPENDIX [IVX]+)')
NDPC_HEADER_RE = re.compile(r'^(\d{1,2})\.(?:—\(|\s+[A-Z])')

# The Schedule's internal paragraph numbering restarts from 1, which would
# collide with real Section 1 if split on the same section-number pattern.
# Isolate it first using this heading, unique in the document, so it's
# chunked separately rather than confused with the main numbered sections.
NDPC_SCHEDULE_MARKER = re.compile(
    r'SCHEDULE\s+Section\s+8\(3\)\s*\nSUPPLEMENTARY PROVISIONS', re.IGNORECASE
)

# The actual NDPC legislation begins after this marker (post Arrangement of Sections)
NDPC_BODY_MARKER = re.compile(
    r'NIGERIA DATA PROTECTION ACT,\s*2023\s*\n\s*ACT', re.IGNORECASE
)

MIN_CHUNK_LENGTH      = 100
MIN_MEANINGFUL_LENGTH = 300


# Indexing functions

def extract_text_from_pdf(pdf_path: str, skip_first_page: bool = False) -> str:
    texts = []
    with pdfplumber.open(pdf_path) as pdf:
        pages = pdf.pages[1:] if skip_first_page else pdf.pages
        for page in pages:
            text = page.extract_text()
            if text:
                texts.append(text)
    return "\n".join(texts)


def skip_ndpc_preamble(text: str) -> str:
    """
    Discard everything before the actual legislative body of the NDPC.
    The body starts at the 'NIGERIA DATA PROTECTION ACT, 2023 / ACT No.'
    heading that follows the Arrangement of Sections.
    Falls back to full text if marker is not found.
    """
    match = NDPC_BODY_MARKER.search(text)
    if match:
        return text[match.start():]
    # Fallback: try to find "ACT No." as a simpler marker
    fallback = text.find("\nACT No.")
    if fallback != -1:
        return text[fallback:]
    return text  # no marker found — use full text


def is_toc_chunk(chunk: str) -> bool:
    """Dot-leader heuristic — flags CBN-style TOC entries."""
    lines = [l for l in chunk.splitlines() if l.strip()]
    if not lines:
        return True
    toc_lines = sum(1 for l in lines if TOC_PATTERN.search(l))
    return (toc_lines / len(lines)) > 0.30


def extract_header(chunk_text: str, pattern: re.Pattern) -> str | None:
    """
    Capture just the clause/part number (e.g. '2.4', 'PART III') from the
    start of a chunk, for use as citable metadata. Returns None if the
    chunk doesn't start with a recognisable header (e.g. preamble text).
    """
    m = pattern.match(chunk_text.strip())
    if not m:
        return None
    # CBN_HEADER_RE has two alternative capture groups (numbered clause vs
    # APPENDIX heading); other patterns only ever populate group 1.
    groups = [g for g in m.groups() if g]
    return groups[0].strip() if groups else None


def _apply_chunking(raw_chunks: list[str], source_prefix: str,
                     header_pattern: re.Pattern) -> list[dict]:
    """
    Shared post-processing:
    - Strip whitespace
    - Drop short / TOC chunks
    - Merge stubs below MIN_MEANINGFUL_LENGTH into next chunk
    - Prepend source_prefix
    - Attach a stable-ish 'section' (clause/part number) as metadata

    Returns a list of {"text": str, "section": str | None} dicts.
    Chunk boundaries and merge behaviour are unchanged from the original
    implementation — only metadata tracking is added.
    """
    filtered = []
    for chunk in raw_chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        # NOTE: previously dropped any chunk under MIN_CHUNK_LENGTH (100 chars)
        # outright via `continue` here — before the merge-forward logic below
        # ever got a chance to absorb it. That silently discarded real,
        # substantive short clauses (e.g. CBN 4.2 "Agents shall be exclusive
        # to one Principal only.", 52 chars) instead of merging them into a
        # neighboring chunk like every other stub. MIN_MEANINGFUL_LENGTH (300)
        # already fully subsumes this filter's job — anything short still
        # gets merged forward in the loop below, it just isn't dropped first.
        if is_toc_chunk(chunk):
            continue
        section = extract_header(chunk, header_pattern)
        filtered.append({"text": chunk, "section": section})

    # Merge stubs — mirrors original string-concatenation logic exactly,
    # carrying the first available section label forward through a merge.
    merged: list[dict] = []
    buffer: dict | None = None
    for item in filtered:
        if buffer:
            item = {
                "text": buffer["text"] + "\n" + item["text"],
                "section": buffer["section"] or item["section"],
            }
            buffer = None
        if len(item["text"]) < MIN_MEANINGFUL_LENGTH:
            buffer = item
        else:
            merged.append(item)
    if buffer:
        if merged:
            merged[-1]["text"] += "\n" + buffer["text"]
            merged[-1]["section"] = merged[-1]["section"] or buffer["section"]
        else:
            merged.append(buffer)

    return [
        {"text": f"{source_prefix}\n{item['text']}", "section": item["section"]}
        for item in merged
    ]


def chunk_text(text: str, source_prefix: str) -> list[dict]:
    """Chunk CBN document using X.X UPPERCASE section-header pattern, plus a
    boundary before APPENDIX headings so an appendix (e.g. the Administrative
    Sanctions table) doesn't silently merge into whatever numbered section
    precedes it, conflating two unrelated provisions under one ID."""
    text = re.sub(r'\n(APPENDIX [IVX]+)', r'\n\n\1', text)
    raw_chunks = SECTION_PATTERN.split(text)
    expanded = []
    for rc in raw_chunks:
        expanded.extend(re.split(r'(?=\nAPPENDIX [IVX]+)', rc))
    return _apply_chunking(expanded, source_prefix, CBN_HEADER_RE)


def chunk_ndpc_text(text: str, source_prefix: str) -> list[dict]:
    """
    Chunk NDPA document by individual numbered SECTION (e.g. "24.—(1)" or
    "66. This Act..."), not by the much coarser PART headers — PART-level
    chunking put up to ~15K chars (ten distinct sections) under one ID,
    making per-clause citations meaningless. The Schedule is isolated
    first since its internal paragraph numbering (1-18) restarts and would
    otherwise collide with real Section 1-18 numbers.
    """
    body_text = skip_ndpc_preamble(text)

    schedule_match = NDPC_SCHEDULE_MARKER.search(body_text)
    if schedule_match:
        main_text = body_text[:schedule_match.start()]
        schedule_text = body_text[schedule_match.start():]
    else:
        main_text = body_text
        schedule_text = ""

    raw_chunks = NDPC_PATTERN.split(main_text)
    section_chunks = _apply_chunking(raw_chunks, source_prefix, NDPC_HEADER_RE)

    final = []
    for item in section_chunks:
        chunk = item["text"]
        if len(chunk) <= 4000:
            final.append(item)
        else:
            sub_chunks = [s.strip() for s in re.split(r'\n{2,}', chunk)
                          if len(s.strip()) >= MIN_CHUNK_LENGTH]
            for sub in sub_chunks:
                if not sub.startswith("SOURCE:"):
                    sub = f"{source_prefix}\n{sub}"
                final.append({"text": sub, "section": item["section"]})

    if schedule_text.strip():
        final.append({
            "text": f"{source_prefix}\n{schedule_text.strip()}",
            "section": "SCHEDULE",
        })

    return final


def assign_ids(chunks: list[dict], id_prefix: str, source_label: str) -> list[dict]:
    """
    Attach a stable ID and source label to each chunk dict.
    IDs are positional (id_prefix_001, id_prefix_002, ...) and are stable
    ONLY for a given run of the indexer against a given PDF. If the source
    PDFs are replaced/updated, chunk boundaries can shift and IDs will not
    necessarily refer to the same clauses — any golden-dataset clause
    references would need to be regenerated after a re-index in that case.
    A chunk with no detected section header (e.g. preamble text) gets
    section="PREAMBLE" rather than None, so downstream consumers always
    have a non-null label to display.
    """
    result = []
    for i, item in enumerate(chunks, start=1):
        result.append({
            "id": f"{id_prefix}_{i:03d}",
            "source": source_label,
            "section": item["section"] or "PREAMBLE",
            "text": item["text"],
        })
    return result


def build_index(chunks: list[dict]) -> faiss.IndexFlatL2:
    texts = [c["text"] for c in chunks]
    print(f"\nLoading embedding model: BAAI/bge-small-en-v1.5 ...")
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    print(f"Encoding {len(texts)} chunks ...")
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype="float32")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    print(f"FAISS index built — {index.ntotal} vectors, dimension {dimension}.")
    return index


def save_artifacts(index: faiss.IndexFlatL2, chunks: list[dict]) -> None:
    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"FAISS index saved → {FAISS_INDEX_PATH}")
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Chunks saved      → {CHUNKS_PATH}")


# Main execution

def main() -> None:
    print("=" * 60)
    print("Nigerian Fintech Compliance Agent — Indexer")
    print("=" * 60)

    print("\n[1/4] Extracting CBN Agent Banking Guidelines ...")
    cbn_text = extract_text_from_pdf(CBN_PDF, skip_first_page=True)
    print(f"      Extracted {len(cbn_text):,} characters.")

    print("\n[1/4] Extracting Nigeria Data Protection Act 2023 ...")
    ndpc_text = extract_text_from_pdf(NDPC_PDF, skip_first_page=False)
    print(f"      Extracted {len(ndpc_text):,} characters.")

    # Verify NDPC body marker is found
    match = NDPC_BODY_MARKER.search(ndpc_text)
    if match:
        print(f"      NDPC body marker found at char {match.start():,}.")
    else:
        print("      WARNING: NDPC body marker not found — using full text (check output).")

    print("\n[2/4] Chunking documents ...")
    cbn_chunks_raw  = chunk_text(cbn_text,       "SOURCE: CBN Agent Banking Guidelines")
    ndpc_chunks_raw = chunk_ndpc_text(ndpc_text, "SOURCE: Nigeria Data Protection Act 2023")

    cbn_chunks  = assign_ids(cbn_chunks_raw,  "cbn",  "CBN")
    ndpc_chunks = assign_ids(ndpc_chunks_raw, "ndpa", "NDPA")

    all_chunks = cbn_chunks + ndpc_chunks
    print(f"      CBN chunks : {len(cbn_chunks)}")
    print(f"      NDPC chunks: {len(ndpc_chunks)}")
    print(f"      Total      : {len(all_chunks)}")

    if not all_chunks:
        raise ValueError("No chunks produced. Check PDFs are text-based and regex matches.")

    print("\n[3/4] Embedding and building FAISS index ...")
    index = build_index(all_chunks)

    print("\n[4/4] Saving artifacts ...")
    save_artifacts(index, all_chunks)

    print("\n✓ Indexing complete. Commit index/ to your repository.")
    print("=" * 60)


if __name__ == "__main__":
    main()
