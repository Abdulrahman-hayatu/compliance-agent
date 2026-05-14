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
NDPC_PATTERN     = re.compile(r'(?=\nPART [IVX]+[—\-\s])')  # NDPC: PART I— style
TOC_PATTERN      = re.compile(r'\.{4,}')                     # dot-leaders in TOC

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


def _apply_chunking(raw_chunks: list[str], source_prefix: str) -> list[str]:
    """
    Shared post-processing:
    - Strip whitespace
    - Drop short / TOC chunks
    - Merge stubs below MIN_MEANINGFUL_LENGTH into next chunk
    - Prepend source_prefix
    """
    filtered = []
    for chunk in raw_chunks:
        chunk = chunk.strip()
        if len(chunk) < MIN_CHUNK_LENGTH:
            continue
        if is_toc_chunk(chunk):
            continue
        filtered.append(chunk)

    # Merge stubs
    merged = []
    buffer = ""
    for chunk in filtered:
        if buffer:
            chunk = buffer + "\n" + chunk
            buffer = ""
        if len(chunk) < MIN_MEANINGFUL_LENGTH:
            buffer = chunk
        else:
            merged.append(chunk)
    if buffer:
        if merged:
            merged[-1] += "\n" + buffer
        else:
            merged.append(buffer)

    return [f"{source_prefix}\n{chunk}" for chunk in merged]


def chunk_text(text: str, source_prefix: str) -> list[str]:
    """Chunk CBN document using X.X UPPERCASE section-header pattern."""
    raw_chunks = SECTION_PATTERN.split(text)
    return _apply_chunking(raw_chunks, source_prefix)


def chunk_ndpc_text(text: str, source_prefix: str) -> list[str]:
    """
    Chunk NDPC document by PART headers.
    First skips the Arrangement of Sections preamble so only real
    legislative body text is indexed.
    Secondary split on paragraph breaks for chunks exceeding 4000 chars.
    """
    body_text = skip_ndpc_preamble(text)
    raw_chunks = NDPC_PATTERN.split(body_text)
    part_chunks = _apply_chunking(raw_chunks, source_prefix)

    final = []
    for chunk in part_chunks:
        if len(chunk) <= 4000:
            final.append(chunk)
        else:
            sub_chunks = [s.strip() for s in re.split(r'\n{2,}', chunk)
                          if len(s.strip()) >= MIN_CHUNK_LENGTH]
            for sub in sub_chunks:
                if not sub.startswith("SOURCE:"):
                    sub = f"{source_prefix}\n{sub}"
                final.append(sub)

    return final


def build_index(chunks: list[str]) -> faiss.IndexFlatL2:
    print(f"\nLoading embedding model: BAAI/bge-small-en-v1.5 ...")
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    print(f"Encoding {len(chunks)} chunks ...")
    embeddings = model.encode(chunks, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype="float32")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    print(f"FAISS index built — {index.ntotal} vectors, dimension {dimension}.")
    return index


def save_artifacts(index: faiss.IndexFlatL2, chunks: list[str]) -> None:
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
    from rag.indexer import skip_ndpc_preamble, NDPC_BODY_MARKER
    match = NDPC_BODY_MARKER.search(ndpc_text)
    if match:
        print(f"      NDPC body marker found at char {match.start():,}.")
    else:
        print("      WARNING: NDPC body marker not found — using full text (check output).")

    print("\n[2/4] Chunking documents ...")
    cbn_chunks  = chunk_text(cbn_text,       "SOURCE: CBN Agent Banking Guidelines")
    ndpc_chunks = chunk_ndpc_text(ndpc_text, "SOURCE: Nigeria Data Protection Act 2023")

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