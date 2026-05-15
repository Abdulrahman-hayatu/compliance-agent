"""
rag/hub_loader.py

Downloads the pre-built FAISS index artifacts from a HuggingFace Dataset
repository into the local index/ directory at app startup if
the files are not already present.
"""

import os
import logging

logger = logging.getLogger(__name__)

BASE_DIR         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_DIR        = os.path.join(BASE_DIR, "index")
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "compliance_index.faiss")
CHUNKS_PATH      = os.path.join(INDEX_DIR, "chunks.pkl")

ARTIFACTS = {
    "compliance_index.faiss": FAISS_INDEX_PATH,
    "chunks.pkl":             CHUNKS_PATH,
}


def ensure_index_available() -> None:
    """
    Check if index artifacts exist locally. If not, download them from the
    HuggingFace Dataset repo specified in the HF_INDEX_REPO environment variable.

    Raises:
        EnvironmentError: if HF_INDEX_REPO is not set and artifacts are missing.
        RuntimeError:     if download fails.
    """
    missing = [name for name, path in ARTIFACTS.items() if not os.path.exists(path)]

    if not missing:
        logger.info("hub_loader: index artifacts already present — skipping download.")
        return

    repo_id = os.getenv("HF_INDEX_REPO")
    if not repo_id:
        raise EnvironmentError(
            "Index artifacts are missing and HF_INDEX_REPO is not set.\n"
            "Either:\n"
            "  (a) Run `python -m rag.indexer` to build the index locally, or\n"
            "  (b) Set HF_INDEX_REPO=YOUR_USERNAME/compliance-index in your environment."
        )

    logger.info(
        "hub_loader: downloading missing artifacts %s from %s ...", missing, repo_id
    )

    try:
        from huggingface_hub import hf_hub_download

        os.makedirs(INDEX_DIR, exist_ok=True)

        for filename, local_path in ARTIFACTS.items():
            if os.path.exists(local_path):
                continue
            logger.info("hub_loader: downloading %s ...", filename)
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                local_dir=INDEX_DIR,
            )
            logger.info("hub_loader: saved %s → %s", filename, downloaded)

        logger.info("hub_loader: all artifacts downloaded successfully.")

    except Exception as exc:
        raise RuntimeError(
            f"Failed to download index artifacts from '{repo_id}': {exc}\n"
            "Check that HF_INDEX_REPO is correct and the dataset is public."
        ) from exc