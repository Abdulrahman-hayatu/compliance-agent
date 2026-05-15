"""
rag/upload_index.py

Run once locally to upload the pre-built FAISS index artifacts to a
HuggingFace Dataset repository so HF Spaces can download them at runtime.

"""

import argparse
import os
from huggingface_hub import HfApi, create_repo
from dotenv import load_dotenv

load_dotenv()

BASE_DIR         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "index", "compliance_index.faiss")
CHUNKS_PATH      = os.path.join(BASE_DIR, "index", "chunks.pkl")


def upload_index(repo_id: str, token: str) -> None:
    print(f"Creating dataset repo: {repo_id} ...")
    create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, token=token)

    api = HfApi()

    print("Uploading compliance_index.faiss ...")
    api.upload_file(
        path_or_fileobj=FAISS_INDEX_PATH,
        path_in_repo="compliance_index.faiss",
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
    )

    print("Uploading chunks.pkl ...")
    api.upload_file(
        path_or_fileobj=CHUNKS_PATH,
        path_in_repo="chunks.pkl",
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
    )

    print(f"\n✓ Index uploaded to: https://huggingface.co/datasets/{repo_id}")
    print(f"  Set HF_INDEX_REPO={repo_id} in your Space secrets.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        required=True,
        help="HuggingFace dataset repo ID",
    )
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN")
    if not token:
        raise EnvironmentError(
            "HF_TOKEN not set. Add your HuggingFace Write token to .env as HF_TOKEN=..."
        )

    upload_index(args.repo, token)