"""
agents/retrieval_agent.py

Agent 2: Retrieval Agent
For each extracted policy claim, retrieves the most relevant regulatory
chunks from the FAISS index using the ComplianceRetriever singleton.
"""

from __future__ import annotations

import logging

from graph.state import ComplianceState
from rag.retriever import get_retriever

logger = logging.getLogger(__name__)


def retrieval_agent(state: ComplianceState) -> ComplianceState:
    """
    Agent 2: Retrieval Agent.

    Reads state["policy_claims"], retrieves top-4 regulatory chunks per claim,
    and writes results to state["retrieved_chunks"].
    """
    state["status"] = "Retrieving relevant regulations..."

    claims: list[str] = state.get("policy_claims", [])

    if not claims:
        logger.warning("retrieval_agent: no policy claims to retrieve against.")
        state["retrieved_chunks"] = {}
        state["status"] = "Retrieval skipped: no claims available."
        return state

    retriever = get_retriever()
    retrieved: dict[str, list[str]] = {}
    failed: list[str] = []

    for claim in claims:
        try:
            chunks = retriever.retrieve(claim, top_k=4)
            retrieved[claim] = chunks
            logger.debug(
                "retrieval_agent: claim='%.60s...' → %d chunks", claim, len(chunks)
            )
        except Exception as exc:
            # One bad claim should not abort the entire pipeline
            logger.error(
                "retrieval_agent: retrieval failed for claim '%.60s' — %s", claim, exc
            )
            retrieved[claim] = []   # empty list keeps the key present for checker_agent
            failed.append(claim)

    state["retrieved_chunks"] = retrieved

    if failed:
        state["status"] = (
            f"Retrieval complete with {len(failed)} error(s) — "
            f"{len(claims) - len(failed)}/{len(claims)} claims retrieved."
        )
    else:
        state["status"] = (
            f"Retrieval complete — {len(claims)} claims matched to regulatory chunks."
        )

    logger.info(
        "retrieval_agent: %d/%d claims retrieved successfully.",
        len(claims) - len(failed), len(claims),
    )
    return state