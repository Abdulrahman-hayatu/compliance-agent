"""
graph/state.py
LangGraph state definition for the compliance pipeline.
Defines the ComplianceState TypedDict which holds all relevant data as it flows through the pipeline:
- uploaded_text: raw text from the user-uploaded document
- policy_claims: extracted policy claims from the parser agent
- retrieved_chunks: mapping of claims to regulatory chunks
- compliance_results: per-claim compliance assessments from the checker agent
- final_report: formatted markdown compliance report
- status: current pipeline stage label (for UI progress updates)
"""

from __future__ import annotations
from typing import TypedDict


class ComplianceState(TypedDict):
    uploaded_text: str              # Raw text from the user-uploaded document
    policy_claims: list[str]        # Extracted policy claims from the parser agent
    retrieved_chunks: dict          # Mapping: claim (str) -> list[str] of regulatory chunks
    compliance_results: list[dict]  # Per-claim compliance assessments from checker agent
    final_report: str               # Formatted markdown compliance report
    status: str                     # Current pipeline stage label (for UI progress updates)