"""
agents/checker_agent.py

Agent 3: Compliance Checker
Assesses each extracted policy claim against retrieved regulatory chunks
and produces a structured per-claim compliance result.
"""

from __future__ import annotations

import json
import logging
import re

from groq import Groq
from graph.state import ComplianceState
from agents.parser_agent import _get_client   # reuse key-validation helper

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL       = "llama-3.3-70b-versatile"
MAX_TOKENS  = 512
TEMPERATURE = 0.0

VALID_STATUSES = {"COMPLIANT", "NON_COMPLIANT", "UNCLEAR"}
REQUIRED_KEYS  = {"status", "regulation_reference", "explanation", "remediation"}

SYSTEM_PROMPT = (
    "You are a Nigerian regulatory compliance expert specializing in CBN Agent "
    "Banking Guidelines and the Nigeria Data Protection Act 2023.\n\n"
    "You will be given a policy claim from a fintech company and relevant "
    "excerpts from Nigerian regulations.\n\n"
    "Assess whether the claim is:\n"
    "- COMPLIANT: clearly aligned with regulations\n"
    "- NON_COMPLIANT: clearly violates or contradicts regulations\n"
    "- UNCLEAR: insufficient information to determine compliance\n\n"
    "Return ONLY a JSON object with this exact structure:\n"
    "{\n"
    '  "status": "COMPLIANT" | "NON_COMPLIANT" | "UNCLEAR",\n'
    '  "regulation_reference": "specific section or clause cited",\n'
    '  "explanation": "one sentence explanation",\n'
    '  "remediation": "specific action needed, or null if compliant"\n'
    "}\n"
    "No preamble. No markdown fences."
)

JSON_OBJ_RE = re.compile(r'\{.*?\}', re.DOTALL)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_checker_response(raw: str) -> dict:
    """
    Robustly parse the LLM compliance assessment JSON.

    Strategy (in order):
    1. Direct json.loads()
    2. Strip markdown fences, try again.
    3. Regex-extract first {...} block, try again.
    4. Return a safe fallback UNCLEAR result.

    Also validates that all required keys are present and status is a known value.
    """
    text = raw.strip()

    def _load(s: str) -> dict | None:
        try:
            result = json.loads(s)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass
        return None

    match = JSON_OBJ_RE.search(text)
    parsed = (
        _load(text)
        or _load(re.sub(r'^```(?:json)?\s*|\s*```$', '', text, flags=re.MULTILINE).strip())
        or (match and _load(match.group()))
    )

    if not parsed:   # None or empty dict — both mean no usable data
        logger.warning("checker_agent: all JSON strategies failed. Raw: %.200s", text)
        return _fallback("JSON parse failed — model returned unparseable output.")

    return _validate(parsed)


def _validate(parsed: dict) -> dict:
    """Ensure required keys exist and status is a recognised value."""
    # Fill any missing keys with safe defaults
    result = {
        "status":               parsed.get("status", "UNCLEAR"),
        "regulation_reference": parsed.get("regulation_reference", "Unknown"),
        "explanation":          parsed.get("explanation", "No explanation provided."),
        "remediation":          parsed.get("remediation", None),
    }

    # Normalise status to uppercase; default to UNCLEAR if unrecognised
    result["status"] = result["status"].upper().strip()
    if result["status"] not in VALID_STATUSES:
        logger.warning(
            "checker_agent: unrecognised status '%s' — defaulting to UNCLEAR.",
            result["status"],
        )
        result["status"] = "UNCLEAR"

    # null JSON → Python None; anything that isn't a string becomes None
    if result["remediation"] is not None and not isinstance(result["remediation"], str):
        result["remediation"] = str(result["remediation"])

    return result


def _fallback(reason: str) -> dict:
    return {
        "status":               "UNCLEAR",
        "regulation_reference": "N/A",
        "explanation":          reason,
        "remediation":          "Manual review required.",
    }


def _build_user_message(claim: str, chunks: list[str]) -> str:
    context = "\n\n".join(chunks) if chunks else "No regulatory context available."
    return f"POLICY CLAIM: {claim}\n\nREGULATORY CONTEXT:\n{context}"


# ── Agent ──────────────────────────────────────────────────────────────────────

def checker_agent(state: ComplianceState) -> ComplianceState:
    """
    Agent 3: Compliance Checker.

    Iterates over policy_claims, calls Groq once per claim with its
    retrieved regulatory chunks, and builds a structured compliance_results list.
    """
    state["status"] = "Checking compliance..."

    claims: list[str] = state.get("policy_claims", [])
    retrieved: dict   = state.get("retrieved_chunks", {})

    if not claims:
        logger.warning("checker_agent: no claims to assess.")
        state["compliance_results"] = []
        state["status"] = "Compliance check skipped: no claims available."
        return state

    client: Groq = _get_client()
    results: list[dict] = []

    for i, claim in enumerate(claims, start=1):
        chunks: list[str] = retrieved.get(claim, [])

        if not chunks:
            logger.warning(
                "checker_agent: no regulatory chunks for claim %d — marking UNCLEAR.", i
            )
            results.append({
                "claim": claim,
                **_fallback("No relevant regulatory excerpts were retrieved for this claim."),
            })
            continue

        user_msg = _build_user_message(claim, chunks)

        try:
            response = client.chat.completions.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
            )
            raw = response.choices[0].message.content or ""
        except Exception as exc:
            logger.error("checker_agent: Groq API error on claim %d — %s", i, exc)
            results.append({
                "claim": claim,
                **_fallback(f"API error: {exc}"),
            })
            continue

        parsed = _parse_checker_response(raw)
        results.append({"claim": claim, **parsed})
        logger.debug(
            "checker_agent: claim %d/%d → %s", i, len(claims), parsed["status"]
        )

    state["compliance_results"] = results

    counts = {s: sum(1 for r in results if r["status"] == s) for s in VALID_STATUSES}
    state["status"] = (
        f"Compliance check complete — "
        f"{counts['COMPLIANT']} compliant, "
        f"{counts['NON_COMPLIANT']} non-compliant, "
        f"{counts['UNCLEAR']} unclear."
    )

    logger.info("checker_agent: %s", state["status"])
    return state