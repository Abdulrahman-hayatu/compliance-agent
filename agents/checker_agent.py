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
import time

from groq import Groq, RateLimitError
from graph.state import ComplianceState
from agents.parser_agent import _get_client   # reuse key-validation helper

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL       = "openai/gpt-oss-120b"
MAX_TOKENS  = 512
TEMPERATURE = 0.0
MAX_RETRIES = 3       # per Groq's own recommended pattern for transient API/JSON-validation failures
RETRY_BACKOFF_BASE_SECONDS = 1.5   # attempt 1: 1.5s, attempt 2: 3.0s

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
    "Each excerpt in REGULATORY CONTEXT is labelled with its source and "
    "section, e.g. '[CBN 9.1]' or '[NDPA PART VI]'. When you cite a "
    "regulation, use exactly one of these labels — do not invent a section "
    "number that is not shown in the context.\n\n"
    "If NONE of the provided excerpts actually address the claim — the "
    "retrieved context is simply the nearest match found, not a guarantee "
    "of relevance — set status to UNCLEAR and set regulation_reference to "
    "\"N/A\". Do not cite one of the labels just because it was offered; "
    "only cite a label when that excerpt genuinely supports your "
    "explanation.\n\n"
    "Return ONLY a JSON object with this exact structure:\n"
    "{\n"
    '  "status": "COMPLIANT" | "NON_COMPLIANT" | "UNCLEAR",\n'
    '  "regulation_reference": "one of the [SOURCE section] labels shown above, e.g. \'CBN 9.1\', or \'N/A\' if no excerpt is relevant",\n'
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


def _build_user_message(claim: str, chunks: list[dict]) -> str:
    if not chunks:
        context = "No regulatory context available."
    else:
        # Label each excerpt with its source/section so the model can cite a
        # specific, verifiable clause (e.g. "CBN 9.1", "NDPA PART VI") instead
        # of guessing a section number from unlabelled text.
        parts = []
        for c in chunks:
            label = f"[{c.get('source', '?')} {c.get('section', '?')}]"
            parts.append(f"{label}\n{c.get('text', '')}")
        context = "\n\n".join(parts)
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
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for i, claim in enumerate(claims, start=1):
        chunks: list[dict] = retrieved.get(claim, [])

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

        raw = None
        last_exc: Exception | None = None
        rate_limited = False
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": user_msg},
                    ],
                )
                raw = response.choices[0].message.content or ""
                usage = getattr(response, "usage", None)
                if usage is not None:
                    p_tok = getattr(usage, "prompt_tokens", 0) or 0
                    c_tok = getattr(usage, "completion_tokens", 0) or 0
                    total_prompt_tokens += p_tok
                    total_completion_tokens += c_tok
                    logger.info(
                        "checker_agent: claim %d token usage — prompt=%d, completion=%d, total=%d",
                        i, p_tok, c_tok, p_tok + c_tok,
                    )
                break
            except RateLimitError as exc:
                # A 429 quota-exceeded error won't be fixed by retrying within
                # seconds -- the daily/token budget is what's exhausted, not a
                # transient blip. Fail fast: don't burn the remaining attempts
                # (and their backoff delays) on a call that can't succeed.
                last_exc = exc
                rate_limited = True
                logger.error(
                    "checker_agent: Groq rate limit hit on claim %d — not retrying "
                    "(quota-exceeded errors won't resolve within a retry window). %s",
                    i, exc,
                )
                break
            except Exception as exc:
                last_exc = exc
                if attempt < MAX_RETRIES:
                    logger.warning(
                        "checker_agent: Groq API error on claim %d, attempt %d/%d — %s. Retrying...",
                        i, attempt, MAX_RETRIES, exc,
                    )
                    time.sleep(RETRY_BACKOFF_BASE_SECONDS * attempt)
                else:
                    logger.error(
                        "checker_agent: Groq API error on claim %d — exhausted %d attempts. "
                        "Last error: %s", i, MAX_RETRIES, exc,
                    )

        if raw is None:
            if rate_limited:
                reason = f"Groq quota/rate limit exceeded (not retried): {last_exc}"
            else:
                reason = f"API error after {MAX_RETRIES} attempts: {last_exc}"
            results.append({
                "claim": claim,
                **_fallback(reason),
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
    logger.info(
        "checker_agent: run total token usage — prompt=%d, completion=%d, total=%d "
        "across %d claim(s), avg=%.0f tokens/claim",
        total_prompt_tokens, total_completion_tokens,
        total_prompt_tokens + total_completion_tokens, len(claims),
        (total_prompt_tokens + total_completion_tokens) / len(claims) if claims else 0,
    )
    return state