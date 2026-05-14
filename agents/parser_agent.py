"""
agents/parser_agent.py

Agent 1: Document Parser
Extracts discrete, testable policy claims from the uploaded fintech document.
"""

from __future__ import annotations

import json
import logging
import os
import re

from groq import Groq
from dotenv import load_dotenv

from graph.state import ComplianceState

load_dotenv()
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL          = "llama-3.3-70b-versatile"
MAX_TOKENS     = 2048
TEMPERATURE    = 0.0   # deterministic output — we want consistent claim extraction

SYSTEM_PROMPT  = (
    "You are a regulatory compliance analyst. Your task is to extract discrete, "
    "testable policy claims from a fintech document. Each claim should be a "
    "single sentence describing what the document states about operations, "
    "data handling, agent relationships, customer treatment, or security.\n\n"
    "Return ONLY a JSON array of strings. No preamble. No markdown fences.\n"
    'Example: ["The company stores customer data on local servers.", '
    '"Agents are permitted to onboard customers remotely."]'
)

# Regex to pull the first JSON array out of a response even with surrounding text
JSON_ARRAY_RE  = re.compile(r'\[.*?\]', re.DOTALL)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_client() -> Groq:
    """Instantiate Groq client; raise clearly if API key is missing."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. "
            "Add it to your .env file or HuggingFace Secrets before running."
        )
    return Groq(api_key=api_key)


def _parse_claims(raw: str) -> list[str]:
    """
    Robustly extract a list of claim strings from the LLM response.

    Strategy (in order):
    1. Direct json.loads() — works when the model behaves.
    2. Strip markdown code fences, try again.
    3. Regex-extract the first [...] block, try again.
    4. Fallback: split on newlines, clean up each line.
    """
    text = raw.strip()

    # Strategy 1: direct parse
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return [str(c).strip() for c in result if str(c).strip()]
    except json.JSONDecodeError:
        pass

    # Strategy 2: strip markdown fences
    cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', text, flags=re.MULTILINE).strip()
    try:
        result = json.loads(cleaned)
        if isinstance(result, list):
            logger.warning("parser_agent: stripped markdown fences before JSON parse.")
            return [str(c).strip() for c in result if str(c).strip()]
    except json.JSONDecodeError:
        pass

    # Strategy 3: regex-extract first [...] block
    match = JSON_ARRAY_RE.search(text)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list):
                logger.warning("parser_agent: extracted JSON array via regex.")
                return [str(c).strip() for c in result if str(c).strip()]
        except json.JSONDecodeError:
            pass

    # Strategy 4: newline fallback
    logger.warning(
        "parser_agent: all JSON strategies failed — falling back to newline split. "
        "Raw response (first 300 chars): %s", text[:300]
    )
    claims = []
    for line in text.splitlines():
        line = line.strip().lstrip('-•*0123456789.)').strip()
        # Drop lines that look like JSON punctuation or are too short to be claims
        if len(line) > 20 and not line.startswith('[') and not line.startswith('{'):
            claims.append(line)
    return claims


# ── Agent ──────────────────────────────────────────────────────────────────────

def parser_agent(state: ComplianceState) -> ComplianceState:
    """
    Agent 1: Document Parser.

    Reads state["uploaded_text"], calls Groq to extract policy claims,
    and writes results back to state["policy_claims"] and state["status"].
    """
    # Update status first so the UI reflects progress immediately
    state["status"] = "Parsing document claims..."

    uploaded_text = state.get("uploaded_text", "").strip()
    if not uploaded_text:
        logger.warning("parser_agent: uploaded_text is empty — returning no claims.")
        state["policy_claims"] = []
        state["status"] = "Parser error: no document text provided."
        return state

    # Truncate to avoid exceeding context window (keep first ~12 000 chars ≈ ~3 000 tokens)
    if len(uploaded_text) > 12_000:
        logger.warning(
            "parser_agent: uploaded_text truncated from %d to 12 000 chars.",
            len(uploaded_text),
        )
        uploaded_text = uploaded_text[:12_000]

    client = _get_client()

    try:
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": uploaded_text},
            ],
        )
        raw = response.choices[0].message.content or ""
        logger.debug("parser_agent raw response (first 500 chars): %s", raw[:500])

    except Exception as exc:
        logger.error("parser_agent: Groq API call failed — %s", exc)
        state["policy_claims"] = []
        state["status"] = f"Parser error: Groq API call failed ({exc})"
        return state

    claims = _parse_claims(raw)

    if not claims:
        logger.warning("parser_agent: no claims extracted from document.")
        state["status"] = "Parser warning: no claims could be extracted."
    else:
        logger.info("parser_agent: extracted %d claims.", len(claims))
        state["status"] = f"Parsing complete — {len(claims)} claims extracted."

    state["policy_claims"] = claims
    return state