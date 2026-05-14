"""
agents/report_agent.py

Agent 4: Report Generator
Produces a structured markdown compliance report from the checker's results.
Falls back to a locally-generated report if the Groq API call fails,
so the pipeline always delivers output.
"""

from __future__ import annotations

import json
import logging
from datetime import date

from groq import Groq
from graph.state import ComplianceState
from agents.parser_agent import _get_client

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL       = "llama-3.3-70b-versatile"
MAX_TOKENS  = 4096   # reports are long — give the model room
TEMPERATURE = 0.2    # slight creativity for readable prose, but mostly deterministic

STATUS_BADGE = {
    "COMPLIANT":     "✅ COMPLIANT",
    "NON_COMPLIANT": "❌ NON-COMPLIANT",
    "UNCLEAR":       "⚠️ UNCLEAR",
}

SYSTEM_PROMPT = (
    "You are a compliance report writer. Generate a clear, structured markdown "
    "compliance report for a Nigerian fintech company based on the assessment "
    "results provided. Include:\n\n"
    "1. An executive summary (2-3 sentences)\n"
    "2. A compliance scorecard (count of COMPLIANT / NON_COMPLIANT / UNCLEAR)\n"
    "3. A detailed findings section — for each finding include the claim, "
    "status badge, regulation reference, and explanation\n"
    "4. A prioritized remediation checklist for all NON_COMPLIANT items\n"
    "5. A closing note on scope limitations (this assessment covers CBN Agent "
    "Banking Guidelines Oct 2025 and NDPC 2023 only)\n\n"
    "Use clear markdown formatting with headers, bold labels, and bullet points."
)


# ── Fallback report ────────────────────────────────────────────────────────────

def _build_fallback_report(results: list[dict]) -> str:
    """
    Build a well-structured markdown report directly from compliance_results
    without calling the LLM. Used when the Groq API call fails.
    """
    today = date.today().strftime("%B %d, %Y")
    counts = {s: sum(1 for r in results if r.get("status") == s)
              for s in ("COMPLIANT", "NON_COMPLIANT", "UNCLEAR")}
    total  = len(results)

    non_compliant = [r for r in results if r.get("status") == "NON_COMPLIANT"]
    compliant_pct = round((counts["COMPLIANT"] / total * 100) if total else 0)

    lines = [
        "# Nigerian Fintech Regulatory Compliance Report",
        f"*Generated: {today} | Scope: CBN Agent Banking Guidelines (Oct 2025) & NDPC 2023*",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        f"This automated assessment evaluated **{total} policy claim(s)** extracted from "
        f"the submitted fintech document against Nigerian regulatory requirements. "
        f"**{counts['COMPLIANT']} claim(s) ({compliant_pct}%) are compliant**, "
        f"{counts['NON_COMPLIANT']} require remediation, and {counts['UNCLEAR']} could "
        "not be conclusively assessed. Immediate action is required on all non-compliant items.",
        "",
        "---",
        "",
        "## Compliance Scorecard",
        "",
        f"| Status | Count |",
        f"|--------|-------|",
        f"| ✅ Compliant     | {counts['COMPLIANT']} |",
        f"| ❌ Non-Compliant | {counts['NON_COMPLIANT']} |",
        f"| ⚠️ Unclear       | {counts['UNCLEAR']} |",
        f"| **Total**        | **{total}** |",
        "",
        "---",
        "",
        "## Detailed Findings",
        "",
    ]

    for i, r in enumerate(results, start=1):
        badge = STATUS_BADGE.get(r.get("status", "UNCLEAR"), "⚠️ UNCLEAR")
        lines += [
            f"### Finding {i}",
            "",
            f"**Claim:** {r.get('claim', 'N/A')}",
            "",
            f"**Status:** {badge}",
            "",
            f"**Regulation Reference:** {r.get('regulation_reference', 'N/A')}",
            "",
            f"**Explanation:** {r.get('explanation', 'N/A')}",
            "",
        ]
        if r.get("remediation"):
            lines += [f"**Remediation:** {r['remediation']}", ""]
        lines.append("---")
        lines.append("")

    lines += [
        "## Remediation Checklist",
        "",
    ]

    if non_compliant:
        for i, r in enumerate(non_compliant, start=1):
            lines.append(f"- [ ] **{i}.** {r.get('remediation', 'Review required.')} "
                         f"*(Re: {r.get('regulation_reference', 'N/A')})*")
    else:
        lines.append("*No remediation items identified.*")

    lines += [
        "",
        "---",
        "",
        "## Scope & Limitations",
        "",
        "This assessment is limited to the following regulatory frameworks:",
        "",
        "- **CBN Circular and Guidelines for the Operations of Agent Banking in Nigeria** (October 6, 2025)",
        "- **Nigeria Data Protection Act 2023 (NDPC)**",
        "",
        "Claims that fall outside these frameworks are marked as **UNCLEAR** and require "
        "manual review by a qualified compliance officer. This report does not constitute "
        "legal advice.",
    ]

    return "\n".join(lines)


# ── Agent ──────────────────────────────────────────────────────────────────────

def report_agent(state: ComplianceState) -> ComplianceState:
    """
    Agent 4: Report Generator.

    Calls Groq to produce a structured markdown compliance report.
    Falls back to a locally-generated report if the API call fails.
    """
    state["status"] = "Generating report..."

    results: list[dict] = state.get("compliance_results", [])

    if not results:
        logger.warning("report_agent: compliance_results is empty.")
        state["final_report"] = _build_fallback_report([])
        state["status"] = "Report generated (no findings to report)."
        return state

    user_message = json.dumps(results, indent=2, ensure_ascii=False)

    try:
        client: Groq = _get_client()
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
        )
        report = (response.choices[0].message.content or "").strip()

        if not report:
            raise ValueError("Groq returned an empty report body.")

        logger.info("report_agent: LLM report generated (%d chars).", len(report))

    except Exception as exc:
        logger.error(
            "report_agent: Groq API failed (%s) — using fallback report.", exc
        )
        report = _build_fallback_report(results)
        report += (
            "\n\n---\n\n"
            "> ⚠️ **Note:** This report was generated locally due to an API error. "
            "Re-run the analysis for an LLM-enhanced report."
        )

    state["final_report"] = report
    state["status"] = "Report ready."
    return state