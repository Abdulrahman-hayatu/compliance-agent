"""
eval/run_eval.py

Golden-dataset evaluation harness for the compliance-agent pipeline.
Reuses the REAL retrieval_agent and checker_agent functions unmodified --
this measures the actual production pipeline, not a reimplementation that
could silently drift from what's deployed.

Two independent passes, run separately because their cost is wildly
different:

  --retrieval-only   Free -- zero Groq API calls. Runs the embedding-based
                      retriever against every golden claim and checks
                      whether the expected clause ID(s) actually get
                      surfaced (via metrics.compute_context_recall). Run
                      this FIRST: a broken retrieval layer means the
                      verdict pass below would just be scoring the
                      checker against bad context, wasting quota on a
                      run that can't succeed regardless of prompt quality.

  --verdict          Costs real Groq tokens -- one checker_agent call per
                      claim in the batch (60 calls for the full set).
                      Compares the actual verdict against expected_verdict
                      via metrics.compute_verdict_metrics and
                      compute_metrics_by_difficulty.

Usage:
    python -m eval.run_eval --retrieval-only
    python -m eval.run_eval --verdict
    python -m eval.run_eval --retrieval-only --verdict     # both, retrieval first
    python -m eval.run_eval --verdict --limit 10           # smoke-test on a subset
    python -m eval.run_eval --verdict --output report.json # save full results
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.retrieval_agent import retrieval_agent
from agents.checker_agent import checker_agent
from eval.metrics import (
    compute_verdict_metrics,
    compute_context_recall,
    compute_metrics_by_difficulty,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

GOLDEN_PATH = Path(__file__).parent / "golden_dataset.jsonl"


def load_golden_dataset(path: Path = GOLDEN_PATH, limit: int | None = None) -> list[dict]:
    items = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    claims = [i["claim"] for i in items]
    if len(claims) != len(set(claims)):
        raise ValueError(
            "Golden dataset has duplicate claim text -- items are matched back "
            "to results by claim text, so duplicates would corrupt scoring."
        )
    if limit is not None:
        items = items[:limit]
    return items


def run_retrieval_pass(items: list[dict]) -> dict:
    logger.info("=== Retrieval-only pass: %d claims, zero Groq API cost ===", len(items))

    state = {"policy_claims": [i["claim"] for i in items]}
    state = retrieval_agent(state)
    retrieved_by_claim = state.get("retrieved_chunks", {})

    cr_items = []
    for item in items:
        retrieved_chunks = retrieved_by_claim.get(item["claim"], [])
        retrieved_ids = [c["id"] for c in retrieved_chunks]
        cr_items.append({
            "expected_clause_ids": item["expected_clause_ids"],
            "retrieved_ids": retrieved_ids,
            "_id": item["id"],
            "_claim": item["claim"],
            "_difficulty": item["difficulty"],
        })

    report = compute_context_recall(cr_items)

    logger.info(
        "Mean recall: %s | applicable=%d | excluded (out_of_scope)=%d",
        report["mean_recall"], report["applicable_items"], report["excluded_out_of_scope"],
    )
    logger.info(
        "  full=%d partial=%d zero=%d",
        report["full_recall_count"], report["partial_recall_count"], report["zero_recall_count"],
    )

    zero_items = [
        it for it in cr_items
        if it["expected_clause_ids"]
        and not (set(it["expected_clause_ids"]) & set(it["retrieved_ids"]))
    ]
    if zero_items:
        logger.info("--- Zero-recall items (nothing expected was retrieved) ---")
        for it in zero_items:
            logger.info(
                "  %s [%s]: expected=%s retrieved=%s | %s",
                it["_id"], it["_difficulty"], it["expected_clause_ids"],
                it["retrieved_ids"], it["_claim"][:80],
            )

    return report


def run_verdict_pass(items: list[dict]) -> dict:
    logger.info("=== Verdict pass: %d claims -- WILL call the Groq API ===", len(items))

    state = {"policy_claims": [i["claim"] for i in items]}
    state = retrieval_agent(state)
    state = checker_agent(state)

    results_by_claim = {r["claim"]: r for r in state.get("compliance_results", [])}

    pairs = []
    diff_items = []
    mismatches = []
    for item in items:
        result = results_by_claim.get(item["claim"])
        if result is None:
            logger.warning("No checker result for %s -- excluded from scoring", item["id"])
            continue
        expected, actual = item["expected_verdict"], result["status"]
        pairs.append((expected, actual))
        diff_items.append({
            "difficulty": item["difficulty"],
            "expected_verdict": expected,
            "actual_verdict": actual,
        })
        if expected != actual:
            mismatches.append({
                "id": item["id"],
                "claim": item["claim"][:80],
                "expected": expected,
                "actual": actual,
                "actual_reference": result.get("regulation_reference"),
                "actual_explanation": (result.get("explanation") or "")[:150],
            })

    verdict_report = compute_verdict_metrics(pairs)
    by_difficulty = compute_metrics_by_difficulty(diff_items)

    logger.info("Accuracy: %s | Macro F1: %s", verdict_report["accuracy"], verdict_report["macro_f1"])
    for label, m in verdict_report["per_label"].items():
        logger.info(
            "  %-14s precision=%s recall=%s f1=%s (support=%d)",
            label, m["precision"], m["recall"], m["f1"], m["support"],
        )
    logger.info("By difficulty tier:")
    for tier, m in by_difficulty.items():
        logger.info("  %-14s accuracy=%s (n=%d)", tier, m["accuracy"], m["count"])

    if mismatches:
        logger.info("--- Mismatches ---")
        for m in mismatches:
            logger.info(
                "  %s: expected=%s actual=%s (ref=%s) | %s",
                m["id"], m["expected"], m["actual"], m["actual_reference"], m["claim"],
            )

    return {
        "verdict_metrics": verdict_report,
        "by_difficulty": by_difficulty,
        "mismatches": mismatches,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Golden-dataset evaluation harness")
    parser.add_argument("--retrieval-only", action="store_true",
                         help="Run the free retrieval-quality pass")
    parser.add_argument("--verdict", action="store_true",
                         help="Run the token-costing verdict-accuracy pass")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only evaluate the first N golden examples (for smoke-testing)")
    parser.add_argument("--output", type=Path, default=None,
                         help="Optional path to write the full JSON report")
    args = parser.parse_args()

    if not args.retrieval_only and not args.verdict:
        parser.error("Specify at least one of --retrieval-only or --verdict")

    items = load_golden_dataset(limit=args.limit)
    logger.info("Loaded %d golden examples from %s\n", len(items), GOLDEN_PATH)

    full_report: dict = {}
    if args.retrieval_only:
        full_report["retrieval"] = run_retrieval_pass(items)
        print()
    if args.verdict:
        full_report["verdict"] = run_verdict_pass(items)

    if args.output:
        args.output.write_text(json.dumps(full_report, indent=2), encoding="utf-8")
        logger.info("\nFull report written to %s", args.output)


if __name__ == "__main__":
    main()
