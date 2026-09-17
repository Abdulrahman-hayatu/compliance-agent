"""
eval/metrics.py

Pure metric-computation functions for the golden-dataset eval harness.
No API calls, no I/O — takes already-collected (expected, actual) pairs
and computes verdict accuracy and retrieval context recall. Kept separate
from run_eval.py so these can be unit-tested with synthetic data, without
needing a live Groq call or the real index.
"""

from __future__ import annotations

from collections import Counter

LABELS = ("COMPLIANT", "NON_COMPLIANT", "UNCLEAR")


def compute_verdict_metrics(pairs: list[tuple[str, str]]) -> dict:
    """
    pairs: list of (expected_verdict, actual_verdict) tuples.

    Returns per-label precision/recall/F1, macro-F1, overall accuracy,
    and a confusion matrix (dict of dicts: confusion[expected][actual]).

    Precision/recall are computed per label using the standard
    one-vs-rest definition: for label L, TP = predicted L when expected
    L; FP = predicted L when expected something else; FN = expected L
    but predicted something else.
    """
    confusion = {e: {a: 0 for a in LABELS} for e in LABELS}
    for expected, actual in pairs:
        if expected not in LABELS or actual not in LABELS:
            raise ValueError(f"Unrecognized label in pair: ({expected!r}, {actual!r})")
        confusion[expected][actual] += 1

    per_label = {}
    for label in LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[e][label] for e in LABELS if e != label)
        fn = sum(confusion[label][a] for a in LABELS if a != label)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        per_label[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": sum(confusion[label].values()),  # how many golden examples have this expected label
        }

    total = len(pairs)
    correct = sum(1 for e, a in pairs if e == a)
    accuracy = correct / total if total > 0 else 0.0
    macro_f1 = sum(m["f1"] for m in per_label.values()) / len(LABELS)

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "per_label": per_label,
        "confusion_matrix": confusion,
        "total_examples": total,
    }


def compute_context_recall(
    items: list[dict],
) -> dict:
    """
    items: list of dicts, each with:
        - 'expected_clause_ids': list[str]
        - 'retrieved_ids': list[str]

    Items with an empty expected_clause_ids (out_of_scope claims — nothing
    in the corpus should match) are excluded from the recall calculation
    entirely, since "recall" is undefined when nothing was expected to be
    retrieved. Reported separately as a count, not folded into the mean.

    Per-item recall = |expected ∩ retrieved| / |expected|.
    Returns the mean recall across applicable items, plus a breakdown of
    how many items got full/partial/zero recall, for diagnosing WHERE
    retrieval is failing, not just that it's failing.
    """
    applicable = [it for it in items if it["expected_clause_ids"]]
    excluded_count = len(items) - len(applicable)

    if not applicable:
        return {
            "mean_recall": None,
            "applicable_items": 0,
            "excluded_out_of_scope": excluded_count,
            "full_recall_count": 0,
            "partial_recall_count": 0,
            "zero_recall_count": 0,
        }

    recalls = []
    full = partial = zero = 0
    for it in applicable:
        expected = set(it["expected_clause_ids"])
        retrieved = set(it["retrieved_ids"])
        hit = len(expected & retrieved)
        recall = hit / len(expected)
        recalls.append(recall)
        if recall == 1.0:
            full += 1
        elif recall == 0.0:
            zero += 1
        else:
            partial += 1

    return {
        "mean_recall": round(sum(recalls) / len(recalls), 4),
        "applicable_items": len(applicable),
        "excluded_out_of_scope": excluded_count,
        "full_recall_count": full,
        "partial_recall_count": partial,
        "zero_recall_count": zero,
    }


def compute_metrics_by_difficulty(
    items: list[dict],
) -> dict:
    """
    items: list of dicts, each with 'difficulty', 'expected_verdict',
    'actual_verdict'.

    Breaks verdict accuracy down per difficulty tier (easy / multi_hop /
    adversarial / out_of_scope), so a strong overall accuracy number
    can't hide a tier that's actually failing — e.g. if adversarial
    accuracy is much lower than easy accuracy, that's the signal that
    actually matters for judging the checker's reasoning quality.
    """
    by_tier: dict[str, list[tuple[str, str]]] = {}
    for it in items:
        by_tier.setdefault(it["difficulty"], []).append(
            (it["expected_verdict"], it["actual_verdict"])
        )

    result = {}
    for tier, pairs in by_tier.items():
        correct = sum(1 for e, a in pairs if e == a)
        result[tier] = {
            "accuracy": round(correct / len(pairs), 4) if pairs else 0.0,
            "count": len(pairs),
        }
    return result


# ── Self-test with synthetic data (no API, no real index needed) ──────────
if __name__ == "__main__":
    print("Running metrics.py self-test with synthetic data...\n")

    # Synthetic verdict pairs: a mix of correct/incorrect across all labels
    pairs = [
        ("COMPLIANT", "COMPLIANT"),
        ("COMPLIANT", "COMPLIANT"),
        ("COMPLIANT", "UNCLEAR"),       # FN for COMPLIANT, FP for UNCLEAR
        ("NON_COMPLIANT", "NON_COMPLIANT"),
        ("NON_COMPLIANT", "COMPLIANT"), # dangerous miss: NON_COMPLIANT called COMPLIANT
        ("UNCLEAR", "UNCLEAR"),
        ("UNCLEAR", "UNCLEAR"),
    ]
    vm = compute_verdict_metrics(pairs)
    print("Verdict metrics:")
    print(f"  accuracy = {vm['accuracy']}, macro_f1 = {vm['macro_f1']}")
    for label, m in vm["per_label"].items():
        print(f"  {label}: P={m['precision']} R={m['recall']} F1={m['f1']} support={m['support']}")
    assert vm["accuracy"] == round(5 / 7, 4)
    assert vm["per_label"]["COMPLIANT"]["support"] == 3
    print("  OK\n")

    # Synthetic context-recall items
    cr_items = [
        {"expected_clause_ids": ["a", "b"], "retrieved_ids": ["a", "b", "c"]},  # full
        {"expected_clause_ids": ["x"], "retrieved_ids": ["y", "z"]},            # zero
        {"expected_clause_ids": ["p", "q"], "retrieved_ids": ["p"]},            # partial
        {"expected_clause_ids": [], "retrieved_ids": []},                       # out_of_scope, excluded
    ]
    cr = compute_context_recall(cr_items)
    print("Context recall:")
    print(f"  mean_recall = {cr['mean_recall']}, applicable = {cr['applicable_items']}, excluded = {cr['excluded_out_of_scope']}")
    print(f"  full={cr['full_recall_count']} partial={cr['partial_recall_count']} zero={cr['zero_recall_count']}")
    assert cr["applicable_items"] == 3
    assert cr["excluded_out_of_scope"] == 1
    assert cr["full_recall_count"] == 1
    assert cr["zero_recall_count"] == 1
    assert cr["partial_recall_count"] == 1
    print("  OK\n")

    # Synthetic difficulty breakdown
    diff_items = [
        {"difficulty": "easy", "expected_verdict": "COMPLIANT", "actual_verdict": "COMPLIANT"},
        {"difficulty": "easy", "expected_verdict": "NON_COMPLIANT", "actual_verdict": "NON_COMPLIANT"},
        {"difficulty": "adversarial", "expected_verdict": "COMPLIANT", "actual_verdict": "NON_COMPLIANT"},
    ]
    db = compute_metrics_by_difficulty(diff_items)
    print("By-difficulty accuracy:", db)
    assert db["easy"]["accuracy"] == 1.0
    assert db["adversarial"]["accuracy"] == 0.0
    print("  OK\n")

    print("All self-tests passed.")
