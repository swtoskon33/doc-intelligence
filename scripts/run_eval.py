"""Run extraction + validation over the golden set and write a markdown eval report.

Produces docs/eval_report.md with per-field precision/recall/F1, overall field accuracy,
and document-level metrics (how many documents are fully valid / need review).

    python scripts/run_eval.py
"""
from __future__ import annotations

import json
from pathlib import Path

from doc_intelligence.eval.metrics import evaluate
from doc_intelligence.extraction.extractor import get_extractor
from doc_intelligence.ingest.documents import ingest_document
from doc_intelligence.types import DocumentType
from doc_intelligence.validation.rules import validate_in_place

GOLDEN = Path("data/samples/golden.json")
REPORT = Path("docs/eval_report.md")


def main() -> None:
    samples = json.loads(GOLDEN.read_text())
    ex = get_extractor()

    results = []
    truth = {}
    needs_review = 0
    valid = 0
    for s in samples:
        doc = ingest_document(s["id"], s["text"])
        # trust the labelled type so extraction uses the right schema
        doc.doc_type = DocumentType(s["doc_type"])
        res = validate_in_place(ex.extract(doc))
        results.append(res)
        truth[s["id"]] = s["truth"]
        needs_review += int(res.needs_review)
        valid += int(res.is_valid)

    report = evaluate(results, truth)
    n = len(samples)

    intro = (
        f"Golden set: {n} documents (invoices, receipts, contracts), extracted with "
        "the offline rule backend and validated. Regenerate with `python "
        "scripts/run_eval.py`."
    )
    lines = [
        "# Evaluation report",
        "",
        intro,
        "",
        "## Document-level",
        "",
        f"- Overall field accuracy: {report.overall_accuracy}",
        f"- Documents fully valid: {valid}/{n}",
        f"- Documents flagged for review: {needs_review}/{n}",
        "",
        "## Per-field",
        "",
        "| Field | Precision | Recall | F1 | Support |",
        "|-------|-----------|--------|-----|---------|",
    ]
    for name in sorted(report.per_field):
        if report.per_field[name].support == 0:
            continue
        s = report.per_field[name]
        lines.append(f"| {name} | {s.precision} | {s.recall} | {s.f1} | {s.support} |")

    note = (
        "Scores are not a flat 1.0 by design: the golden set includes documents with "
        "missing or paraphrased fields, so the rule backend misses some. The point is a "
        "discriminative harness that surfaces which fields are weak."
    )
    lines += ["", note, ""]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines))
    print(f"wrote {REPORT}")
    print(f"overall accuracy {report.overall_accuracy}, valid {valid}/{n}, review {needs_review}/{n}")


if __name__ == "__main__":
    main()
