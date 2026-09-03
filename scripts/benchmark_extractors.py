"""Benchmark every extraction backend on the same golden set.

Scores rule / LLM / HF side by side: per-field precision, recall and F1, document-level
accuracy, review rate, and latency per document. Backends that are unavailable (no API
key, transformers not installed) are reported as skipped rather than failing the run.

    python scripts/benchmark_extractors.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from doc_intelligence.eval.metrics import evaluate
from doc_intelligence.extraction.extractor import get_extractor
from doc_intelligence.ingest.documents import ingest_document
from doc_intelligence.types import DocumentType
from doc_intelligence.validation.rules import validate_in_place

GOLDEN = Path("data/samples/golden.json")
OUT = Path("docs/model_comparison.md")
BACKENDS = ["rule", "llm", "hf"]


def _load():
    samples = json.loads(GOLDEN.read_text())
    docs = []
    truth = {}
    for s in samples:
        doc = ingest_document(s["id"], s["text"])
        doc.doc_type = DocumentType(s["doc_type"])
        docs.append(doc)
        truth[s["id"]] = s["truth"]
    return docs, truth


def run_backend(name, docs, truth):
    ex = get_extractor(name)
    if not getattr(ex, "available", True):
        return None
    results = []
    t0 = time.perf_counter()
    review = 0
    valid = 0
    for doc in docs:
        res = validate_in_place(ex.extract(doc))
        results.append(res)
        review += int(res.needs_review)
        valid += int(res.is_valid)
    elapsed_ms = (time.perf_counter() - t0) * 1000 / len(docs)
    report = evaluate(results, truth)
    return {
        "name": name,
        "accuracy": report.overall_accuracy,
        "valid": valid,
        "review": review,
        "latency_ms": round(elapsed_ms, 2),
        "per_field": report.per_field,
    }


def main() -> None:
    docs, truth = _load()
    n = len(docs)
    rows = []
    skipped = []
    for name in BACKENDS:
        out = run_backend(name, docs, truth)
        if out is None:
            skipped.append(name)
        else:
            rows.append(out)

    intro = (
        f"Every extraction backend scored on the same golden set ({n} documents: clean "
        "and noisy invoices with Swiss VAT, receipts, contracts). Regenerate with "
        "`python scripts/benchmark_extractors.py`."
    )
    lines = [
        "# Extraction backend comparison",
        "",
        intro,
        "",
        "| Backend | Field accuracy | Valid docs | Flagged for review | Latency / doc (ms) |",
        "|---------|----------------|------------|--------------------|--------------------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['accuracy']} | {r['valid']}/{n} | {r['review']}/{n} "
            f"| {r['latency_ms']} |"
        )

    if skipped:
        note = (
            "Skipped: " + ", ".join(skipped) + ". The LLM backend needs OPENAI_API_KEY and "
            "the HF backend needs `transformers` installed; both are optional so CI stays "
            "offline and dependency-light. The rule backend always runs."
        )
        lines += ["", note]

    # per-field table for the best-performing available backend
    if rows:
        best = max(rows, key=lambda r: r["accuracy"])
        lines += ["", f"## Per-field detail ({best['name']})", "",
                  "| Field | Precision | Recall | F1 | Support |",
                  "|-------|-----------|--------|-----|---------|"]
        for fname in sorted(best["per_field"]):
            s = best["per_field"][fname]
            if s.support == 0:
                continue
            lines.append(f"| {fname} | {s.precision} | {s.recall} | {s.f1} | {s.support} |")

    takeaway = (
        "Reading the table: the rule backend is fast and free but brittle on noisy or "
        "German-language text; an LLM backend trades latency and cost for robustness; the "
        "HF question-answering backend sits in between and needs no API key once the model "
        "is cached. The interface is identical, so swapping backends is one env var."
    )
    lines += ["", takeaway, ""]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT}")
    for r in rows:
        print(f"  {r['name']}: acc={r['accuracy']} review={r['review']}/{n} latency={r['latency_ms']}ms")
    if skipped:
        print("  skipped:", ", ".join(skipped))


if __name__ == "__main__":
    main()
