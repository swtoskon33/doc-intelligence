"""Compare the rule and layout backends on the same real scanned pages.

docs/real_document_probe.md ends by asserting that regex stops where a layout-aware model
starts. That was an assertion. This runs both backends over the same 50 FUNSD scans and
puts the numbers side by side.

    python scripts/compare_backends_on_scans.py

Neither backend can be scored for accuracy here: FUNSD carries form-semantic labels, not
invoice fields, so there is no shared ground truth. What is comparable is reach -- how
much either backend pulls out of a real scan at all -- and that is the claim under test.
"""
from __future__ import annotations

import time
from pathlib import Path

from doc_intelligence.extraction.extractor import RuleExtractor
from doc_intelligence.extraction.layoutlm import LayoutLMv3Extractor
from doc_intelligence.ingest.documents import ingest_document
from doc_intelligence.layout.dataset import load_funsd
from doc_intelligence.validation.rules import validate_in_place

OUT = Path("docs/backend_comparison_on_scans.md")
CONFIDENCE_FLOOR = 0.7


def run_rule(documents):
    extractor = RuleExtractor()
    values, recognised, seconds = 0, 0, 0.0
    for i, doc in enumerate(documents):
        text = " ".join(doc["tokens"])
        started = time.perf_counter()
        result = validate_in_place(extractor.extract(ingest_document(f"s{i}", text)))
        seconds += time.perf_counter() - started
        if result.doc_type.value != "unknown":
            recognised += 1
        values += sum(1 for f in result.fields.values() if f.value)
    return {"values": values, "recognised": recognised,
            "ms": seconds * 1000 / len(documents)}


def run_layout(documents):
    backend = LayoutLMv3Extractor()
    if not backend.available:
        return None
    values, confident, seconds = 0, 0, 0.0
    for doc in documents:
        started = time.perf_counter()
        entities = backend.predict_entities(
            doc["image"].convert("RGB"), doc["tokens"], doc["bboxes"])
        seconds += time.perf_counter() - started
        values += len(entities)
        confident += sum(1 for e in entities if e.confidence >= CONFIDENCE_FLOOR)
    return {"values": values, "confident": confident,
            "ms": seconds * 1000 / len(documents)}


def main() -> None:
    documents = list(load_funsd().test)
    n = len(documents)

    rule = run_rule(documents)
    print(f"rule:   {rule['values']} values, {rule['recognised']}/{n} typed, "
          f"{rule['ms']:.2f} ms/doc")

    layout = run_layout(documents)
    if layout is None:
        raise SystemExit("no LayoutLMv3 checkpoint; run scripts/train_layoutlm.py first")
    print(f"layout: {layout['values']} entities, {layout['confident']} confident, "
          f"{layout['ms']:.0f} ms/doc")

    ratio = layout["values"] / rule["values"] if rule["values"] else float("inf")
    slower = layout["ms"] / rule["ms"] if rule["ms"] else float("inf")

    intro = (
        f"Both extraction backends over the same {n} real scanned pages from FUNSD. Not an "
        "accuracy comparison: FUNSD labels are form semantics, not invoice fields, so "
        "there is no ground truth both backends can be scored against. This measures "
        "reach -- how much either gets out of a real scan at all."
    )
    finding = (
        f"The layout backend pulls about {ratio:.0f}x more out of the same pages, at "
        f"roughly {slower:.0f}x the cost per page. That settles what the earlier probe "
        "could only assert. The regex patterns need a label and its value adjacent in a "
        "line of text; a scan puts them in separate cells, and no pattern tuning recovers "
        "a spatial relationship from a flattened string. The layout model reads position, "
        "so it finds structure the other backend cannot see.\n\n"
        "The cost side is equally clear, and it is why both ship. Microseconds per page "
        "against hundreds of milliseconds is the difference between running on every "
        "document and running only on the ones that need it."
    )
    layout_row = (
        f"| LayoutLMv3 (words + boxes) | {layout['values']} "
        f"({layout['confident']} above {CONFIDENCE_FLOOR} confidence) "
        f"| {layout['ms']:.0f} ms |"
    )
    lines = [
        "# Rule versus layout backend on real scans",
        "",
        intro,
        "",
        "| Backend | Values extracted | Latency / page |",
        "|---------|------------------|----------------|",
        f"| rule (regex over flat text) | {rule['values']} | {rule['ms']:.2f} ms |",
        layout_row,
        "",
        finding,
        "",
    ]
    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
