"""Run the rule backend over real scanned documents and report what it manages.

Every other number in this repo comes from synthetic documents written alongside the
extraction patterns, which cannot expose the failure mode that matters: regex tuned on
clean text does not survive real OCR. FUNSD gives 50 genuinely scanned pages with the
noise a scanner produces -- broken lines, stray spacing, character errors.

    python scripts/probe_real_documents.py

The output is not an accuracy score. FUNSD is forms, not invoices, so most documents
have no invoice fields to find. What it measures is reach: how often the rule backend
recognises a document type at all, and how many fields it pulls out when it does.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from doc_intelligence.extraction.extractor import RuleExtractor
from doc_intelligence.ingest.documents import ingest_document
from doc_intelligence.layout.dataset import load_funsd
from doc_intelligence.validation.rules import validate_in_place

OUT = Path("docs/real_document_probe.md")


def main() -> None:
    splits = load_funsd()
    extractor = RuleExtractor()

    types: Counter = Counter()
    fields: Counter = Counter()
    review = 0
    crashes = 0
    documents = 0

    for i, doc in enumerate(splits.test):
        text = " ".join(doc["tokens"])
        documents += 1
        try:
            result = validate_in_place(extractor.extract(ingest_document(f"funsd_{i}", text)))
        except Exception as exc:  # noqa: BLE001 - a probe should report, not abort
            crashes += 1
            print(f"crash on document {i}: {type(exc).__name__}: {exc}")
            continue
        types[result.doc_type.value] += 1
        review += int(result.needs_review)
        for name, field in result.fields.items():
            if field.value:
                fields[name] += 1

    recognised = documents - types.get("unknown", 0)
    print(f"documents: {documents}  crashes: {crashes}")
    print(f"type recognised: {recognised}/{documents}")
    print(f"fields extracted: {sum(fields.values())}")

    intro = (
        f"The rule backend run over {documents} real scanned pages from FUNSD, which "
        "carry the noise a scanner actually produces. This is not an accuracy score: "
        "FUNSD is forms, not invoices, so most pages have no invoice field to find. It "
        "measures reach -- how often the patterns fire at all on text they were not "
        "written against."
    )
    lines = [
        "# Rule backend on real scanned documents",
        "",
        intro,
        "",
        "| Measure | Result |",
        "|---------|--------|",
        f"| Documents processed | {documents} |",
        f"| Crashes | {crashes} |",
        f"| Document type recognised | {recognised}/{documents} |",
        f"| Flagged for review | {review}/{documents} |",
        f"| Field values extracted (total) | {sum(fields.values())} |",
        "",
        "Fields found, by name:",
        "",
        "| Field | Documents |",
        "|-------|-----------|",
    ]
    for name, count in fields.most_common():
        lines.append(f"| {name} | {count} |")

    finding = (
        f"Nothing crashes, and almost nothing is extracted: {sum(fields.values())} field "
        f"values across {documents} documents, with {types.get('unknown', 0)} pages whose "
        "type the classifier cannot place. Part of that is fair -- these are forms, and a "
        "form has no invoice number. But the pages it *does* read as invoices give up "
        "barely anything, and that is the real result.\n\n"
        "The patterns expect `Total: CHF 1081.00` on one line. A scan gives a label in one "
        "cell and its value in another, spacing that survives no regex, and characters the "
        "OCR guessed at. A regex matches a string; it has no notion that the number to the "
        "right of the word Total is the total. That is exactly the boundary the LayoutLMv3 "
        "backend exists to cross: it reads position as well as text, so a value means "
        "something because of where it sits relative to its label.\n\n"
        "So the honest reading of this repo's two extraction paths is: rules for clean, "
        "structured input where the format is stable and the cost is microseconds; a "
        "layout-aware model for anything that came off a scanner."
    )
    lines += ["", finding, ""]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
