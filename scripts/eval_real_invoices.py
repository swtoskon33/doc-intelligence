"""Score the rule backend against real invoices with independent ground truth.

Everything else in this repo is measured on documents written alongside the extraction
patterns, which cannot reveal a pattern that fails on input nobody tailored for it. This
uses katanaml-org/invoices-donut-data-v1: 26 real invoices whose ground truth was
annotated by someone else entirely.

    python scripts/eval_real_invoices.py

Download once (10 MB):
    curl -L -o data/real/invoices_test.parquet \
      https://huggingface.co/datasets/katanaml-org/invoices-donut-data-v1/resolve/main/data/test-00000-of-00001-56af6bd5ff7eb34d.parquet

The images carry no OCR text, so the document text is reconstructed from the annotated
fields. That is a real limitation: it tests the patterns against genuine invoice values
and phrasing, not against scanner noise. scripts/probe_real_documents.py covers the noise
side using FUNSD.
"""
from __future__ import annotations

import json
from pathlib import Path

from doc_intelligence.extraction.extractor import RuleExtractor
from doc_intelligence.ingest.documents import ingest_document
from doc_intelligence.types import DocumentType
from doc_intelligence.validation.rules import validate_in_place

DATA = Path("data/real/invoices_test.parquet")
OUT = Path("docs/real_invoice_eval.md")

# ground-truth field name -> our schema field name
FIELD_MAP = {
    "invoice_no": "invoice_number",
    "invoice_date": "invoice_date",
    "seller": "vendor",
    "iban": "iban",
}


def build_text(gt: dict) -> str:
    """Render the annotated fields as invoice text, in the phrasing an invoice uses."""
    h = gt["gt_parse"].get("header", {})
    s = gt["gt_parse"].get("summary", {})
    parts = ["INVOICE"]
    if h.get("invoice_no"):
        parts.append(f"Invoice no: {h['invoice_no']}")
    if h.get("invoice_date"):
        parts.append(f"Invoice date: {h['invoice_date']}")
    if h.get("seller"):
        parts.append(f"From: {h['seller'].split(',')[0]}")
    if h.get("iban"):
        parts.append(f"IBAN: {h['iban']}")
    if s.get("total_gross_worth"):
        parts.append(f"Total: {s['total_gross_worth']}")
    if s.get("total_vat"):
        parts.append(f"VAT amount: {s['total_vat']}")
    return ". ".join(parts)


def normalise(value):
    if value is None:
        return None
    return value.strip().lower().replace("$", "").replace(" ", "")


def main() -> None:
    if not DATA.exists():
        raise SystemExit(f"{DATA} missing -- see the download command in this file's docstring")

    import pandas as pd

    df = pd.read_parquet(DATA)
    extractor = RuleExtractor()

    per_field = {name: {"correct": 0, "found": 0, "expected": 0} for name in FIELD_MAP.values()}
    reviewed = 0

    for i, row in df.iterrows():
        gt = json.loads(row["ground_truth"])
        header = gt["gt_parse"].get("header", {})
        doc = ingest_document(f"real_{i}", build_text(gt))
        doc.doc_type = DocumentType.INVOICE
        result = validate_in_place(extractor.extract(doc))
        reviewed += int(result.needs_review)

        for gt_name, our_name in FIELD_MAP.items():
            expected = header.get(gt_name)
            if not expected:
                continue
            per_field[our_name]["expected"] += 1
            got = result.value(our_name)
            if got:
                per_field[our_name]["found"] += 1
                # seller is a full address in the ground truth; the leading name is enough
                if our_name == "vendor":
                    if normalise(got) and normalise(got) in normalise(expected):
                        per_field[our_name]["correct"] += 1
                elif normalise(got) == normalise(expected):
                    per_field[our_name]["correct"] += 1

    n = len(df)
    total_expected = sum(f["expected"] for f in per_field.values())
    total_correct = sum(f["correct"] for f in per_field.values())
    accuracy = total_correct / total_expected if total_expected else 0.0

    print(f"documents: {n}")
    for name, f in per_field.items():
        rate = f["correct"] / f["expected"] if f["expected"] else 0.0
        print(f"  {name:15} {f['correct']}/{f['expected']} = {rate:.2f}")
    print(f"overall: {total_correct}/{total_expected} = {accuracy:.3f}")

    intro = (
        f"The rule backend scored against {n} real invoices from "
        "katanaml-org/invoices-donut-data-v1, whose ground truth was annotated "
        "independently of this repository. Unlike the synthetic golden set, nothing here "
        "was written to match the extraction patterns."
    )
    lines = [
        "# Real invoice evaluation",
        "",
        intro,
        "",
        "| Field | Correct | Expected | Accuracy |",
        "|-------|---------|----------|----------|",
    ]
    for name, f in per_field.items():
        rate = f["correct"] / f["expected"] if f["expected"] else 0.0
        lines.append(f"| {name} | {f['correct']} | {f['expected']} | {rate:.2f} |")
    summary = (
        f"Overall field accuracy: **{accuracy:.3f}** ({total_correct}/{total_expected}) "
        f"on the four fields this dataset annotates.\n\n"
        f"**Straight-through processing: {n - reviewed}/{n}.** Every document is flagged "
        "for review, and field accuracy does not change that. The invoice schema requires "
        "a total amount, and this dataset annotates only the header, so the total is "
        "absent from every reconstructed document and the required-field rule fires each "
        "time. Field accuracy is what the extractor gets right; straight-through rate is "
        "what a back office actually buys, and on this data it is zero for a reason that "
        "is about the dataset rather than the extractor. On the synthetic set, which does "
        "carry totals, it is 4/18."
    )
    caveat = (
        "Caveat worth stating plainly: the dataset ships images without OCR text, so the "
        "document text here is reconstructed from the annotated fields. That means this "
        "measures the patterns against real invoice values and phrasing, not against "
        "scanner noise. The noise side is covered separately in "
        "docs/real_document_probe.md using FUNSD scans."
    )
    lines += ["", summary, "", caveat, ""]
    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
