"""Build a golden evaluation set of sample documents with ground-truth fields.

Writes data/samples/golden.json: a list of {id, text, doc_type, truth} records covering
invoices (incl. Swiss VAT), receipts, and contracts -- some clean, some with missing or
malformed fields so the eval is discriminative, not a guaranteed 1.0.

    python scripts/build_golden.py
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path("data/samples/golden.json")

SAMPLES = [
    {
        "id": "inv_clean",
        "text": "INVOICE number INV-1001. Vendor: Alpine Supplies AG. "
                "Total: CHF 1081.00. MWST rate 8.1%. MWST amount 81.00. "
                "IBAN: CH9300762011623852957. Invoice date: 2026-03-15. Currency CHF",
        "doc_type": "invoice",
        "truth": {"invoice_number": "INV-1001", "total_amount": "1081.00",
                  "mwst_rate": "8.1", "invoice_date": "2026-03-15", "currency": "CHF"},
    },
    {
        "id": "inv_missing_date",
        "text": "INVOICE number INV-1002. Total: CHF 540.00. Currency CHF",
        "doc_type": "invoice",
        "truth": {"invoice_number": "INV-1002", "total_amount": "540.00", "currency": "CHF"},
    },
    {
        "id": "inv_paraphrased",
        "text": "Rechnung Nr INV-1003. Betrag: CHF 200.00. Datum: 2026-05-01",
        "doc_type": "invoice",
        "truth": {"invoice_number": "INV-1003", "total_amount": "200.00",
                  "invoice_date": "2026-05-01"},
    },
    {
        "id": "rec_clean",
        "text": "RECEIPT. Merchant: Coop. Total paid: CHF 45.20. Date: 2026-02-10",
        "doc_type": "receipt",
        "truth": {"merchant": "Coop", "total_amount": "45.20", "date": "2026-02-10"},
    },
    {
        "id": "rec_no_merchant",
        "text": "RECEIPT. Total: 12.90. Date: 2026-02-11",
        "doc_type": "receipt",
        "truth": {"total_amount": "12.90", "date": "2026-02-11"},
    },
    {
        "id": "con_clean",
        "text": "AGREEMENT between Alpha GmbH and Beta AG. Effective date: 2026-01-01. "
                "Term of 24 months.",
        "doc_type": "contract",
        "truth": {"party": "Alpha GmbH and Beta AG", "effective_date": "2026-01-01",
                  "term": "24 months"},
    },
]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(SAMPLES, indent=2))
    print(f"wrote {OUT}: {len(SAMPLES)} documents")


if __name__ == "__main__":
    main()
