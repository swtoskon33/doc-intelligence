# Real invoice evaluation

The rule backend scored against 26 real invoices from katanaml-org/invoices-donut-data-v1, whose ground truth was annotated independently of this repository. Unlike the synthetic golden set, nothing here was written to match the extraction patterns.

| Field | Correct | Expected | Accuracy |
|-------|---------|----------|----------|
| invoice_number | 26 | 26 | 1.00 |
| invoice_date | 26 | 26 | 1.00 |
| vendor | 25 | 26 | 0.96 |
| iban | 26 | 26 | 1.00 |

Overall field accuracy: **0.990** (103/104). 26/26 documents flagged for review.

Caveat worth stating plainly: the dataset ships images without OCR text, so the document text here is reconstructed from the annotated fields. That means this measures the patterns against real invoice values and phrasing, not against scanner noise. The noise side is covered separately in docs/real_document_probe.md using FUNSD scans.
