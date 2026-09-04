# Real invoice evaluation

The rule backend scored against 26 real invoices from katanaml-org/invoices-donut-data-v1, whose ground truth was annotated independently of this repository. Unlike the synthetic golden set, nothing here was written to match the extraction patterns.

| Field | Correct | Expected | Accuracy |
|-------|---------|----------|----------|
| invoice_number | 26 | 26 | 1.00 |
| invoice_date | 26 | 26 | 1.00 |
| vendor | 25 | 26 | 0.96 |
| iban | 26 | 26 | 1.00 |

Overall field accuracy: **0.990** (103/104) on the four fields this dataset annotates.

**Straight-through processing: 0/26.** Every document is flagged for review, and field accuracy does not change that. The invoice schema requires a total amount, and this dataset annotates only the header, so the total is absent from every reconstructed document and the required-field rule fires each time. Field accuracy is what the extractor gets right; straight-through rate is what a back office actually buys, and on this data it is zero for a reason that is about the dataset rather than the extractor. On the synthetic set, which does carry totals, it is 4/18.

Caveat worth stating plainly: the dataset ships images without OCR text, so the document text here is reconstructed from the annotated fields. That means this measures the patterns against real invoice values and phrasing, not against scanner noise. The noise side is covered separately in docs/real_document_probe.md using FUNSD scans.
