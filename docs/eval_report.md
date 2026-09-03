# Evaluation report

Golden set: 12 documents (invoices, receipts, contracts), extracted with the offline rule backend and validated. Regenerate with `python scripts/run_eval.py`.

## Document-level

- Overall field accuracy: 0.829
- Documents fully valid: 8/12
- Documents flagged for review: 9/12

## Per-field

| Field | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| currency | 0.429 | 1.0 | 0.6 | 3 |
| date | 1.0 | 1.0 | 1.0 | 3 |
| effective_date | 1.0 | 1.0 | 1.0 | 2 |
| iban | 1.0 | 1.0 | 1.0 | 2 |
| invoice_date | 1.0 | 1.0 | 1.0 | 6 |
| invoice_number | 0.833 | 0.714 | 0.769 | 7 |
| merchant | 0.5 | 0.5 | 0.5 | 2 |
| mwst_rate | 1.0 | 1.0 | 1.0 | 4 |
| party | 0.0 | 0.0 | 0.0 | 2 |
| term | 1.0 | 1.0 | 1.0 | 1 |
| total_amount | 0.778 | 0.778 | 0.778 | 9 |

Scores are not a flat 1.0 by design: the golden set includes documents with missing or paraphrased fields, so the rule backend misses some. The point is a discriminative harness that surfaces which fields are weak.
