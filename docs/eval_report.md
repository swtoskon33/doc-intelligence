# Evaluation report

Golden set: 6 documents (invoices, receipts, contracts), extracted with the offline rule backend and validated. Regenerate with `python scripts/run_eval.py`.

## Document-level

- Overall field accuracy: 0.737
- Documents fully valid: 4/6
- Documents flagged for review: 4/6

## Per-field

| Field | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| currency | 0.667 | 1.0 | 0.8 | 2 |
| date | 1.0 | 1.0 | 1.0 | 2 |
| effective_date | 1.0 | 1.0 | 1.0 | 1 |
| invoice_date | 1.0 | 1.0 | 1.0 | 2 |
| invoice_number | 1.0 | 0.667 | 0.8 | 3 |
| merchant | 0.0 | 0.0 | 0.0 | 1 |
| mwst_rate | 1.0 | 1.0 | 1.0 | 1 |
| party | 0.0 | 0.0 | 0.0 | 1 |
| term | 1.0 | 1.0 | 1.0 | 1 |
| total_amount | 0.6 | 0.6 | 0.6 | 5 |

Scores are not a flat 1.0 by design: the golden set includes documents with missing or paraphrased fields, so the rule backend misses some. The point is a discriminative harness that surfaces which fields are weak.
