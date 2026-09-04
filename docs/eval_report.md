# Evaluation report

Golden set: 18 synthetic documents (invoices, receipts, contracts), extracted with the offline rule backend and validated. Regenerate with `python scripts/run_eval.py`.

## Document-level

- Overall field accuracy: 0.984
- Documents fully valid: 14/18
- Documents flagged for review: 15/18

## Per-field

| Field | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| currency | 0.25 | 1.0 | 0.4 | 3 |
| date | 1.0 | 1.0 | 1.0 | 3 |
| effective_date | 1.0 | 1.0 | 1.0 | 2 |
| iban | 1.0 | 1.0 | 1.0 | 4 |
| invoice_date | 0.917 | 1.0 | 0.957 | 11 |
| invoice_number | 0.923 | 0.923 | 0.923 | 13 |
| merchant | 1.0 | 1.0 | 1.0 | 2 |
| mwst_rate | 0.857 | 1.0 | 0.923 | 6 |
| party | 1.0 | 1.0 | 1.0 | 2 |
| term | 1.0 | 1.0 | 1.0 | 1 |
| total_amount | 0.933 | 1.0 | 0.966 | 14 |

Scores are not a flat 1.0 by design: the golden set includes documents with missing or paraphrased fields, so the rule backend misses some. The point is a discriminative harness that surfaces which fields are weak.
