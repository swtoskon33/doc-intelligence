# Evaluation report

Golden set: 15 synthetic documents (invoices, receipts, contracts), extracted with the offline rule backend and validated. Regenerate with `python scripts/run_eval.py`.

## Document-level

- Overall field accuracy: 0.833
- Documents fully valid: 9/15
- Documents flagged for review: 12/15

## Per-field

| Field | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| currency | 0.3 | 1.0 | 0.462 | 3 |
| date | 1.0 | 1.0 | 1.0 | 3 |
| effective_date | 1.0 | 1.0 | 1.0 | 2 |
| iban | 1.0 | 1.0 | 1.0 | 2 |
| invoice_date | 0.875 | 0.875 | 0.875 | 8 |
| invoice_number | 0.889 | 0.8 | 0.842 | 10 |
| merchant | 0.5 | 0.5 | 0.5 | 2 |
| mwst_rate | 0.8 | 1.0 | 0.889 | 4 |
| party | 0.0 | 0.0 | 0.0 | 2 |
| term | 1.0 | 1.0 | 1.0 | 1 |
| total_amount | 0.75 | 0.818 | 0.783 | 11 |

Scores are not a flat 1.0 by design: the golden set includes documents with missing or paraphrased fields, so the rule backend misses some. The point is a discriminative harness that surfaces which fields are weak.
