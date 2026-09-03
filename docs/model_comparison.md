# Extraction backend comparison

Every extraction backend scored on the same golden set (15 documents: clean and noisy invoices with Swiss VAT, receipts, contracts). Regenerate with `python scripts/benchmark_extractors.py`.

| Backend | Field accuracy | Valid docs | Flagged for review | Latency / doc (ms) |
|---------|----------------|------------|--------------------|--------------------|
| rule | 0.833 | 9/15 | 12/15 | 0.31 |
| hf | 0.542 | 5/15 | 13/15 | 559.92 |

Skipped: llm. The LLM backend needs OPENAI_API_KEY; the HF backend needs `transformers` installed. Both are optional, so CI stays offline and dependency-light while the rule backend always runs.

## Per-field detail (rule)

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

Reading the table: the rule backend is fast and free but brittle on noisy or German-language text; an LLM backend trades latency and cost for robustness; the HF question-answering backend sits in between and needs no API key once the model is cached. The interface is identical, so swapping backends is one env var.
