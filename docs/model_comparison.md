# Extraction backend comparison

Every extraction backend scored on the same golden set (12 documents: clean and noisy invoices with Swiss VAT, receipts, contracts). Regenerate with `python scripts/benchmark_extractors.py`.

| Backend | Field accuracy | Valid docs | Flagged for review | Latency / doc (ms) |
|---------|----------------|------------|--------------------|--------------------|
| rule | 0.829 | 8/12 | 9/12 | 0.42 |
| hf | 0.561 | 6/12 | 10/12 | 530.4 |

Skipped: llm. The LLM backend needs OPENAI_API_KEY; the HF backend needs `transformers` installed. Both are optional, so CI stays offline and dependency-light while the rule backend always runs.

## Per-field detail (rule)

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

Reading the table: the rule backend is fast and free but brittle on noisy or German-language text; an LLM backend trades latency and cost for robustness; the HF question-answering backend sits in between and needs no API key once the model is cached. The interface is identical, so swapping backends is one env var.
