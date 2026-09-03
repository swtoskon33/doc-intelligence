# Extraction backend comparison

Every extraction backend scored on the same golden set (18 documents: clean and noisy invoices with Swiss VAT, receipts, contracts). Regenerate with `python scripts/benchmark_extractors.py`.

| Backend | Field accuracy | Valid docs | Flagged for review | Latency / doc (ms) |
|---------|----------------|------------|--------------------|--------------------|
| rule | 0.902 | 12/18 | 15/18 | 0.27 |
| hf | 0.541 | 5/18 | 16/18 | 411.2 |

Skipped: llm. The LLM backend needs OPENAI_API_KEY; the HF backend needs `transformers` installed. Both are optional, so CI stays offline and dependency-light while the rule backend always runs.

## Per-field detail (rule)

| Field | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| currency | 0.25 | 1.0 | 0.4 | 3 |
| date | 1.0 | 1.0 | 1.0 | 3 |
| effective_date | 1.0 | 1.0 | 1.0 | 2 |
| iban | 1.0 | 1.0 | 1.0 | 4 |
| invoice_date | 0.917 | 1.0 | 0.957 | 11 |
| invoice_number | 0.909 | 0.769 | 0.833 | 13 |
| merchant | 0.5 | 0.5 | 0.5 | 2 |
| mwst_rate | 0.857 | 1.0 | 0.923 | 6 |
| party | 0.0 | 0.0 | 0.0 | 2 |
| term | 1.0 | 1.0 | 1.0 | 1 |
| total_amount | 0.933 | 1.0 | 0.966 | 14 |

Reading the table: the rule backend is fast and free but brittle on noisy or German-language text; an LLM backend trades latency and cost for robustness; the HF question-answering backend sits in between and needs no API key once the model is cached. The interface is identical, so swapping backends is one env var.
