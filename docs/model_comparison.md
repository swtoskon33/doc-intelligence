# Extraction backend comparison

Every extraction backend scored on the same golden set (12 hand-written synthetic documents: clean and noisy invoices with Swiss VAT, receipts, contracts). Regenerate with `python scripts/benchmark_extractors.py`.

These 12 documents are synthetic, written to cover specific cases. They separate the backends and exercise every branch of the pipeline, but are far too few to support a generalisation claim: read the numbers as a comparison under identical conditions, not as accuracy estimates. The LayoutLMv3 results come from FUNSD, a real annotated benchmark of 199 documents.

| Backend | Field accuracy | Valid docs | Flagged for review | Latency / doc (ms) |
|---------|----------------|------------|--------------------|--------------------|
| rule | 0.829 | 8/12 | 9/12 | 0.38 |
| hf | 0.561 | 6/12 | 10/12 | 1982.69 |

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

## LayoutLMv3, measured separately

LayoutLMv3 is not in the table above because it cannot share that evaluation: it is
fine-tuned on FUNSD with form-semantic labels (question / answer / header), while the
other backends extract invoice fields from the synthetic set. Same task family,
different dataset and different label space, so a single row would be misleading.

| Backend | Dataset | Labels | Precision | Recall | F1 |
|---------|---------|--------|-----------|--------|-----|
| rule | 12 synthetic docs | invoice fields | - | - | field acc. 0.829 |
| hf | 12 synthetic docs | invoice fields | - | - | field acc. 0.561 |
| LayoutLMv3 | FUNSD, 50 test docs | form entities | 0.791 | 0.836 | 0.813 |

The LayoutLMv3 figures are entity-level seqeval scores after three epochs of
fine-tuning (see docs/layoutlm_training.json). The layout ablation in
docs/layout_ablation.md shows the same checkpoint dropping to 0.014 F1 when the
bounding boxes are zeroed, which is the evidence that the spatial signal is doing
the work.
