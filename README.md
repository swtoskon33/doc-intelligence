# doc-intelligence

[![CI](https://github.com/swtoskon33/doc-intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/swtoskon33/doc-intelligence/actions/workflows/ci.yml)

Intelligent Document Processing service. Classifies documents, splits multi-document
batches, extracts structured fields, validates them against business rules, and flags
low-confidence or invalid results for human review. Built offline-first: the rule-based
backend runs in CI with no API keys; an LLM backend (OpenAI / Azure OpenAI) is the
production path behind the same interface.

Batch -> Split -> Classify -> Extract -> Validate -> Review? -> Output
(pages) Docsplit doc type schema-driven business rules HITL structured
+ confidence IBAN, MWST + is_valid


## Features

- **Splitting** (Docsplit-style): break a multi-document page batch into individual
  documents with page ranges and inferred type.
- **Classification**: infer document type (invoice / receipt / contract) from the text.
- **Schema-driven extraction**: field definitions live in YAML (`schemas/`), consumed by
  both the rule and LLM backends. Swiss-relevant invoice fields: `iban`, `mwst_amount`,
  `mwst_rate`, `currency`, alongside `invoice_number`, `total_amount`, dates.
- **Validation**: required fields per type, date formats, IBAN mod-97 checksum, and Swiss
  VAT (MWST) consistency. Produces `validation_errors` and an `is_valid` flag.
- **Human-in-the-loop**: `needs_review` triggers on low confidence OR any validation
  failure, with `review_reasons` naming exactly what caused it.
- **Evaluation**: per-field precision / recall / F1 plus document-level metrics, written
  to a committed report (`docs/eval_report.md`).
- **Serving & deploy**: FastAPI (`POST /extract`, `GET /health`), Docker image, and
  Kubernetes manifests (deployment, service, HPA) with liveness/readiness probes.

## Extraction backends compared

Three backends behind one interface, scored on the same golden set of 12 documents
(regenerate: `python scripts/benchmark_extractors.py`, full table in
docs/model_comparison.md):

| Backend | Field accuracy | Flagged for review | Latency / doc |
|---------|----------------|--------------------|---------------|
| rule (regex, schema-driven) | 0.829 | 9/12 | 0.4 ms |
| hf (transformer QA) | 0.561 | 10/12 | 500 ms |
| llm (OpenAI / Azure) | needs an API key | - | - |

On structured documents the regex backend beats a general-purpose transformer on both
accuracy and latency. The SQuAD-trained QA model is tuned for prose, not forms: it has no
layout signal, and it cannot abstain — when a field is absent it returns a confident span
from elsewhere in the document. That is the argument for keeping rules where the format is
stable, and for a layout-aware model (LayoutLM family) or an LLM that can return null
where it is not.

## How it maps to IDP

Classification, splitting, extraction, validation, and review are the core stages of a
document-processing back office in regulated industries (accounting, insurance, banking).
This repo implements each stage as a testable module, offline and reproducible.

## API

```bash
uvicorn doc_intelligence.serving.main:app --port 8000

curl -X POST localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{"document_id": "inv1", "text": "INVOICE number INV-42. Total: CHF 1081.00. MWST rate 8.1%. MWST amount 81.00. IBAN: CH9300762011623852957. Invoice date: 2026-09-15"}'
```

Returns the extracted fields with confidence, `is_valid`, `needs_review`, and
`review_reasons`.

## Quickstart

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest
python scripts/run_eval.py        # regenerate docs/eval_report.md
uvicorn doc_intelligence.serving.main:app --port 8000
```

## Deploy

```bash
docker build -t doc-intelligence:latest .
kubectl apply -f k8s/
```

## Backends

| Component  | Offline (default)      | Alternatives                                    |
|------------|------------------------|-------------------------------------------------|
| Extraction | RuleExtractor (schema) | `EXTRACTION_BACKEND=hf` or `=llm`               |
| OCR        | direct text            | Azure Document Intelligence                     |

## Layout

```
src/doc_intelligence/
  types.py         domain types (RawDocument, Field, ExtractionResult, ValidationError)
  ingest/          document type inference
  splitting/       Docsplit-style boundary detection
  schemas/         YAML field schemas + registry (single source of truth)
  extraction/      base interface + rule / llm / hf backends
  validation/      business rules (required fields, dates, IBAN, MWST)
  eval/            per-field precision / recall / F1
  serving/         FastAPI app + ASGI entrypoint

tests/             unit + integration (25 tests)
k8s/               deployment, service, HPA
scripts/           build_golden.py, run_eval.py, benchmark_extractors.py, serve.py
docs/              eval_report.md, model_comparison.md
```

## Stack

Python 3.11, FastAPI, Pydantic, PyYAML, pytest, ruff, Docker, Kubernetes, GitHub Actions.

## License

MIT
