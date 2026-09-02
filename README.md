# doc-intelligence

Intelligent Document Processing service. Extracts structured fields from unstructured
documents (invoices, receipts, contracts) and returns them with confidence scores.
Includes evaluation, a FastAPI service, and Kubernetes manifests.

The extraction backend is pluggable via `EXTRACTION_BACKEND`: `rule` (deterministic,
offline, CI default) or `llm` (OpenAI / Azure OpenAI, production). OCR plugs into the
ingest layer (e.g. Azure Document Intelligence).

```
Document  ->  Ingest  ->  Extract  ->  Validate  ->  Review?  ->  Structured output
(text/OCR)    classify    fields+conf   rules        HITL         validated fields
```


## Features

- Document type inference (invoice / receipt / contract) from text signals.
- Schema-based field extraction per document type, each field with a confidence score.
- Low-confidence fields flag the document for review (`needs_review`).
- Per-field precision / recall / F1 evaluation against ground truth.
- FastAPI service: `POST /extract`, `GET /health`.
- Docker image and Kubernetes manifests (deployment, service, HPA) with liveness and
  readiness probes on `/health`.

## API

```bash
uvicorn doc_intelligence.serving.main:app --port 8000

curl -X POST localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{"document_id": "inv1", "text": "INVOICE number INV-42. Total: 1250.00. Due date: 2026-09-15"}'
```

```json
{
  "document_id": "inv1",
  "doc_type": "invoice",
  "fields": {
    "invoice_number": {"value": "INV-42", "confidence": 0.8},
    "total_amount": {"value": "1250.00", "confidence": 0.8},
    "due_date": {"value": "2026-09-15", "confidence": 0.8},
    "vendor": {"value": null, "confidence": 0.0}
  },
  "needs_review": true
}
```

## Quickstart

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest
uvicorn doc_intelligence.serving.main:app --port 8000
```

## Deploy

```bash
docker build -t doc-intelligence:latest .
kubectl apply -f k8s/
```

## Backends

| Component  | Offline (default)      | Production                          |
|------------|------------------------|-------------------------------------|
| Extraction | RuleExtractor (regex)  | LLM (`EXTRACTION_BACKEND=llm`)      |
| OCR        | direct text            | Azure Document Intelligence         |

## Layout

src/doc_intelligence/
types.py domain types (RawDocument, Field, ExtractionResult)
ingest/ document type inference
extraction/ pluggable field extractor with confidence scoring
eval/ per-field precision / recall / F1
serving/ FastAPI app + ASGI entrypoint
tests/ unit + integration (10 tests)
k8s/ deployment, service, HPA
.github/workflows/ lint + tests + coverage + docker build & smoke test


## Stack

Python 3.11, FastAPI, Pydantic, pytest, ruff, Docker, Kubernetes, GitHub Actions.

## License

MIT
