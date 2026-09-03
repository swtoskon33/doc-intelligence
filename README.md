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


## Architecture

Every diagram below reflects code in this repository.

### End-to-end document pipeline

```mermaid
flowchart TD
    A[PDF / image or text] --> B[OCR backend<br/>local or Azure DI]
    B --> C[words + bounding boxes]
    C --> D[normalise boxes to 0-1000]
    D --> E[LayoutLMv3 processor]
    E --> F[token classification]
    F --> G[BIO decoding to entities]
    G --> H[confidence per entity]
    H --> I{valid and confident?}
    I -->|yes| J[structured output]
    I -->|no| K[human review<br/>with reasons]
```

### Extraction backends

LayoutLMv3 is an additional backend; the existing three are untouched.

```mermaid
flowchart TD
    D[Document] --> O[OCR]
    O --> R{EXTRACTION_BACKEND}
    R -->|rule| A[regex over YAML schema]
    R -->|hf| B[transformer QA per field]
    R -->|llm| C[LLM structured JSON]
    R -->|layoutlmv3| E[LayoutLMv3 token classification]
    A --> V[validation + confidence]
    B --> V
    C --> V
    E --> V
    V --> W{auto-accept or review}
```

### LayoutLMv3 data flow

```mermaid
flowchart TD
    I[page image] --> P[LayoutLMv3 processor<br/>apply_ocr=False]
    W[words] --> P
    B[boxes 0-1000] --> P
    P --> E[LayoutLMv3 encoder]
    E --> T[token logits]
    T --> S[softmax to per-word label + score]
    S --> D[BIO decoding]
    D --> N[entities: label, text, confidence, span box]
```

### Training pipeline

```mermaid
flowchart TD
    F[FUNSD: 199 annotated documents] --> S[document-level split<br/>119 / 30 / 50]
    S --> E[encode: processor + word_ids label alignment]
    E --> T[PyTorch loop: AdamW, weight decay]
    T --> V[per-epoch validation<br/>seqeval P/R/F1]
    V --> C{improved?}
    C -->|yes| K[save checkpoint]
    C -->|no, twice| X[early stop]
    K --> R[test evaluation]
    X --> R
    R --> M[MLflow: params, metrics]
    R --> J[docs/layoutlm_training.json]
```

### Serving and model lifecycle

Aliases select which backend serves traffic; promotion is an alias flip, not a redeploy.

```mermaid
flowchart TD
    C[client] --> A[FastAPI]
    A -->|POST /extract| AL{alias}
    AL -->|champion| B1[backend]
    AL -->|challenger| B2[backend]
    A -->|POST /extract/layout| L[LayoutLMv3 backend]
    B1 --> V[validation + review decision]
    B2 --> V
    L --> V
    V --> R[response: fields, confidence, review reasons]
    A --> P[/metrics: latency, review rate, validation failures/]
    P --> G[Prometheus + Grafana]
```

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

| Backend | Field accuracy | Flagged for review | Latency / doc (ms) |
|---------|----------------|--------------------|---------------|
| rule (regex, schema-driven) | 0.829 | 9/12 | 0.4 |
| hf (transformer QA) | 0.561 | 10/12 | 1983 |
| llm (OpenAI / Azure) | not run (needs an API key) | - | - |

On structured documents the regex backend beats a general-purpose transformer on both
accuracy and latency. The SQuAD-trained QA model is tuned for prose, not forms: it has no
layout signal, and it cannot abstain — when a field is absent it returns a confident span
from elsewhere in the document. That is the argument for keeping rules where the format is
stable, and for a layout-aware model (LayoutLM family) or an LLM that can return null
where it is not.

## Serving, aliases and monitoring

Requests are served through an alias, so promotion is a config flip rather than a
redeploy:

```
POST /extract  {"document_id": "d1", "text": "...", "alias": "champion"}
GET  /aliases  -> {"champion": "rule", "challenger": "llm"}
GET  /metrics  -> Prometheus scrape
GET  /health   -> liveness probe
```

Aliases come from `CHAMPION_BACKEND` / `CHALLENGER_BACKEND`; OCR from `OCR_BACKEND`.

Metrics exposed: extraction count by backend / alias / document type, latency
histogram, human-review rate, validation-failure rate, and OCR backend usage. A ready
dashboard is in `monitoring/grafana_dashboard.json`.

## Scanned documents

`/extract` accepts `image_base64` alongside `text`. The local OCR backend (default)
treats the payload as text so CI needs no cloud calls; `OCR_BACKEND=azure` routes scans
through Azure Document Intelligence (prebuilt-read) when credentials are present.

## Experiment tracking

`scripts/benchmark_extractors.py` logs every backend evaluation to MLflow: backend and
golden-set size as params, field accuracy, valid documents, review rate and latency as
metrics. Tracking no-ops when MLflow is absent, so the benchmark still runs offline.

## End-to-end pipeline

`DocumentPipeline` runs the whole flow a back office needs and ends with a decision,
not just fields:

```
batch -> split -> classify -> select backend -> extract -> validate -> decide
                                                                        |
                                            auto_accept  <-------------- +
                                            needs_review (with reasons)
```

Backend selection is policy: the cheap deterministic backend runs first, and a fallback
is only used when the first result would land in review anyway.

### Vendor memory

Invoices from the same sender repeat their IBAN, VAT rate and currency. When a document
omits one, the pipeline recalls it from the last accepted document of that vendor and
marks it at lower confidence, since it was remembered rather than read. Accepted
documents feed back into the store. This is small-scale retrieval that reduces missing
fields; it is not a language model.

### Trained classifier

`scripts/train_classifier.py` fits a TF-IDF + logistic-regression classifier on the
labelled golden set and reports it against the keyword heuristic:

| Classifier | Accuracy |
|------------|----------|
| trained (TF-IDF + logistic regression) | 1.000 |
| keyword heuristic | 0.917 |

Small numbers on a small set, but the comparison is measured rather than assumed.

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

## Running the evaluations

```
python scripts/build_golden.py          # regenerate the golden set
python scripts/run_eval.py              # per-field metrics -> docs/eval_report.md
python scripts/benchmark_extractors.py  # backend comparison -> docs/model_comparison.md
python scripts/train_classifier.py      # train the baseline classifier + compare
python -m pytest                        # 42 tests (1 skipped without transformers)
```

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
  ocr/             OCR backends (local, Azure Document Intelligence)
  pipeline/        end-to-end orchestrator with accept/review decisions
  memory/          vendor memory for recalling stable fields
  models/          trained baseline document classifier
  monitoring/      Prometheus metrics
  tracking/        MLflow experiment tracking
  serving/         FastAPI app, alias registry, ASGI entrypoint

tests/             unit + integration (42 tests)
k8s/               deployment, service, HPA
monitoring/        grafana_dashboard.json
scripts/           build_golden.py, run_eval.py, benchmark_extractors.py,
                   train_classifier.py, serve.py
docs/              eval_report.md, model_comparison.md
```

## Stack

Python 3.11, PyTorch, Hugging Face Transformers (LayoutLMv3), scikit-learn, MLflow,
FastAPI, Pydantic, PyYAML, Prometheus, pytest, ruff, Docker, Kubernetes, GitHub Actions.

## License

MIT
