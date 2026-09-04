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

```
                    PDF / image
                         |
                        OCR
                         |
              text + bounding boxes
                         |
               LayoutLMv3 processor
                         |
              token classification
                         |
                    BIO decoding
                         |
              confidence estimation
                         |
                schema validation
                         |
                +--------+--------+
                |                 |
          valid, confident    invalid or
                |             low confidence
                |                 |
             output          human review
```


### Extraction backends

```
                       document
                          |
                         OCR
                          |
                 EXTRACTION_BACKEND
                          |
      +---------+---------+---------+
      |         |         |         |
    rules     HF QA      LLM   LayoutLMv3
   (regex)  (per field) (JSON)  (+ boxes)
      |         |         |         |
      +---------+---------+---------+
                          |
              validation + confidence
                          |
                +---------+---------+
                |                   |
          auto-accept          human review
```


### LayoutLMv3 data flow

```
   page image        words          boxes
        |              |              |
        |              |     normalise to 0-1000
        |              |              |
        +--------------+--------------+
                       |
            LayoutLMv3 processor (apply_ocr=False)
                       |
              LayoutLMv3 encoder
                       |
               token classification
                       |
         softmax -> per-word label + score
                       |
                  BIO decoding
                       |
    entities: label, text, confidence, span box
```


### What this does and does not show

The fine-tuning runs on FUNSD, whose labels are form semantics (question / answer /
header), not the invoice fields the rule and LLM backends extract. So the numbers
demonstrate that the pipeline -- dataset, preprocessing, training loop,
checkpointing, decoding, serving -- works end to end on a real annotated benchmark.
They are not an invoice-extraction score, and the two evaluations in this repo are
deliberately kept separate for that reason.

Retraining on invoice-specific BIO labels is a dataset change, not a code change:
point the loader at annotated invoices, update the label list, rerun the same
script. The obstacle is annotated data, not the implementation.

### Training pipeline

```
        FUNSD: 199 annotated documents
                       |
        document-level split 119 / 30 / 50
                       |
     encode: processor + word_ids label alignment
                       |
        PyTorch loop (AdamW, weight decay)
                       |
        per-epoch validation (seqeval P/R/F1)
                       |
             +---------+---------+
             |                   |
        improved:            no gain twice:
      save checkpoint          early stop
             |                   |
             +---------+---------+
                       |
               test evaluation
                       |
          +------------+------------+
          |                         |
   MLflow (params,        docs/layoutlm_training.json
    metrics, history)
```


### Serving and model lifecycle

```
                      client
                        |
                     FastAPI
                        |
        +---------------+---------------+
        |                               |
   POST /extract                POST /extract/layout
        |                               |
    alias router                 LayoutLMv3 backend
   (champion /                          |
    challenger)                         |
        |                               |
        +---------------+---------------+
                        |
          validation + review decision
                        |
     response: fields, confidence, review reasons
                        |
        /metrics -> Prometheus -> Grafana
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

Three backends behind one interface, scored on the same golden set of 18 synthetic
documents (hand-written to cover noisy OCR text, a German invoice, a reduced Swiss VAT rate,
Swiss and German number and date formats, a UK IBAN, an impossible date, a
malformed IBAN and missing fields). Small and synthetic by design: it separates
the backends but is not a generalisation claim. The LayoutLMv3 results come from
FUNSD, a real annotated benchmark of 199 documents
(regenerate: `python scripts/benchmark_extractors.py`, full table in
docs/model_comparison.md):

| Backend | Field accuracy | Flagged for review | Latency / doc (ms) |
|---------|----------------|--------------------|---------------|
| rule (regex, schema-driven) | 0.902 | 15/18 | 0.3 |
| hf (transformer QA) | 0.541 | 16/18 | 411 |
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
labelled golden set and scores it by leave-one-out cross-validation: every document
is predicted by a model that never saw it. With 15 documents, training accuracy
would only measure memorisation.

| Classifier | Accuracy (15 documents) |
|------------|--------------------------|
| keyword heuristic | 0.933 |
| trained model, leave-one-out | 0.667 |
| trained model on its own training data | 0.800 (reference only) |

The heuristic wins, and that is the useful result. At this sample size TF-IDF has
too little to learn from, while the keywords it competes against (INVOICE, RECEIPT,
AGREEMENT) are a strong signal in this domain. The trained model would need an order
of magnitude more documents before it earned its place; until then the cheap
classifier is the right default. Full numbers in docs/classifier_report.json.

## The rule backend on real scans

Every benchmark above uses synthetic documents written alongside the extraction patterns,
which cannot show the failure that matters. Run over 50 genuinely scanned FUNSD pages
(`python scripts/probe_real_documents.py`, details in docs/real_document_probe.md):

| Measure | Result |
|---------|--------|
| Documents processed | 50 |
| Crashes | 0 |
| Document type recognised | 15/50 (the rest are memos and forms) |
| Field values extracted | 9 |

Nothing crashes and almost nothing is extracted. Most of the unrecognised pages are a
correct abstention rather than a miss: they are internal memos and fax covers, sales
reports and requisition forms from a corporate archive, none of which is an invoice, a
receipt or a contract. Placing them in one of those three would be inventing a type.

What does matter is the pages it *does* read as invoices, which give up barely anything. The patterns expect `Total: CHF 1081.00` on one line; a scan gives a label in
one cell, its value in another, and characters the OCR guessed at. A regex matches a
string, it has no notion that the number beside the word Total is the total.

That is the boundary the LayoutLMv3 backend exists to cross, and the reason both paths
are here: rules for clean structured input where the format is stable and the cost is
microseconds, a layout-aware model for anything off a scanner.

## How it maps to IDP

Classification, splitting, extraction, validation and review are the stages of a
document back office in regulated industries. Each is a testable module here, running
offline and reproducibly.

## API

```
uvicorn doc_intelligence.serving.main:app --port 8000

curl -X POST localhost:8000/extract \\
  -H "Content-Type: application/json" \\
  -d '{"document_id": "inv1", "text": "INVOICE number INV-42. Total: CHF 1081.00. MWST rate 8.1%. Invoice date: 15.09.2026"}'
```

Returns the extracted fields with confidence, `is_valid`, `needs_review` and
`review_reasons`. `POST /extract/layout` takes words with bounding boxes and runs the
LayoutLMv3 backend instead.

## Quickstart

```
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"                    # core + tests
pip install -e ".[dev,ml,tracking]"        # + PyTorch, LayoutLMv3, MLflow
python -m pytest
uvicorn doc_intelligence.serving.main:app --port 8000
```

## Running the evaluations

```
python scripts/build_golden.py          # regenerate the golden set
python scripts/run_eval.py              # per-field metrics -> docs/eval_report.md
python scripts/benchmark_extractors.py  # backend comparison -> docs/model_comparison.md
python scripts/train_classifier.py      # classifier, leave-one-out
python scripts/train_layoutlm.py        # LayoutLMv3 fine-tuning (needs [ml])
python scripts/ablate_layout.py         # layout ablation
```

## Deploy

```
docker build -t doc-intelligence:latest .
kubectl apply -f k8s/                   # deployment, service, HPA

# only needed if the challenger alias points at the LLM backend
kubectl create secret generic doc-intelligence-secrets \\
  --from-literal=openai-api-key=sk-...
```

## Backends

| Component  | Offline (default)      | Alternatives                        |
|------------|------------------------|-------------------------------------|
| Extraction | RuleExtractor (schema) | `=hf`, `=llm` or `=layoutlmv3`      |
| OCR        | direct text            | Azure Document Intelligence         |

Selected with `EXTRACTION_BACKEND` and `OCR_BACKEND`.

## Layout

```
src/doc_intelligence/
  types.py         domain types (RawDocument, Field, ExtractionResult, ValidationError)
  ingest/          document type inference (English, German, French signals)
  splitting/       Docsplit-style boundary detection
  schemas/         YAML field schemas + registry (single source of truth)
  extraction/      base interface + rule / llm / hf / layoutlmv3 backends
  layout/          FUNSD dataset, LayoutLMv3 preprocessing and training
  validation/      business rules (required fields, dates, IBAN, MWST)
  eval/            per-field precision / recall / F1
  pipeline/        end-to-end orchestrator with accept/review decisions
  memory/          vendor memory for recalling stable fields
  models/          trained baseline document classifier
  ocr/             OCR backends (local, Azure Document Intelligence)
  monitoring/      Prometheus metrics
  tracking/        MLflow experiment tracking
  serving/         FastAPI app, alias registry, ASGI entrypoint

tests/             unit + integration (76 tests)
k8s/               deployment, service, HPA
monitoring/        grafana_dashboard.json
scripts/           build_golden.py, run_eval.py, benchmark_extractors.py,
                   train_classifier.py, train_layoutlm.py, ablate_layout.py, serve.py
docs/              eval_report.md, model_comparison.md, layout_ablation.md,
                   layoutlm_training.json, classifier_report.json
```

## Stack

Python 3.11, PyTorch, Hugging Face Transformers (LayoutLMv3), scikit-learn, MLflow,
FastAPI, Pydantic, PyYAML, Prometheus, pytest, ruff, Docker, Kubernetes, GitHub Actions.

## License

MIT
