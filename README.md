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

Three backends behind one interface, scored on the same golden set of 15 synthetic
documents (hand-written to cover noisy OCR text, a German invoice, a reduced Swiss VAT rate,
a Swiss-format date, an impossible date, a malformed IBAN and missing fields). Small and synthetic by design: it separates
the backends but is not a generalisation claim. The LayoutLMv3 results come from
FUNSD, a real annotated benchmark of 199 documents
(regenerate: `python scripts/benchmark_extractors.py`, full table in
docs/model_comparison.md):

| Backend | Field accuracy | Flagged for review | Latency / doc (ms) |
|---------|----------------|--------------------|---------------|
| rule (regex, schema-driven) | 0.833 | 12/15 | 0.3 |
| hf (transformer QA) | 0.542 | 13/15 | 560 |
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


