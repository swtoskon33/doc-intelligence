"""FastAPI serving layer: submit a document, get validated structured fields back.

Endpoints:
  GET  /health     liveness probe (used by Kubernetes).
  GET  /metrics    Prometheus scrape endpoint.
  GET  /aliases    which backend each serving alias currently points at.
  POST /extract    OCR (if needed) -> extract -> validate -> typed fields + review flag.

Requests pick a serving alias (champion by default); promotion is an alias flip in the
registry rather than a redeploy.
"""
from __future__ import annotations

import time

from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from doc_intelligence.extraction.base import Extractor
from doc_intelligence.ingest.documents import ingest_document
from doc_intelligence.monitoring import metrics as m
from doc_intelligence.ocr.backends import decode_image_payload, get_ocr
from doc_intelligence.serving.registry import AliasRegistry
from doc_intelligence.validation.rules import validate_in_place


class ExtractRequest(BaseModel):
    """Submit either raw text or a base64-encoded scan (routed through OCR)."""

    document_id: str
    text: str | None = None
    image_base64: str | None = None
    alias: str = "champion"


class FieldOut(BaseModel):
    value: str | None
    confidence: float


class ExtractResponse(BaseModel):
    document_id: str
    doc_type: str
    fields: dict[str, FieldOut]
    served_by: str
    alias: str
    ocr_backend: str
    is_valid: bool
    needs_review: bool
    review_reasons: list[str]


def create_app(extractor: Extractor | None = None) -> FastAPI:
    app = FastAPI(title="doc-intelligence", version="0.2.0")
    registry = AliasRegistry()
    ocr = get_ocr()
    override = extractor  # tests can pin a single extractor

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics")
    def prometheus_metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/aliases")
    def aliases() -> dict[str, str]:
        return registry.as_dict()

    @app.post("/extract", response_model=ExtractResponse)
    def extract(req: ExtractRequest) -> ExtractResponse:
        try:
            ex = override or registry.get(req.alias)
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if req.image_base64:
            text = ocr.to_text(decode_image_payload(req.image_base64))
        else:
            text = req.text or ""
        if not text.strip():
            raise HTTPException(status_code=422, detail="no readable text in the request")

        started = time.perf_counter()
        doc = ingest_document(req.document_id, text)
        result = validate_in_place(ex.extract(doc))
        elapsed = time.perf_counter() - started

        m.record(
            backend=ex.name, alias=req.alias, doc_type=result.doc_type.value,
            seconds=elapsed, needs_review=result.needs_review,
            is_valid=result.is_valid, ocr_backend=ocr.name,
        )

        return ExtractResponse(
            document_id=result.document_id,
            doc_type=result.doc_type.value,
            fields={n: FieldOut(value=f.value, confidence=f.confidence)
                    for n, f in result.fields.items()},
            served_by=ex.name,
            alias=req.alias,
            ocr_backend=ocr.name,
            is_valid=result.is_valid,
            needs_review=result.needs_review,
            review_reasons=result.review_reasons,
        )

    return app
