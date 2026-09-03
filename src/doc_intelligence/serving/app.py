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

# a 512Mi container should not be asked to decode an arbitrarily large payload
MAX_IMAGE_BYTES = 8 * 1024 * 1024


class ExtractRequest(BaseModel):
    """Submit either raw text or a base64-encoded scan (routed through OCR)."""

    document_id: str
    text: str | None = None
    image_base64: str | None = None
    alias: str = "champion"


class LayoutRequest(BaseModel):
    """Layout-aware extraction: words with their bounding boxes, already normalised."""

    document_id: str
    words: list[str]
    boxes: list[list[int]]
    image_base64: str | None = None


class EntityOut(BaseModel):
    label: str
    text: str
    confidence: float
    box: list[int]


class LayoutResponse(BaseModel):
    document_id: str
    entities: list[EntityOut]
    served_by: str
    needs_review: bool


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

    @app.post("/extract/layout", response_model=LayoutResponse)
    def extract_layout(req: LayoutRequest) -> LayoutResponse:
        """Run the layout-aware backend over words and their boxes."""
        from PIL import Image

        from doc_intelligence.extraction.layoutlm import LayoutLMv3Extractor

        backend = LayoutLMv3Extractor()
        if not backend.available:
            raise HTTPException(
                status_code=503,
                detail="LayoutLMv3 checkpoint not found; train it with scripts/train_layoutlm.py",
            )
        if len(req.words) != len(req.boxes):
            raise HTTPException(status_code=422, detail="words and boxes must be the same length")

        if req.image_base64:
            import io

            image = Image.open(io.BytesIO(decode_image_payload(req.image_base64))).convert("RGB")
        else:
            # LayoutLMv3 needs an image tensor; a blank page keeps the text+layout signal
            image = Image.new("RGB", (1000, 1000), "white")

        started = time.perf_counter()
        entities = backend.predict_entities(image, req.words, req.boxes)
        elapsed = time.perf_counter() - started

        low_confidence = any(e.confidence < 0.7 for e in entities)
        m.record(
            backend=backend.name, alias="layout", doc_type="form", seconds=elapsed,
            needs_review=low_confidence, is_valid=True, ocr_backend=ocr.name,
        )
        return LayoutResponse(
            document_id=req.document_id,
            entities=[EntityOut(label=e.label, text=e.text, confidence=e.confidence, box=e.box)
                      for e in entities],
            served_by=backend.name,
            needs_review=low_confidence,
        )

    @app.post("/extract", response_model=ExtractResponse)
    def extract(req: ExtractRequest) -> ExtractResponse:
        try:
            ex = override or registry.get(req.alias)
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if req.image_base64:
            if len(req.image_base64) > MAX_IMAGE_BYTES:
                raise HTTPException(status_code=413, detail="image payload too large")
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
