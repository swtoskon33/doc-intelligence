"""FastAPI serving layer: submit a document, get structured fields back.

Endpoints:
  GET  /health            liveness probe (used by Kubernetes).
  POST /extract           ingest -> extract -> return typed fields + review flag.

The extractor is built once at startup (from EXTRACTION_BACKEND) and reused, so the
service is stateless and horizontally scalable behind a load balancer.
"""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from doc_intelligence.extraction.extractor import Extractor, get_extractor
from doc_intelligence.ingest.documents import ingest_document
from doc_intelligence.validation.rules import validate_in_place


class ExtractRequest(BaseModel):
    document_id: str
    text: str


class FieldOut(BaseModel):
    value: str | None
    confidence: float


class ExtractResponse(BaseModel):
    document_id: str
    doc_type: str
    fields: dict[str, FieldOut]
    is_valid: bool
    needs_review: bool
    review_reasons: list[str]


def create_app(extractor: Extractor | None = None) -> FastAPI:
    app = FastAPI(title="doc-intelligence", version="0.1.0")
    ex = extractor or get_extractor()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/extract", response_model=ExtractResponse)
    def extract(req: ExtractRequest) -> ExtractResponse:
        doc = ingest_document(req.document_id, req.text)
        result = validate_in_place(ex.extract(doc))
        return ExtractResponse(
            document_id=result.document_id,
            doc_type=result.doc_type.value,
            fields={
                name: FieldOut(value=f.value, confidence=f.confidence)
                for name, f in result.fields.items()
            },
            is_valid=result.is_valid,
            needs_review=result.needs_review,
            review_reasons=result.review_reasons,
        )

    return app
