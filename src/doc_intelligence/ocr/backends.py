"""OCR backends: turn a scanned page image into text the pipeline can extract from.

Two backends behind one interface:
  local - no OCR engine; accepts text directly (or decodes an embedded text payload).
          The CI default, so tests run with no cloud calls.
  azure - Azure Document Intelligence (the production path for real scans). Requires
          AZURE_DI_ENDPOINT and AZURE_DI_KEY; reports itself unavailable without them.
"""
from __future__ import annotations

import base64
import os


class OCRBackend:
    """Base interface. `name` identifies the backend in metrics and reports."""

    name: str = "base"
    available: bool = True

    def to_text(self, payload: bytes | str) -> str:  # pragma: no cover
        raise NotImplementedError


class LocalOCR(OCRBackend):
    """Offline path: the payload already is (or decodes to) text."""

    name = "local"

    def to_text(self, payload: bytes | str) -> str:
        if isinstance(payload, str):
            return payload
        try:
            return payload.decode("utf-8")
        except UnicodeDecodeError:
            # a real image with no OCR engine available -- nothing to read
            return ""


class AzureDocumentIntelligenceOCR(OCRBackend):
    """Azure Document Intelligence: layout-aware OCR for scanned documents.

    Uses the prebuilt-read model and concatenates the recognised lines. The client is
    imported lazily so the dependency stays optional.
    """

    name = "azure"

    def __init__(self, endpoint: str | None = None, key: str | None = None) -> None:
        self.endpoint = endpoint or os.getenv("AZURE_DI_ENDPOINT")
        self.key = key or os.getenv("AZURE_DI_KEY")
        self.available = bool(self.endpoint and self.key)

    def to_text(self, payload: bytes | str) -> str:
        if not self.available:
            return ""
        from azure.ai.documentintelligence import DocumentIntelligenceClient
        from azure.core.credentials import AzureKeyCredential

        client = DocumentIntelligenceClient(self.endpoint, AzureKeyCredential(self.key))
        data = payload.encode() if isinstance(payload, str) else payload
        poller = client.begin_analyze_document("prebuilt-read", body=data)
        result = poller.result()
        lines = []
        for page in getattr(result, "pages", []):
            lines.extend(line.content for line in getattr(page, "lines", []))
        return "\n".join(lines)


def decode_image_payload(image_base64: str) -> bytes:
    """Decode a base64 image payload from the API into raw bytes."""
    return base64.b64decode(image_base64)


def get_ocr(backend: str | None = None) -> OCRBackend:
    """Select an OCR backend from OCR_BACKEND (default: local)."""
    name = (backend or os.getenv("OCR_BACKEND", "local")).lower()
    if name == "azure":
        return AzureDocumentIntelligenceOCR()
    return LocalOCR()
