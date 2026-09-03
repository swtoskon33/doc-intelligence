"""Tests for the OCR document path."""
import base64

import pytest
from fastapi.testclient import TestClient

from doc_intelligence.ocr.backends import AzureDocumentIntelligenceOCR, LocalOCR, get_ocr
from doc_intelligence.serving.main import app


@pytest.mark.unit
def test_local_ocr_passes_text_through():
    assert LocalOCR().to_text("INVOICE 42") == "INVOICE 42"
    assert LocalOCR().to_text(b"INVOICE 42") == "INVOICE 42"


@pytest.mark.unit
def test_local_ocr_returns_empty_for_binary_image():
    assert LocalOCR().to_text(b"\x89PNG\r\n\x1a\n") == ""


@pytest.mark.unit
def test_azure_backend_unavailable_without_credentials():
    ocr = AzureDocumentIntelligenceOCR(endpoint=None, key=None)
    assert not ocr.available
    assert ocr.to_text(b"anything") == ""


@pytest.mark.unit
def test_local_is_the_default_backend():
    assert get_ocr().name == "local"


@pytest.mark.integration
def test_extract_accepts_base64_image_payload():
    client = TestClient(app)
    payload = base64.b64encode(b"INVOICE number INV-77. Total: CHF 10.00. Invoice date: 2026-02-02").decode()
    r = client.post("/extract", json={"document_id": "d2", "image_base64": payload})
    assert r.status_code == 200
    assert r.json()["fields"]["invoice_number"]["value"] == "INV-77"


@pytest.mark.integration
def test_extract_rejects_empty_request():
    r = TestClient(app).post("/extract", json={"document_id": "d3"})
    assert r.status_code == 422
