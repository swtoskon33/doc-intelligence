"""Integration test: the full ingest -> extract -> serve path."""
import pytest
from fastapi.testclient import TestClient

from doc_intelligence.serving.main import app


@pytest.mark.integration
def test_extract_endpoint_returns_fields():
    client = TestClient(app)
    r = client.post("/extract", json={
        "document_id": "inv1",
        "text": "INVOICE number INV-42. Total: 1250.00. Due date: 2026-09-15",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["doc_type"] == "invoice"
    assert body["fields"]["invoice_number"]["value"] == "INV-42"


@pytest.mark.integration
def test_health():
    assert TestClient(app).get("/health").status_code == 200
