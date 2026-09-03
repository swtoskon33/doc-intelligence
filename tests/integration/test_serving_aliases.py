"""Tests for serving aliases and Prometheus metrics."""
import pytest
from fastapi.testclient import TestClient

from doc_intelligence.serving.main import app
from doc_intelligence.serving.registry import AliasRegistry


@pytest.mark.unit
def test_default_aliases():
    r = AliasRegistry()
    assert r.backend_for("champion") == "rule"
    assert r.get("champion").name == "rule"


@pytest.mark.unit
def test_promote_repoints_alias():
    r = AliasRegistry()
    r.promote("champion", "hf")
    assert r.backend_for("champion") == "hf"


@pytest.mark.unit
def test_unknown_alias_raises():
    with pytest.raises(KeyError):
        AliasRegistry().backend_for("nope")


@pytest.mark.integration
def test_extract_reports_alias_and_backend():
    r = TestClient(app).post("/extract", json={
        "document_id": "d1",
        "text": "INVOICE number INV-9. Total: CHF 50.00. Invoice date: 2026-01-01",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["alias"] == "champion"
    assert body["served_by"] == "rule"


@pytest.mark.integration
def test_unknown_alias_returns_400():
    r = TestClient(app).post("/extract", json={
        "document_id": "d1", "text": "INVOICE number INV-9", "alias": "nope",
    })
    assert r.status_code == 400


@pytest.mark.integration
def test_aliases_endpoint():
    body = TestClient(app).get("/aliases").json()
    assert "champion" in body and "challenger" in body


@pytest.mark.integration
def test_metrics_endpoint_exposes_counters():
    client = TestClient(app)
    client.post("/extract", json={"document_id": "d1", "text": "INVOICE number INV-9"})
    text = client.get("/metrics").text
    assert "doc_extractions_total" in text
    assert "doc_extraction_latency_seconds" in text
