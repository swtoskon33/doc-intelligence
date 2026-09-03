"""Prometheus metrics for the extraction service.

Exposes the signals you actually page on: request latency, which backend served, how
often documents land in human review, and how often business validation fails. Scraped
from /metrics.
"""
from __future__ import annotations

from prometheus_client import Counter, Histogram

EXTRACTIONS = Counter(
    "doc_extractions_total",
    "Extraction requests served",
    ["backend", "alias", "doc_type"],
)

LATENCY = Histogram(
    "doc_extraction_latency_seconds",
    "End-to-end extraction latency",
    ["backend"],
    buckets=(0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0),
)

NEEDS_REVIEW = Counter(
    "doc_needs_review_total",
    "Documents flagged for human review",
    ["backend"],
)

VALIDATION_FAILURES = Counter(
    "doc_validation_failures_total",
    "Documents failing at least one business rule",
    ["backend"],
)

OCR_REQUESTS = Counter(
    "doc_ocr_requests_total",
    "Documents routed through an OCR backend",
    ["ocr_backend"],
)


def record(backend: str, alias: str, doc_type: str, seconds: float,
           needs_review: bool, is_valid: bool, ocr_backend: str) -> None:
    """Record one served request across all metrics."""
    EXTRACTIONS.labels(backend=backend, alias=alias, doc_type=doc_type).inc()
    LATENCY.labels(backend=backend).observe(seconds)
    OCR_REQUESTS.labels(ocr_backend=ocr_backend).inc()
    if needs_review:
        NEEDS_REVIEW.labels(backend=backend).inc()
    if not is_valid:
        VALIDATION_FAILURES.labels(backend=backend).inc()
