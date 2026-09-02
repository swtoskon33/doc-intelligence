"""Document ingestion: turn a raw document into a RawDocument the pipeline can use.

In production this is where OCR (e.g. Azure Document Intelligence) and PDF parsing live.
Here we keep it offline and deterministic: text documents are read directly, and the
document type is inferred from simple keyword signals. The interface is what matters --
a real OCR backend swaps in behind ingest_document without changing downstream code.
"""
from __future__ import annotations

from doc_intelligence.types import DocumentType, RawDocument

_TYPE_SIGNALS = {
    DocumentType.INVOICE: ["invoice", "invoice number", "amount due", "bill to"],
    DocumentType.RECEIPT: ["receipt", "total paid", "change due", "cashier"],
    DocumentType.CONTRACT: ["agreement", "party", "terms and conditions", "hereby"],
}


def infer_type(text: str) -> DocumentType:
    """Infer the document type from keyword signals (a cheap classifier stand-in)."""
    low = text.lower()
    best, best_hits = DocumentType.UNKNOWN, 0
    for dtype, signals in _TYPE_SIGNALS.items():
        hits = sum(1 for s in signals if s in low)
        if hits > best_hits:
            best, best_hits = dtype, hits
    return best


def ingest_document(doc_id: str, text: str) -> RawDocument:
    """Produce a RawDocument from raw text, inferring its type."""
    if not text or not text.strip():
        raise ValueError("cannot ingest an empty document")
    return RawDocument(id=doc_id, doc_type=infer_type(text), text=text)
