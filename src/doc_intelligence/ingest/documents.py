"""Document ingestion: turn a raw document into a RawDocument the pipeline can use.

In production this is where OCR (e.g. Azure Document Intelligence) and PDF parsing live.
Here we keep it offline and deterministic: text documents are read directly, and the
document type is inferred from keyword signals in English, German and French, since a
Swiss back office receives all three. The interface is what matters --
a real OCR backend swaps in behind ingest_document without changing downstream code.
"""
from __future__ import annotations

from doc_intelligence.types import DocumentType, RawDocument

# Strong signals name the document type outright, usually in its heading. Weak signals
# (an amount label, a date) appear on several types, so they only break ties.
_STRONG_SIGNALS = {
    DocumentType.INVOICE: ["invoice", "rechnung", "facture", "fattura"],
    DocumentType.RECEIPT: ["receipt", "quittung", "kassenbon", "beleg"],
    DocumentType.CONTRACT: ["agreement", "contract", "vertrag", "vereinbarung"],
}

_WEAK_SIGNALS = {
    DocumentType.INVOICE: ["invoice number", "amount due", "bill to", "rechnungsnummer",
                           "zahlbar", "due date"],
    DocumentType.RECEIPT: ["total paid", "change due", "cashier", "merchant"],
    DocumentType.CONTRACT: ["terms and conditions", "hereby", "vertragsparteien",
                            "effective date"],
}


def infer_type(text: str) -> DocumentType:
    """Infer the document type from keyword signals.

    Strong signals (the word "invoice" or "Quittung" in the text) outweigh weak ones
    (an amount label like "Betrag", which appears on invoices and receipts alike), so a
    receipt that mentions an amount is not misread as an invoice.
    """
    low = text.lower()
    scores: dict[DocumentType, int] = {}
    for dtype, signals in _STRONG_SIGNALS.items():
        scores[dtype] = 10 * sum(1 for s in signals if s in low)
    for dtype, signals in _WEAK_SIGNALS.items():
        scores[dtype] = scores.get(dtype, 0) + sum(1 for s in signals if s in low)

    best = max(scores, key=lambda d: scores[d])
    return best if scores[best] > 0 else DocumentType.UNKNOWN


def ingest_document(doc_id: str, text: str) -> RawDocument:
    """Produce a RawDocument from raw text, inferring its type."""
    if not text or not text.strip():
        raise ValueError("cannot ingest an empty document")
    return RawDocument(id=doc_id, doc_type=infer_type(text), text=text)
