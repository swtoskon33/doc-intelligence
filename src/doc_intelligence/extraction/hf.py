"""Hugging Face extractor: transformer-based field extraction from document text.

A learned alternative to the regex rules. Uses a question-answering model: for each
field in the schema, the field name becomes a question ("What is the invoice number?")
answered against the document text, and the model's answer score becomes the field
confidence.

Weight is the trade-off: transformers and the model are a large download, so the import
is lazy and CI skips this backend when the dependency is absent. That keeps the offline
rule path -- and the whole test suite -- runnable with no downloads.
"""
from __future__ import annotations

import os

from doc_intelligence.extraction.base import Extractor
from doc_intelligence.schemas.registry import load_schema
from doc_intelligence.types import ExtractionResult, Field, RawDocument

DEFAULT_MODEL = "distilbert-base-cased-distilled-squad"

_QUESTIONS = {
    "invoice_number": "What is the invoice number?",
    "vendor": "Who is the vendor?",
    "iban": "What is the IBAN?",
    "total_amount": "What is the total amount?",
    "mwst_amount": "What is the VAT amount?",
    "mwst_rate": "What is the VAT rate?",
    "currency": "What is the currency?",
    "invoice_date": "What is the invoice date?",
    "due_date": "What is the due date?",
    "merchant": "What is the merchant name?",
    "date": "What is the date?",
    "party": "Who are the parties?",
    "effective_date": "What is the effective date?",
    "term": "What is the term?",
}


def transformers_available() -> bool:
    """True if transformers is installed (so callers can skip this backend)."""
    try:
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


class HFExtractor(Extractor):
    """Extractive question answering over the document text, one question per field."""

    name = "hf"

    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.getenv("HF_MODEL", DEFAULT_MODEL)
        self._pipe = None
        self.available = transformers_available()

    def _pipeline(self):
        if self._pipe is None:
            from transformers import pipeline  # lazy: heavy import

            self._pipe = pipeline("question-answering", model=self.model)
        return self._pipe

    def extract(self, doc: RawDocument) -> ExtractionResult:
        specs = load_schema(doc.doc_type)
        names = [s.name for s in specs]

        if not self.available:
            fields = {n: Field(n, None, 0.0) for n in names}
            return ExtractionResult(doc.id, doc.doc_type, fields)

        qa = self._pipeline()
        fields = {}
        for n in names:
            question = _QUESTIONS.get(n, f"What is the {n.replace('_', ' ')}?")
            try:
                out = qa(question=question, context=doc.text)
                answer = (out.get("answer") or "").strip()
                score = float(out.get("score", 0.0))
            except (RuntimeError, ValueError, KeyError):  # one bad question should not sink the doc
                answer, score = "", 0.0
            fields[n] = Field(n, answer or None, confidence=round(score, 3) if answer else 0.0)
        return ExtractionResult(doc.id, doc.doc_type, fields)
