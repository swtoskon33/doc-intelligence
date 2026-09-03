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
        """Load tokenizer + QA model directly.

        The `question-answering` pipeline was removed in recent transformers releases,
        so we run the model ourselves: encode (question, context), take the highest
        scoring start/end span, and decode it.
        """
        if self._pipe is None:
            import torch
            from transformers import AutoModelForQuestionAnswering, AutoTokenizer

            tok = AutoTokenizer.from_pretrained(self.model)
            model = AutoModelForQuestionAnswering.from_pretrained(self.model)
            model.eval()

            def answer(question: str, context: str) -> tuple[str, float]:
                enc = tok(question, context, return_tensors="pt", truncation=True, max_length=512)
                with torch.no_grad():
                    out = model(**enc)
                start_probs = out.start_logits.softmax(-1)[0]
                end_probs = out.end_logits.softmax(-1)[0]
                start = int(start_probs.argmax())
                end = int(end_probs.argmax())
                if end < start:
                    return "", 0.0
                span = enc["input_ids"][0][start:end + 1]
                text = tok.decode(span, skip_special_tokens=True).strip()
                # tokenizer artifacts: 'INV - 1001' -> 'INV-1001', '1081. 00' -> '1081.00'
                text = text.replace(' - ', '-').replace('. ', '.').replace(' , ', ',')
                score = float(start_probs[start] * end_probs[end])
                return text, score

            self._pipe = answer
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
                answer, score = qa(question, doc.text)
            except (RuntimeError, ValueError, KeyError):  # one bad question should not sink the doc
                answer, score = "", 0.0
            fields[n] = Field(n, answer or None, confidence=round(score, 3) if answer else 0.0)
        return ExtractionResult(doc.id, doc.doc_type, fields)
