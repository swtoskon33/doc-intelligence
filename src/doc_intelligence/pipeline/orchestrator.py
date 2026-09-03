"""Document-processing pipeline: batch in, decisions out.

Runs the full flow a back office needs, in order:

    split -> classify -> pick a backend -> extract -> validate -> decide

The decision step is the point: each document is either auto-accepted or routed to a
human, with the reasons attached. Backend selection is policy, not magic -- the default
prefers the cheap deterministic backend and escalates to a fallback only when it fails.

Vendor memory (optional) fills stable fields -- IBAN, VAT rate, currency -- from the last
accepted document of the same sender when this one does not state them.

When a document arrives with word bounding boxes, process_layout_document routes it
through the layout-aware backend instead: same decision logic, same review routing,
but the entities come from LayoutLMv3 rather than regex over flat text.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from doc_intelligence.extraction.base import Extractor
from doc_intelligence.extraction.extractor import get_extractor
from doc_intelligence.ingest.documents import ingest_document
from doc_intelligence.memory.vendor import VendorMemory
from doc_intelligence.splitting.splitter import split_documents
from doc_intelligence.types import DocumentType, ExtractionResult, Field
from doc_intelligence.validation.rules import validate_in_place

RECALLED_CONFIDENCE = 0.6  # a remembered value is weaker evidence than a read one


class Decision(str, Enum):
    AUTO_ACCEPT = "auto_accept"
    NEEDS_REVIEW = "needs_review"


@dataclass
class ProcessedDocument:
    """One document after the full pipeline: what was extracted and what to do with it."""

    document_id: str
    page_range: tuple[int, int]
    doc_type: DocumentType
    result: ExtractionResult
    backend: str
    decision: Decision
    reasons: list[str]


def select_backend(doc_type: DocumentType, primary: Extractor,
                   fallback: Extractor | None) -> Extractor:
    """Backend policy: start with the primary; the fallback is used on escalation."""
    return primary


class DocumentPipeline:
    """Splits a batch, processes each document, and decides accept vs review."""

    def __init__(self, primary: Extractor | None = None,
                 fallback: Extractor | None = None,
                 memory: VendorMemory | None = None) -> None:
        self.primary = primary or get_extractor("rule")
        self.fallback = fallback
        self.memory = memory

    def _apply_memory(self, result: ExtractionResult) -> ExtractionResult:
        """Fill missing stable fields from what we know about this vendor."""
        if self.memory is None:
            return result
        vendor = result.value("vendor")
        known = self.memory.lookup(vendor) if vendor else {}
        filled = False
        for name, remembered in known.items():
            field = result.fields.get(name)
            if field is not None and field.value is None:
                field.value = remembered
                field.confidence = RECALLED_CONFIDENCE
                filled = True
        return validate_in_place(result) if filled else result

    def _process_one(self, doc_id: str, text: str, page_range: tuple[int, int],
                     doc_type: DocumentType) -> ProcessedDocument:
        doc = ingest_document(doc_id, text)
        doc.doc_type = doc_type
        extractor = select_backend(doc_type, self.primary, self.fallback)
        result = validate_in_place(extractor.extract(doc))

        # escalate to the fallback backend when the cheap one produced a weak result
        if self.fallback is not None and result.needs_review:
            alt_doc = ingest_document(doc_id, text)
            alt_doc.doc_type = doc_type
            alt = validate_in_place(self.fallback.extract(alt_doc))
            if not alt.needs_review:
                result, extractor = alt, self.fallback

        result = self._apply_memory(result)
        decision = Decision.NEEDS_REVIEW if result.needs_review else Decision.AUTO_ACCEPT
        return ProcessedDocument(
            document_id=doc_id, page_range=page_range, doc_type=result.doc_type,
            result=result, backend=extractor.name, decision=decision,
            reasons=result.review_reasons,
        )

    def process_batch(self, batch_id: str, pages: list[str]) -> list[ProcessedDocument]:
        """Split a page batch into documents and process each one."""
        out = []
        for i, split in enumerate(split_documents(pages)):
            out.append(self._process_one(
                f"{batch_id}_{i}", split.text, split.page_range, split.doc_type,
            ))
        return out

    def process_document(self, doc_id: str, text: str) -> ProcessedDocument:
        """Process a single document; the type is inferred from its text."""
        doc = ingest_document(doc_id, text)
        return self._process_one(doc_id, text, (0, 0), doc.doc_type)

    def process_layout_document(
        self, doc_id: str, words: list[str], boxes: list[list[int]], image=None,
    ) -> ProcessedDocument:
        """Process a document that carries word positions, via the layout backend.

        The layout backend predicts entities rather than schema fields, so its entities
        are mapped onto Fields keyed by entity label. Everything downstream -- confidence
        thresholds, validation, the accept/review decision -- is the same code path the
        text backends use.
        """
        from PIL import Image

        from doc_intelligence.extraction.layoutlm import LayoutLMv3Extractor

        backend = LayoutLMv3Extractor()
        if not backend.available:
            raise RuntimeError(
                "no LayoutLMv3 checkpoint; train one with scripts/train_layoutlm.py")
        if len(words) != len(boxes):
            raise ValueError("words and boxes must be the same length")

        page = image or Image.new("RGB", (1000, 1000), "white")
        entities = backend.predict_entities(page, words, boxes)

        # one field per entity, keyed label_index so repeated labels do not collide
        fields: dict[str, Field] = {}
        for i, ent in enumerate(entities):
            fields[f"{ent.label.lower()}_{i}"] = Field(
                name=f"{ent.label.lower()}_{i}", value=ent.text, confidence=ent.confidence)

        result = ExtractionResult(
            document_id=doc_id, doc_type=DocumentType.UNKNOWN, fields=fields)
        result = validate_in_place(result)
        decision = Decision.NEEDS_REVIEW if result.needs_review else Decision.AUTO_ACCEPT
        return ProcessedDocument(
            document_id=doc_id, page_range=(0, 0), doc_type=result.doc_type,
            result=result, backend=backend.name, decision=decision,
            reasons=result.review_reasons,
        )

    def remember_accepted(self, processed: ProcessedDocument) -> None:
        """Feed an auto-accepted document back into vendor memory."""
        if self.memory is None or processed.decision is not Decision.AUTO_ACCEPT:
            return
        vendor = processed.result.value("vendor")
        if vendor:
            self.memory.remember(vendor, {
                k: f.value for k, f in processed.result.fields.items() if f.value
            })
