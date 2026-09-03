"""LayoutLMv3 extraction backend: token classification over words and boxes.

Uses the fine-tuned checkpoint from scripts/train_layoutlm.py. Unlike the other
backends, this one is layout-aware: it consumes word positions, not just text, which is
what a form or invoice actually encodes.

Inference path: words + boxes -> processor -> token logits -> BIO decoding -> entities.
Each entity carries the mean softmax probability of its tokens as confidence, and the
bounding box of its span, so a reviewer can be shown where a value came from.

The checkpoint is loaded lazily and the backend reports itself unavailable when it is
missing, so CI and the other backends are unaffected.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from doc_intelligence.extraction.base import Extractor
from doc_intelligence.layout.dataset import ID2LABEL
from doc_intelligence.types import ExtractionResult, Field, RawDocument

CHECKPOINT = Path(os.getenv("LAYOUTLM_OUTPUT_DIR", "models/layoutlmv3"))


@dataclass
class Entity:
    """A decoded BIO span: its label, text, confidence and bounding box."""

    label: str
    text: str
    confidence: float
    box: list[int]


def decode_bio(words, boxes, label_ids, scores) -> list[Entity]:
    """Group word-level BIO predictions into entities.

    A B- tag opens a span, matching I- tags extend it, anything else closes it. The
    entity confidence is the mean token probability; its box is the union of the word
    boxes it covers.
    """
    entities: list[Entity] = []
    current_label, current_words, current_scores, current_boxes = None, [], [], []

    def flush():
        if current_label and current_words:
            xs0 = min(b[0] for b in current_boxes)
            ys0 = min(b[1] for b in current_boxes)
            xs1 = max(b[2] for b in current_boxes)
            ys1 = max(b[3] for b in current_boxes)
            entities.append(Entity(
                label=current_label,
                text=" ".join(current_words),
                confidence=round(sum(current_scores) / len(current_scores), 3),
                box=[xs0, ys0, xs1, ys1],
            ))

    for word, box, label_id, score in zip(words, boxes, label_ids, scores):
        tag = ID2LABEL.get(int(label_id), "O")
        if tag.startswith("B-"):
            flush()
            current_label = tag[2:]
            current_words, current_scores, current_boxes = [word], [score], [box]
        elif tag.startswith("I-") and current_label == tag[2:]:
            current_words.append(word)
            current_scores.append(score)
            current_boxes.append(box)
        else:
            flush()
            current_label, current_words, current_scores, current_boxes = None, [], [], []
    flush()
    return entities


class LayoutLMv3Extractor(Extractor):
    """Fine-tuned LayoutLMv3 token classifier as an extraction backend."""

    name = "layoutlmv3"

    def __init__(self, checkpoint: Path | None = None) -> None:
        self.checkpoint = checkpoint or CHECKPOINT
        self.available = self.checkpoint.exists()
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is None:
            from transformers import AutoProcessor, LayoutLMv3ForTokenClassification

            self._processor = AutoProcessor.from_pretrained(
                str(self.checkpoint), apply_ocr=False)
            self._model = LayoutLMv3ForTokenClassification.from_pretrained(
                str(self.checkpoint))
            self._model.eval()
        return self._processor, self._model

    def predict_entities(self, image, words: list[str], boxes: list[list[int]]) -> list[Entity]:
        """Run the model over a page and return decoded entities."""
        import torch

        processor, model = self._load()
        encoding = processor(image, words, boxes=boxes, truncation=True,
                             padding="max_length", max_length=512, return_tensors="pt")
        word_ids = encoding.word_ids(0)
        with torch.no_grad():
            logits = model(**encoding).logits[0]
        probs = torch.softmax(logits, dim=-1)
        best = probs.argmax(-1)

        # take the first subword of each word as that word's prediction
        seen: set[int] = set()
        word_labels, word_scores = [], []
        for pos, wid in enumerate(word_ids):
            if wid is None or wid in seen:
                continue
            seen.add(wid)
            word_labels.append(int(best[pos]))
            word_scores.append(float(probs[pos, best[pos]]))

        n = min(len(words), len(word_labels))
        return decode_bio(words[:n], boxes[:n], word_labels[:n], word_scores[:n])

    def extract(self, doc: RawDocument) -> ExtractionResult:
        """Text-only fallback path: without word boxes the model cannot be applied.

        The backend is layout-aware by design; the API supplies words and boxes through
        predict_entities. This keeps the Extractor interface satisfied and makes the
        limitation explicit rather than silently returning noise.
        """
        return ExtractionResult(
            document_id=doc.id, doc_type=doc.doc_type,
            fields={"_layout_required": Field("_layout_required", None, 0.0)},
        )
