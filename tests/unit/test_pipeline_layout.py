"""The orchestrator can route a document through the layout backend."""
import pytest

from doc_intelligence.extraction.layoutlm import LayoutLMv3Extractor
from doc_intelligence.pipeline.orchestrator import Decision, DocumentPipeline

CHECKPOINT = LayoutLMv3Extractor().available

WORDS = ["TO:", "John", "Smith", "DATE:", "2026-01-01"]
BOXES = [[0, 0, 50, 20], [60, 0, 120, 20], [130, 0, 200, 20],
         [0, 30, 60, 50], [70, 30, 200, 50]]


@pytest.mark.unit
def test_layout_path_rejects_mismatched_lengths():
    with pytest.raises((ValueError, RuntimeError)):
        DocumentPipeline().process_layout_document("d1", WORDS, BOXES[:2])


@pytest.mark.unit
@pytest.mark.skipif(not CHECKPOINT, reason="no fine-tuned checkpoint available")
def test_layout_path_uses_the_same_decision_logic():
    out = DocumentPipeline().process_layout_document("d1", WORDS, BOXES)
    assert out.backend == "layoutlmv3"
    assert out.decision in (Decision.AUTO_ACCEPT, Decision.NEEDS_REVIEW)
    assert out.result.fields                       # entities became fields
    for f in out.result.fields.values():
        assert 0.0 <= f.confidence <= 1.0
    if out.decision is Decision.NEEDS_REVIEW:
        assert out.reasons                         # review always explains itself
