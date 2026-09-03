"""Tests for the LayoutLMv3 backend: BIO decoding, boxes, and availability."""
import pytest

from doc_intelligence.extraction.layoutlm import Entity, LayoutLMv3Extractor, decode_bio
from doc_intelligence.layout.dataset import LABEL2ID, normalize_box


@pytest.mark.unit
def test_box_normalization_scales_to_1000_space():
    assert normalize_box([50, 100, 150, 200], 500, 1000) == [100, 100, 300, 200]


@pytest.mark.unit
def test_box_normalization_clamps_out_of_range():
    assert normalize_box([-10, 0, 900, 1200], 500, 1000) == [0, 0, 1000, 1000]


@pytest.mark.unit
def test_box_normalization_rejects_zero_dimensions():
    with pytest.raises(ValueError):
        normalize_box([0, 0, 10, 10], 0, 100)


@pytest.mark.unit
def test_bio_decoding_joins_continuation_tags():
    words = ["Attorney", "General", "Betty"]
    boxes = [[0, 0, 10, 10], [11, 0, 20, 10], [21, 0, 30, 10]]
    labels = [LABEL2ID["B-ANSWER"], LABEL2ID["I-ANSWER"], LABEL2ID["I-ANSWER"]]
    entities = decode_bio(words, boxes, labels, [0.9, 0.8, 0.7])
    assert len(entities) == 1
    assert entities[0].text == "Attorney General Betty"
    assert entities[0].box == [0, 0, 30, 10]          # union of the word boxes
    assert entities[0].confidence == pytest.approx(0.8, abs=0.01)  # mean of scores


@pytest.mark.unit
def test_bio_decoding_starts_a_new_entity_on_b_tag():
    labels = [LABEL2ID["B-QUESTION"], LABEL2ID["B-ANSWER"]]
    entities = decode_bio(["Total:", "100"], [[0, 0, 5, 5], [6, 0, 10, 5]], labels, [0.9, 0.9])
    assert [e.label for e in entities] == ["QUESTION", "ANSWER"]


@pytest.mark.unit
def test_bio_decoding_ignores_orphan_i_tags():
    # an I- tag with no matching B- before it must not open an entity
    entities = decode_bio(["stray"], [[0, 0, 5, 5]], [LABEL2ID["I-ANSWER"]], [0.9])
    assert entities == []


@pytest.mark.unit
def test_backend_reports_unavailable_without_checkpoint(tmp_path):
    ex = LayoutLMv3Extractor(checkpoint=tmp_path / "does-not-exist")
    assert not ex.available
    assert ex.name == "layoutlmv3"


@pytest.mark.unit
def test_entity_carries_a_box():
    e = Entity(label="ANSWER", text="x", confidence=0.5, box=[1, 2, 3, 4])
    assert e.box == [1, 2, 3, 4]
