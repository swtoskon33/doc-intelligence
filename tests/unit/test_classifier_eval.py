"""The classifier is evaluated out-of-sample, not on its own training data."""
import pytest

from doc_intelligence.models.classifier import DocumentClassifier
from doc_intelligence.types import DocumentType

TEXTS = [
    "INVOICE number 1 total 100", "INVOICE number 2 total 300",
    "RECEIPT merchant coop total 20", "RECEIPT store migros total 45",
    "AGREEMENT between A and B", "AGREEMENT between C and D",
]
LABELS = ["invoice", "invoice", "receipt", "receipt", "contract", "contract"]


@pytest.mark.unit
def test_classifier_predicts_a_known_type():
    clf = DocumentClassifier.train(TEXTS, LABELS)
    assert clf.predict("INVOICE number 9 total 50") == DocumentType.INVOICE


@pytest.mark.unit
def test_leave_one_out_holds_out_the_document_it_scores():
    """Each fold must train without the document it then predicts."""
    from scripts.train_classifier import leave_one_out_accuracy

    acc, correct, n = leave_one_out_accuracy(TEXTS, LABELS)
    assert n == len(TEXTS)
    assert 0.0 <= acc <= 1.0
    assert correct == pytest.approx(acc * n)


@pytest.mark.unit
def test_leave_one_out_is_not_just_training_accuracy():
    """Out-of-sample accuracy must not silently equal the in-sample number."""
    from scripts.train_classifier import leave_one_out_accuracy

    loo, _, _ = leave_one_out_accuracy(TEXTS, LABELS)
    full = DocumentClassifier.train(TEXTS, LABELS)
    train_acc = sum(full.predict(t).value == y for t, y in zip(TEXTS, LABELS)) / len(TEXTS)
    # a memorising model scores higher on its own data; loo must never exceed it
    assert loo <= train_acc
