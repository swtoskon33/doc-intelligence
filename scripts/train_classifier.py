"""Train the baseline document classifier and evaluate it honestly.

The classifier is a small TF-IDF + logistic-regression model over document text. With
only 15 documents, accuracy on the training set is meaningless -- the model can memorise
it. Reported numbers therefore come from leave-one-out cross-validation: each document is
predicted by a model that never saw it, which is the strictest available estimate at this
sample size.

The keyword heuristic in `ingest` needs no training, so it is scored on the same
documents for a like-for-like comparison.

    python scripts/train_classifier.py
"""
from __future__ import annotations

import json
from pathlib import Path

from doc_intelligence.ingest.documents import infer_type
from doc_intelligence.models.classifier import DocumentClassifier
from doc_intelligence.types import DocumentType

GOLDEN = Path("data/samples/golden.json")


def leave_one_out_accuracy(texts: list[str], labels: list[str]) -> tuple[float, int, int]:
    """Predict each document with a model trained on all the others.

    Returns (accuracy, correct, n). Leave-one-out rather than k-fold because some classes
    have very few examples; holding out a single document keeps every fold trainable.
    """
    correct = 0
    for i in range(len(texts)):
        train_texts = texts[:i] + texts[i + 1:]
        train_labels = labels[:i] + labels[i + 1:]
        if len(set(train_labels)) < 2:      # a fold with one class cannot be fitted
            continue
        model = DocumentClassifier.train(train_texts, train_labels)
        if model.predict(texts[i]).value == labels[i]:
            correct += 1
    return correct / len(texts), correct, len(texts)


def main() -> None:
    samples = json.loads(GOLDEN.read_text())
    texts = [s["text"] for s in samples]
    labels = [s["doc_type"] for s in samples]
    n = len(samples)

    # honest estimate: every prediction comes from a model that never saw the document
    loo_acc, loo_correct, _ = leave_one_out_accuracy(texts, labels)

    # the heuristic has no training phase, so the same documents are a fair test for it
    kw_correct = sum(infer_type(t) == DocumentType(y) for t, y in zip(texts, labels))
    kw_acc = kw_correct / n

    # the shipped model is fitted on everything; its training accuracy is reported only
    # to show the gap, never as a performance claim
    full = DocumentClassifier.train(texts, labels)
    full.save()
    train_correct = sum(full.predict(t).value == y for t, y in zip(texts, labels))

    print(f"documents: {n}")
    print(f"  trained classifier, leave-one-out: {loo_correct}/{n} = {loo_acc:.3f}")
    print(f"  keyword heuristic:                 {kw_correct}/{n} = {kw_acc:.3f}")
    print(f"  (trained classifier on its own training data: "
          f"{train_correct}/{n} = {train_correct / n:.3f} -- memorisation, not performance)")
    print("saved models/classifier.pkl")

    report = {
        "documents": n,
        "leave_one_out_accuracy": round(loo_acc, 3),
        "keyword_heuristic_accuracy": round(kw_acc, 3),
        "training_accuracy_for_reference_only": round(train_correct / n, 3),
        "method": ("leave-one-out cross-validation; the keyword heuristic requires no "
                   "training so it is scored on the same documents"),
    }
    Path("docs/classifier_report.json").write_text(json.dumps(report, indent=2))
    print("wrote docs/classifier_report.json")


if __name__ == "__main__":
    main()
