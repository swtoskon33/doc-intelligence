"""Train the baseline document classifier on the labelled golden set.

Compares the trained model against the keyword heuristic already in `ingest`, so the
choice between them is measured rather than assumed.

    python scripts/train_classifier.py
"""
from __future__ import annotations

import json
from pathlib import Path

from doc_intelligence.ingest.documents import infer_type
from doc_intelligence.models.classifier import DocumentClassifier
from doc_intelligence.types import DocumentType

GOLDEN = Path("data/samples/golden.json")


def main() -> None:
    samples = json.loads(GOLDEN.read_text())
    texts = [s["text"] for s in samples]
    labels = [s["doc_type"] for s in samples]

    clf = DocumentClassifier.train(texts, labels)
    clf.save()

    # measure both classifiers on the same data
    trained_correct = sum(clf.predict(t).value == y for t, y in zip(texts, labels))
    keyword_correct = sum(infer_type(t) == DocumentType(y) for t, y in zip(texts, labels))
    n = len(samples)

    print(f"trained on {n} documents -> models/classifier.pkl")
    print(f"  trained classifier accuracy: {trained_correct}/{n} = {trained_correct/n:.3f}")
    print(f"  keyword heuristic accuracy:  {keyword_correct}/{n} = {keyword_correct/n:.3f}")


if __name__ == "__main__":
    main()
