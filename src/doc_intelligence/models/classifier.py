"""Trained document classifier: TF-IDF + logistic regression.

The keyword classifier in `ingest` is a baseline; this is the learned alternative,
trained on the labelled golden set. Small and offline (scikit-learn, no GPU), it exists
to make the comparison honest: a trained model is measured against the heuristic rather
than assumed to be better.

Train:   python scripts/train_classifier.py
Predict: DocumentClassifier.load().predict(text)
"""
from __future__ import annotations

import pickle
from pathlib import Path

from doc_intelligence.types import DocumentType

MODEL_PATH = Path("models/classifier.pkl")


class DocumentClassifier:
    """Wraps a fitted sklearn pipeline mapping document text to a DocumentType."""

    def __init__(self, pipeline) -> None:
        self.pipeline = pipeline

    @classmethod
    def train(cls, texts: list[str], labels: list[str]) -> DocumentClassifier:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline

        pipe = make_pipeline(
            TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1),
            LogisticRegression(max_iter=1000),
        )
        pipe.fit(texts, labels)
        return cls(pipe)

    def predict(self, text: str) -> DocumentType:
        label = self.pipeline.predict([text])[0]
        try:
            return DocumentType(label)
        except ValueError:
            return DocumentType.UNKNOWN

    def confidence(self, text: str) -> float:
        return float(max(self.pipeline.predict_proba([text])[0]))

    def save(self, path: Path = MODEL_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.pipeline, f)

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> DocumentClassifier | None:
        """Load a trained classifier, or None if it has not been trained yet."""
        if not path.exists():
            return None
        with open(path, "rb") as f:
            return cls(pickle.load(f))
