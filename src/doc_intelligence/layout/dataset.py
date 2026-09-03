"""FUNSD dataset for LayoutLMv3 token classification.

FUNSD is the standard form-understanding benchmark: 199 annotated documents (149 train,
50 test) with word-level bounding boxes and BIO labels. It ships ready for LayoutLMv3 --
images, tokens, boxes and ner_tags -- which is why it is the reference dataset in the
LayoutLMv3 model card.

Labels are form semantics (question / answer / header / other) rather than invoice
fields. The same pipeline retrains on invoice-specific labels once annotated invoices
exist; nothing in the training or inference code depends on the label set.
"""
from __future__ import annotations

from dataclasses import dataclass

DATASET_NAME = "nielsr/funsd-layoutlmv3"

# FUNSD BIO tag set, in the dataset's own order.
LABELS = [
    "O",
    "B-HEADER", "I-HEADER",
    "B-QUESTION", "I-QUESTION",
    "B-ANSWER", "I-ANSWER",
]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = dict(enumerate(LABELS))


@dataclass
class DatasetSplits:
    train: object
    validation: object
    test: object

    @property
    def sizes(self) -> dict[str, int]:
        return {
            "train": len(self.train),
            "validation": len(self.validation),
            "test": len(self.test),
        }


def datasets_available() -> bool:
    try:
        import datasets  # noqa: F401
    except ImportError:
        return False
    return True


def load_funsd(val_fraction: float = 0.2, seed: int = 42) -> DatasetSplits:
    """Load FUNSD and carve a validation split out of train.

    FUNSD ships only train/test, so validation comes from a seeded split of train --
    document-level, so no document appears in two splits.
    """
    from datasets import load_dataset

    raw = load_dataset(DATASET_NAME)
    split = raw["train"].train_test_split(test_size=val_fraction, seed=seed)
    return DatasetSplits(
        train=split["train"], validation=split["test"], test=raw["test"],
    )


def normalize_box(box: list[int], width: int, height: int) -> list[int]:
    """Scale a pixel box to LayoutLMv3's 0-1000 coordinate space, clamped in range."""
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    x0, y0, x1, y1 = box
    scaled = [
        int(1000 * x0 / width), int(1000 * y0 / height),
        int(1000 * x1 / width), int(1000 * y1 / height),
    ]
    return [max(0, min(1000, v)) for v in scaled]
