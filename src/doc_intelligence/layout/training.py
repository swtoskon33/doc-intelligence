"""LayoutLMv3 preprocessing and fine-tuning.

Turns FUNSD examples (image, words, boxes, BIO tags) into model inputs and fine-tunes
LayoutLMv3ForTokenClassification. Everything that matters is configurable through the
environment, and the run is seeded so results are reproducible.

The dataset already carries normalised 0-1000 boxes, so the processor is created with
apply_ocr=False -- we supply words and boxes ourselves rather than letting it run its
own OCR, which is what you would do in production with a real OCR backend.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np

from doc_intelligence.layout.dataset import ID2LABEL, LABEL2ID, LABELS

MODEL_NAME = os.getenv("LAYOUTLM_MODEL_NAME", "microsoft/layoutlmv3-base")


@dataclass
class TrainConfig:
    """Training hyperparameters, all overridable from the environment."""

    model_name: str = MODEL_NAME
    max_length: int = int(os.getenv("LAYOUTLM_MAX_LENGTH", "512"))
    batch_size: int = int(os.getenv("LAYOUTLM_BATCH_SIZE", "2"))
    learning_rate: float = float(os.getenv("LAYOUTLM_LEARNING_RATE", "5e-5"))
    epochs: int = int(os.getenv("LAYOUTLM_EPOCHS", "3"))
    weight_decay: float = float(os.getenv("LAYOUTLM_WEIGHT_DECAY", "0.01"))
    seed: int = int(os.getenv("LAYOUTLM_SEED", "42"))
    device: str = os.getenv("LAYOUTLM_DEVICE", "")
    output_dir: str = os.getenv("LAYOUTLM_OUTPUT_DIR", "models/layoutlmv3")


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(preferred: str = "") -> str:
    """CUDA if present, then Apple MPS, else CPU."""
    import torch

    if preferred:
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_processor(model_name: str = MODEL_NAME):
    """LayoutLMv3 processor with OCR disabled -- we provide words and boxes."""
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(model_name, apply_ocr=False)


def encode_example(processor, example, max_length: int = 512):
    """Encode one document: image + words + boxes + word-level labels -> model inputs.

    The processor splits words into subword tokens, so labels are propagated with
    word_ids(): the first subword of a word carries its label, continuation subwords and
    special tokens are masked out with -100 so they do not contribute to the loss.
    """
    encoding = processor(
        example["image"].convert("RGB"),
        example["tokens"],
        boxes=example["bboxes"],
        word_labels=example["ner_tags"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    return {k: v.squeeze(0) for k, v in encoding.items()}


def build_dataset(processor, hf_split, max_length: int = 512):
    """Wrap a HuggingFace split in a torch Dataset that encodes lazily."""
    import torch

    class EncodedDocuments(torch.utils.data.Dataset):
        def __len__(self):
            return len(hf_split)

        def __getitem__(self, idx):
            return encode_example(processor, hf_split[idx], max_length)

    return EncodedDocuments()


def compute_metrics(predictions, label_ids):
    """Token-level and entity-level metrics, ignoring masked (-100) positions."""
    from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

    preds = np.argmax(predictions, axis=2)
    true_labels, true_preds = [], []
    for pred_row, label_row in zip(preds, label_ids):
        row_labels, row_preds = [], []
        for p, lbl in zip(pred_row, label_row):
            if lbl == -100:
                continue
            row_labels.append(ID2LABEL[int(lbl)])
            row_preds.append(ID2LABEL[int(p)])
        true_labels.append(row_labels)
        true_preds.append(row_preds)

    return {
        "precision": round(precision_score(true_labels, true_preds), 4),
        "recall": round(recall_score(true_labels, true_preds), 4),
        "f1": round(f1_score(true_labels, true_preds), 4),
        "report": classification_report(true_labels, true_preds, output_dict=True),
    }


def load_model(model_name: str = MODEL_NAME, num_labels: int = len(LABELS)):
    from transformers import LayoutLMv3ForTokenClassification

    return LayoutLMv3ForTokenClassification.from_pretrained(
        model_name, num_labels=num_labels, id2label=ID2LABEL, label2id=LABEL2ID,
    )
