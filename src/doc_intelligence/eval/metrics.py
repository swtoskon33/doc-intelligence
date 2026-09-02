"""Evaluation: measure extraction quality against ground truth, field by field.

Document extraction is judged per field, not per document -- an extractor that gets the
total right but the date wrong should score partial credit, and you want to know *which*
fields are weak. This computes per-field precision/recall/F1 plus overall accuracy, over
a labelled set.
"""
from __future__ import annotations

from dataclasses import dataclass

from doc_intelligence.types import ExtractionResult


@dataclass
class FieldScore:
    field: str
    precision: float
    recall: float
    f1: float
    support: int  # how many documents had a ground-truth value for this field


@dataclass
class EvalReport:
    per_field: dict[str, FieldScore]
    overall_accuracy: float  # fraction of all ground-truth fields extracted correctly


def _norm(value):
    """Normalise a value for comparison: lowercase, strip, drop thousands separators."""
    if value is None:
        return None
    return value.strip().lower().replace(",", "")


def evaluate(results: list[ExtractionResult], truth: dict[str, dict[str, str]]) -> EvalReport:
    """Compare extractions against ground truth.

    results: extractor outputs.
    truth: {document_id: {field_name: correct_value}}.
    """
    # tp/fp/fn per field
    tp: dict[str, int] = {}
    fp: dict[str, int] = {}
    fn: dict[str, int] = {}
    correct = 0
    total_truth = 0

    for res in results:
        gold = truth.get(res.document_id, {})
        # union of fields the extractor produced and fields that have a gold value
        names = set(res.fields) | set(gold)
        for name in names:
            pred = _norm(res.value(name))
            gold_val = _norm(gold.get(name))
            if gold_val is not None:
                total_truth += 1
            if pred is not None and gold_val is not None and pred == gold_val:
                tp[name] = tp.get(name, 0) + 1
                correct += 1
            elif pred is not None and pred != gold_val:
                fp[name] = fp.get(name, 0) + 1
                if gold_val is not None:
                    fn[name] = fn.get(name, 0) + 1
            elif pred is None and gold_val is not None:
                fn[name] = fn.get(name, 0) + 1

    per_field: dict[str, FieldScore] = {}
    for name in set(tp) | set(fp) | set(fn):
        t, f_p, f_n = tp.get(name, 0), fp.get(name, 0), fn.get(name, 0)
        precision = t / (t + f_p) if (t + f_p) else 0.0
        recall = t / (t + f_n) if (t + f_n) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_field[name] = FieldScore(name, round(precision, 3), round(recall, 3),
                                     round(f1, 3), support=t + f_n)

    overall = correct / total_truth if total_truth else 0.0
    return EvalReport(per_field=per_field, overall_accuracy=round(overall, 3))
