"""Schema registry: field definitions per document type, loaded from YAML.

Moving schemas out of code means new fields or document types are a YAML edit, not a
code change -- and both the rule and LLM extractors consume the same definitions, so
they stay in sync.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import yaml

from doc_intelligence.types import DocumentType

_SCHEMA_DIR = Path(__file__).parent


@dataclass
class FieldSpec:
    name: str
    pattern: re.Pattern
    required: bool


@cache
def load_schema(doc_type: DocumentType) -> list[FieldSpec]:
    """Load and compile the field specs for a document type (cached)."""
    path = _SCHEMA_DIR / f"{doc_type.value}.yaml"
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text())
    specs = []
    for name, cfg in data.get("fields", {}).items():
        specs.append(FieldSpec(
            name=name,
            pattern=re.compile(cfg["pattern"], re.IGNORECASE),
            required=bool(cfg.get("required", False)),
        ))
    return specs


def required_fields(doc_type: DocumentType) -> list[str]:
    """Names of required fields for a document type (used by validation)."""
    return [s.name for s in load_schema(doc_type) if s.required]
