"""LLM extractor: structured JSON extraction with an LLM (OpenAI / Azure OpenAI).

The production path. The prompt is built from the same YAML schema the rule backend
uses, and the model is asked for strict JSON so the output maps onto typed Fields.

Offline behaviour: without an API key the extractor runs in `mock` mode -- it returns
empty fields with zero confidence rather than failing, so CI and the benchmark harness
can include the backend without network access. Real runs set OPENAI_API_KEY.
"""
from __future__ import annotations

import json
import os

from doc_intelligence.extraction.base import Extractor
from doc_intelligence.schemas.registry import load_schema
from doc_intelligence.types import ExtractionResult, Field, RawDocument


def _build_prompt(doc: RawDocument, field_names: list[str]) -> str:
    fields = ", ".join(field_names)
    return (
        "Extract the following fields from the document text and return strict JSON "
        f"with exactly these keys: {fields}. Use null when a field is absent. "
        "Return only JSON, no prose.\n\nDocument:\n" + doc.text
    )


class LLMExtractor(Extractor):
    """Schema-driven extraction via an LLM with JSON output."""

    name = "llm"

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.available = bool(self.api_key)

    def _call(self, prompt: str) -> dict:
        """Call the LLM. Only reached when an API key is configured."""
        from openai import OpenAI  # imported lazily so the package stays optional

        client = OpenAI(api_key=self.api_key)
        resp = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        return json.loads(resp.choices[0].message.content)

    def extract(self, doc: RawDocument) -> ExtractionResult:
        specs = load_schema(doc.doc_type)
        names = [s.name for s in specs]

        if not self.available:
            # offline mock: no key, so report nothing found rather than crashing
            fields = {n: Field(n, None, 0.0) for n in names}
            return ExtractionResult(doc.id, doc.doc_type, fields)

        data = self._call(_build_prompt(doc, names))
        fields = {}
        for n in names:
            raw = data.get(n)
            value = str(raw).strip() if raw not in (None, "", "null") else None
            # an LLM that returns a value for a requested field is treated as confident
            fields[n] = Field(n, value, confidence=0.9 if value else 0.0)
        return ExtractionResult(doc.id, doc.doc_type, fields)
