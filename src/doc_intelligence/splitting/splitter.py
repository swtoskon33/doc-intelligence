"""Document splitting: break a multi-document page batch into individual documents.

Scanned batches and multi-page PDFs often contain several documents concatenated (an
invoice, then a receipt, then a contract). Splitting detects the boundaries so each
document is extracted on its own.

Offline heuristic: a new document starts on a page whose text shows a strong type signal
(an "INVOICE" / "RECEIPT" / "AGREEMENT" header). A real deployment can swap in an LLM
boundary classifier behind split_documents.
"""
from __future__ import annotations

from dataclasses import dataclass

from doc_intelligence.ingest.documents import infer_type
from doc_intelligence.types import DocumentType

_START_SIGNALS = [
    "invoice", "rechnung", "receipt", "quittung", "agreement", "vertrag", "contract",
]


@dataclass
class SplitDocument:
    """One detected document within a batch: its text, page range, and inferred type."""

    text: str
    page_start: int
    page_end: int
    doc_type: DocumentType

    @property
    def page_range(self) -> tuple[int, int]:
        return (self.page_start, self.page_end)


def _is_boundary(page_text: str) -> bool:
    """True if this page looks like the start of a new document."""
    low = page_text.lower()
    first_line = next((ln for ln in low.splitlines() if ln.strip()), "")
    return any(sig in first_line for sig in _START_SIGNALS)


def split_documents(pages: list[str]) -> list[SplitDocument]:
    """Group page texts into individual documents by boundary detection."""
    if not pages:
        return []
    starts = [0]
    for i in range(1, len(pages)):
        if _is_boundary(pages[i]):
            starts.append(i)
    docs = []
    for idx, start in enumerate(starts):
        end = (starts[idx + 1] - 1) if idx + 1 < len(starts) else len(pages) - 1
        text = "\n".join(pages[start:end + 1])
        docs.append(SplitDocument(
            text=text, page_start=start, page_end=end, doc_type=infer_type(text),
        ))
    return docs
