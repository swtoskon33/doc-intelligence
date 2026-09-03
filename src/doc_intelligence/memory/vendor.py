"""Vendor memory: reuse what we already know about a sender.

Invoices from the same vendor repeat: the same IBAN, the same VAT rate, the same
currency. When extraction misses one of those fields, the most recent accepted document
from that vendor is a better guess than nothing -- and a *contradiction* with it is a
signal worth flagging.

Deliberately simple: an in-memory store keyed by vendor name, with exact then substring
matching. That is enough to demonstrate the recall-and-fill behaviour and its effect on
the review rate. A production version would key on a normalised vendor identifier (or
the IBAN) and live in a database rather than a JSON file -- the interface here is the
part that would survive that change.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

STORE = Path("data/vendor_memory.json")


@dataclass
class VendorRecord:
    vendor: str
    fields: dict[str, str] = field(default_factory=dict)


class VendorMemory:
    """Small nearest-neighbour store of previously accepted vendor documents."""

    def __init__(self, records: list[VendorRecord] | None = None) -> None:
        self.records = records or []

    def remember(self, vendor: str, fields: dict[str, str]) -> None:
        """Store the stable fields of an accepted document for this vendor."""
        keep = {k: v for k, v in fields.items()
                if k in ("iban", "mwst_rate", "currency") and v}
        if not vendor or not keep:
            return
        for rec in self.records:
            if rec.vendor.lower() == vendor.lower():
                rec.fields.update(keep)
                return
        self.records.append(VendorRecord(vendor=vendor, fields=keep))

    def lookup(self, vendor: str) -> dict[str, str]:
        """Exact-then-substring match on vendor name; empty dict when unknown."""
        if not vendor:
            return {}
        low = vendor.lower()
        for rec in self.records:
            if rec.vendor.lower() == low:
                return dict(rec.fields)
        for rec in self.records:
            if low in rec.vendor.lower() or rec.vendor.lower() in low:
                return dict(rec.fields)
        return {}

    def save(self, path: Path = STORE) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(
            [{"vendor": r.vendor, "fields": r.fields} for r in self.records], indent=2))

    @classmethod
    def load(cls, path: Path = STORE) -> VendorMemory:
        if not path.exists():
            return cls()
        data = json.loads(path.read_text())
        return cls([VendorRecord(d["vendor"], d.get("fields", {})) for d in data])
