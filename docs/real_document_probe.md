# Rule backend on real scanned documents

The rule backend run over 50 real scanned pages from FUNSD, which carry the noise a scanner actually produces. This is not an accuracy score: FUNSD is forms, not invoices, so most pages have no invoice field to find. It measures reach -- how often the patterns fire at all on text they were not written against.

| Measure | Result |
|---------|--------|
| Documents processed | 50 |
| Crashes | 0 |
| Document type recognised | 15/50 |
| Flagged for review | 15/50 |
| Field values extracted (total) | 9 |

Fields found, by name:

| Field | Documents |
|-------|-----------|
| vendor | 4 |
| party | 2 |
| merchant | 1 |
| total_amount | 1 |
| invoice_number | 1 |

Nothing crashes, and almost nothing is extracted: 9 field values across 50 documents, with 35 pages whose type the classifier cannot place. Part of that is fair -- these are forms, and a form has no invoice number. But the pages it *does* read as invoices give up barely anything, and that is the real result.

The patterns expect `Total: CHF 1081.00` on one line. A scan gives a label in one cell and its value in another, spacing that survives no regex, and characters the OCR guessed at. A regex matches a string; it has no notion that the number to the right of the word Total is the total. That is exactly the boundary the LayoutLMv3 backend exists to cross: it reads position as well as text, so a value means something because of where it sits relative to its label.

So the honest reading of this repo's two extraction paths is: rules for clean, structured input where the format is stable and the cost is microseconds; a layout-aware model for anything that came off a scanner.

