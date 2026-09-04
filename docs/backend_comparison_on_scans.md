# Rule versus layout backend on real scans

Both extraction backends over the same 50 real scanned pages from FUNSD. Not an accuracy comparison: FUNSD labels are form semantics, not invoice fields, so there is no ground truth both backends can be scored against. This measures reach -- how much either gets out of a real scan at all.

| Backend | Values extracted | Latency / page |
|---------|------------------|----------------|
| rule (regex over flat text) | 7 | 0.17 ms |
| LayoutLMv3 (words + boxes) | 1919 (1617 above 0.7 confidence) | 514 ms |

The layout backend pulls about 274x more out of the same pages, at roughly 3073x the cost per page. That settles what the earlier probe could only assert. The regex patterns need a label and its value adjacent in a line of text; a scan puts them in separate cells, and no pattern tuning recovers a spatial relationship from a flattened string. The layout model reads position, so it finds structure the other backend cannot see.

The cost side is equally clear, and it is why both ship. Microseconds per page against hundreds of milliseconds is the difference between running on every document and running only on the ones that need it.
