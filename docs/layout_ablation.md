# Layout ablation

Does layout actually contribute, or is LayoutLMv3 just a text model with extra machinery? Measured on the fine-tuned checkpoint over the FUNSD test split (50 documents) by degrading the spatial signal at inference time.

| Layout input | Precision | Recall | F1 | F1 drop vs full |
|--------------|-----------|--------|-----|-----------------|
| full (real boxes) | 0.7913 | 0.8358 | 0.8129 | - |
| no layout (zeroed boxes) | 0.2154 | 0.0071 | 0.0137 | 0.7992 |
| shuffled boxes | 0.119 | 0.2296 | 0.1567 | 0.6562 |

Method: `full` uses the real bounding boxes the model was trained on; `no-layout` collapses every box to [0,0,0,0] so all words claim the same position; `shuffled` permutes the boxes so each word carries another word's position. Text and image inputs are identical in all three. The drop is therefore attributable to spatial information alone.

Removing layout costs 0.799 F1 and scrambling it costs 0.656 F1: without spatial information the model collapses almost completely. LayoutLMv3 is not a text model with extra machinery, the layout signal dominates.

The ordering is worth noting. Zeroed boxes score *worse* than shuffled ones (0.014 vs 0.157) even though shuffling actively misinforms the model. Collapsing every box to [0,0,0,0] puts the input far outside the distribution the model was trained on, whereas shuffled boxes are still valid coordinates, just attached to the wrong words, so the model degrades rather than breaks.

This is the empirical case for a layout-aware model over a text-only one on structured documents.
