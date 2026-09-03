# Layout ablation

Does layout actually contribute, or is LayoutLMv3 just a text model with extra machinery? Measured on the fine-tuned checkpoint over the FUNSD test split (50 documents) by degrading the spatial signal at inference time.

| Layout input | Precision | Recall | F1 | F1 drop vs full |
|--------------|-----------|--------|-----|-----------------|
| full (real boxes) | 0.7913 | 0.8358 | 0.8129 | - |
| no layout (zeroed boxes) | 0.2154 | 0.0071 | 0.0137 | 0.7992 |
| shuffled boxes | 0.119 | 0.2296 | 0.1567 | 0.6562 |

Method: `full` uses the real bounding boxes the model was trained on; `no-layout` collapses every box to [0,0,0,0] so all words claim the same position; `shuffled` permutes the boxes so each word carries another word's position. Text and image inputs are identical in all three. The drop is therefore attributable to spatial information alone.

Removing layout costs 0.7992 F1 and scrambling it costs 0.6562 F1. Shuffling is the harsher condition: a missing signal is merely uninformative, while a wrong one actively misleads the model. This is the empirical case for a layout-aware model over a text-only one on structured documents.
