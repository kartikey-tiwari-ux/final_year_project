# Project Architecture

Grounded in `PROJECT_REQUIREMENTS_ANALYSIS.md`, `LITERATURE_REVIEW.md`, `RESEARCH_GAP.md`,
and `DATASET_SELECTION.md`. Model choices are explicitly justified against this
project's real hardware constraint: **CPU-only, no GPU** (see `REPRODUCIBILITY.md`).

## Dataset-derived constants (MVSA-Single, primary — see `DATASET_SELECTION.md`)
- Input: one tweet (text) + one attached image per sample.
- Label: single overall sentiment ∈ **{positive, neutral, negative}** (3-class).
  Exact class counts/distribution will be confirmed on load and recorded in
  `results/` — not assumed in advance (literature reports an imbalanced,
  positive-leaning distribution; macro-F1 and per-class metrics are prioritized
  accordingly per the requirements doc).
- If MVSA is unreachable and the Memotion 7k fallback is used instead, the label used
  is Memotion's Task A overall sentiment (also positive/neutral/negative), keeping the
  3-class framing consistent across both dataset options.

## Text pipeline (Model A input, and the text branch of Model C)

```
raw tweet text
  -> URL / @mention normalization (strip URLs; replace @mentions with a generic token)
  -> emoji handling: demojize (emoji -> short text description, e.g. "😊" -> ":smiling_face:")
     so emoji sentiment signal is preserved as text rather than dropped or left as an
     unknown token (the project synopsis explicitly names emojis as part of the signal)
  -> lowercase, whitespace cleanup
  -> tokenization (model-specific: TF-IDF vocabulary for the baseline; WordPiece
     tokenizer for the transformer model)
  -> embedding:
       - Baseline:    TF-IDF vector
       - Transformer: DistilBERT (distilbert-base-uncased) pooled [CLS] representation
```

**Model choice rationale:** a full BERT-base fine-tune on CPU is slow for iterative
experimentation at this dataset's scale; DistilBERT retains ~97% of BERT's language
understanding at ~60% of the size/inference cost (standard, well-documented
distillation result), making it the practical choice for a CPU-only environment. Two
text variants are trained for comparison, per the master prompt's requirement for a
simple baseline *and* a transformer model:
- **A1 — Traditional baseline:** TF-IDF (unigrams+bigrams, capped vocabulary) + Logistic
  Regression (or Linear SVM). Fast, fully CPU-friendly, standard pre-transformer
  approach (per Literature Review Theme 1).
- **A2 — Transformer baseline:** DistilBERT. First attempted as a **frozen encoder +
  small classifier head** (fast — no backprop through the transformer); **lightweight
  fine-tuning** (few epochs, small batch, frozen lower layers if needed for speed) is
  attempted as a second variant if CPU training time proves acceptable, with wall-clock
  time logged honestly either way.
- The better-performing of A1/A2 (by validation macro-F1) is designated "Model A" for
  the headline text-vs-image-vs-multimodal comparison; **both results are reported**,
  not just the winner.

## Image pipeline (Model B input, and the image branch of Model C)

```
raw image
  -> validate/filter (drop missing/corrupt files, document how many were dropped)
  -> resize to 224x224, convert to RGB
  -> normalize (ImageNet mean/std, since pretrained encoders expect this)
  -> embedding:
       - Baseline: small CNN trained from scratch (a few conv+pool blocks)
       - Modern:   ResNet18 pretrained on ImageNet (torchvision), used as a frozen
                   feature extractor (penultimate-layer embedding), classifier head
                   trained on top
```

**Model choice rationale:** Literature Review 2.1 (Dosovitskiy et al., ViT) and 2.2
(Agung et al. 2024) both support this: Vision Transformers need large-scale pretraining
data to beat CNNs and underperform when trained from scratch on modest datasets, while
transfer learning from a pretrained CNN substantially outperforms training from scratch
on a dataset of this size (thousands, not millions, of images). ResNet18 is chosen over
larger CNNs (ResNet50, EfficientNet) specifically for CPU inference speed; MobileNetV3-
Small is the documented fallback if ResNet18 inference proves too slow at this dataset
size. A from-scratch small CNN is trained as the required "CNN/vision baseline" for
comparison against the pretrained "modern image model" (ResNet18).

## Multimodal pipeline (Model C)

```
text embedding (DistilBERT [CLS], 768-dim)   image embedding (ResNet18 penultimate, 512-dim)
                    \                                      /
                     \                                    /
                      -----> feature concatenation (1280-dim) ----->
                                        |
                              MLP classification head
                         (hidden layer + dropout + softmax, 3 classes)
```

**Primary fusion — feature-level (early) concatenation fusion.** Chosen as the primary
method because: (a) it is the most direct way to let a single classifier jointly weigh
text and image evidence per-sample, (b) it is the standard first fusion strategy
documented across both independent surveys reviewed (Gandhi et al. 2023; Das & Singh
2023), and (c) it is cheap to train on CPU (the transformer/CNN encoders stay frozen;
only the small MLP head is trained), keeping iteration fast given this project's
hardware.

**Required ablation — late (decision-level) fusion.** Train Model A and Model B fully
independently, then combine their predicted class probabilities (simple average, and a
validation-tuned weighted average) at decision time. Compared against the feature-
concatenation fusion to answer: *does joint training on combined features beat simply
combining two independently-trained models' opinions?* This directly operationalizes
the required "Text only / Image only / Text + Image" ablation from
`PROJECT_REQUIREMENTS_ANALYSIS.md`, extended with the fusion-strategy comparison the
master build prompt also asks for.

**Explicitly out of scope for the primary build (documented, not silently dropped):**
cross-modal attention / multimodal-transformer fusion (e.g., CMGA-style, Literature
Review 3.4) is the more sophisticated option surveyed in the literature, but it adds
real training cost and architectural complexity; per development rule 13 ("if a
technique cannot realistically run in the environment, select a scientifically
reasonable alternative and document why"), it is attempted only as a stretch
experiment **after** the required A/B/C comparison and ablation are complete and
reported, time/compute permitting — never substituted silently for the required
comparison.

## Explainability (no extra heavy dependencies — implemented directly against
PyTorch/Transformers APIs already in use)

- **Text:** per-token importance via gradient × input (saliency) on the embedding
  layer, and/or direct inspection of DistilBERT's self-attention weights toward the
  [CLS] token — both are standard, lightweight, and don't require adding SHAP/LIME/
  Captum as dependencies (keeping disk footprint down, per this project's hardware
  constraint).
- **Image:** Grad-CAM (Selvaraju et al. 2017, Literature Review 2.3), implemented via
  forward/backward hooks on ResNet18's last convolutional block — no extra library
  needed.
- **Multimodal:** modality-contribution analysis by ablating one branch at inference
  time (zeroing the text or image embedding and observing the prediction/confidence
  shift) — simple, interpretable, and directly answers "how much did each modality
  contribute to this specific prediction," framed as exploratory per
  `RESEARCH_GAP.md` point 4, not a rigorously validated attribution method.

## Evaluation (applies identically to Models A, B, C and the fusion ablation)

Accuracy, Precision, Recall, F1 (macro + weighted), confusion matrix on the held-out
test split; ROC-AUC/PR-AUC computed one-vs-rest across the 3 classes where useful for
visualizing per-class separability. See `PROJECT_REQUIREMENTS_ANALYSIS.md` for the full
evaluation requirements and `REPRODUCIBILITY.md` for split/leakage discipline.

## Non-clinical framing (maintained in every component above)

Every model in this architecture outputs one of the dataset's own sentiment labels
(positive / neutral / negative) — never a clinical/diagnostic label. This is enforced
at the output layer (fixed label set, hardcoded to match `DATASET_SELECTION.md`'s
documented label semantics) and in every UI/report surface (see `ETHICS_AND_PRIVACY.md`).
