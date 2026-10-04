# Viva Preparation

Technically correct, student-friendly answers grounded in what this project actually
did. Cross-references point to the source-of-truth document for each topic.

## Project motivation

**Q: Why this project?**
Social media posts often combine text and images, and each can carry sentiment
information the other misses (e.g., a caption that reads neutral but an image that
reads distressed, or vice versa). We wanted to test, with real experiments rather than
assumption, whether combining both modalities actually improves sentiment
classification — and specifically where/why it helps or doesn't. See
`PROJECT_REQUIREMENTS_ANALYSIS.md`.

**Q: Is this a mental-health diagnosis tool?**
No. It classifies sentiment (positive/neutral/negative) — the dataset's own labels.
We explicitly never claim to detect depression, anxiety, or any clinical condition.
Sentiment is used as a *proxy signal* relevant to mental-health-related monitoring
research, exactly as the project synopsis frames it, not a diagnosis. See
`ETHICS_AND_PRIVACY.md`.

## Dataset

**Q: What dataset did you use, and why?**
MVSA-Single — a real, published Twitter dataset of 4,511 (after standard cleaning)
image+text pairs, each labeled positive/neutral/negative by a human annotator (Niu et
al., 2016). We chose it because it's the field's standard benchmark for exactly our
comparison (text vs. image vs. multimodal), it's a real size (thousands, not a toy
set), and it's legitimately available. Full comparison against alternatives (Memotion,
T4SA, CrisisMMD) in `DATASET_SELECTION.md`.

**Q: Did you download the data yourself? Any problems?**
Yes — and we hit a real problem: the dataset's official distribution link was dead
(404) when we went to use it. We found and verified an alternative path: a
HuggingFace community mirror for the images/text (verified by confirming it had
exactly the paper's reported 4,869 pairs), and a published train/dev/test split with
labels from a separate peer-reviewed paper's code release (CLMLF, NAACL 2022
Findings). We documented this honestly rather than hiding the detour — see
`DATASET_SELECTION.md` "Actual acquisition."

**Q: Is your train/test split your own?**
No — we deliberately used an existing published split (from the CLMLF paper) instead
of inventing our own, specifically so our results are comparable to other work using
the same split, not just internally consistent.

**Q: Did you check for data leakage?**
Yes, and we found a small amount: 56 texts and 10 images (out of 4,511) are exact
duplicates appearing in more than one split — caused by Twitter retweets in the
*published* split we used, not something we introduced. We measured and disclosed
this (`REPRODUCIBILITY.md`, `src/data/check_leakage.py`) rather than assuming the
split was clean.

**Q: What does your label mapping mean, concretely?**
0 = positive, 1 = neutral, 2 = negative. We didn't just assume this from the paper —
we spot-checked it by reading actual tweet texts per label and confirming they matched
(e.g., label-2 texts contained words like "#Depressed", "helpless").

## Models and architecture

**Q: Walk me through your pipeline.**
Text: clean (strip URLs, normalize mentions, convert emoji to text) → either TF-IDF
features or a frozen DistilBERT embedding → classifier. Image: resize/normalize →
either a small CNN trained from scratch, or a frozen pretrained ResNet18 embedding →
classifier. Multimodal: concatenate the text and image embeddings → a small MLP
trained jointly. Full diagram and rationale: `PROJECT_ARCHITECTURE.md`.

**Q: What is BERT / DistilBERT, and why did you use it?**
BERT is a transformer language model pretrained on large text corpora to produce
contextual word/sentence representations. DistilBERT is a smaller, distilled version
(~60% of the size, ~97% of the language understanding) — we chose it specifically
because this project runs on CPU only, with no GPU, and DistilBERT is the practical
choice for that constraint while still being a modern transformer.

**Q: What is a CNN, and why ResNet18?**
A Convolutional Neural Network learns spatial filters directly from image data.
ResNet18 is a well-known CNN architecture pretrained on ImageNet; we used it as a
*frozen feature extractor* (its weights aren't updated) because transfer learning
from a model already trained on millions of images works much better than training a
small CNN from scratch on our much smaller dataset — this is both a literature-backed
choice (Agung et al. 2024) and the only practical choice given our CPU-only hardware.

**Q: What is transfer learning, in your own words?**
Reusing a model already trained on a large, different dataset (ImageNet, for images;
a huge text corpus, for DistilBERT) as a starting point or fixed feature extractor,
instead of training from scratch on our much smaller dataset. It gives us
much stronger features than we could learn from ~3,600 training images alone.

**Q: What fusion method did you use? Why not something fancier like cross-modal
attention?**
Our primary method is feature-level (early) fusion: concatenate the text and image
embeddings, train a small classifier on top. We also tested late (decision-level)
fusion: train text and image classifiers separately, then combine their predicted
probabilities. We chose not to implement a more complex cross-modal attention/
transformer fusion as the primary method because (a) it adds real training cost our
CPU-only hardware struggles with, and (b) the literature validates it mainly on
video+audio+text benchmarks (MOSI/MOSEI), not static image+text social posts like
ours — see `LITERATURE_REVIEW.md` 3.4. We documented this as a scope decision, not an
oversight.

## Results

**Q: What were your results?**
Text-only (frozen DistilBERT) got 61.6% accuracy / 54.6% macro-F1. Image-only (frozen
ResNet18) got 52.7% / 42.8% — clearly weaker, since images carry less explicit
sentiment signal than text in tweets. Multimodal fusion got 64.4% / 56.3% — beating
both. Full numbers: `ERROR_ANALYSIS.md`.

**Q: Why macro-F1 and not just accuracy?**
Our dataset is imbalanced: positive is 59.5% of training data, neutral only 10.4%.
A model could get high accuracy just by always predicting "positive" while being
useless on the other two classes. Macro-F1 treats each class equally regardless of
its size, so it exposes that kind of failure — which accuracy would hide.

**Q: Did multimodal fusion always win? Any cases where it didn't?**
No, not unconditionally — and we report this honestly rather than hiding it. When we
broke results down by whether the text and image models *agreed* or *disagreed*:
fusion clearly wins on the 53.3% of posts where they disagree (57.1% vs. 50.0% text-
only), but shows a small *regression* on the 46.7% where they already agree (72.9% vs.
74.8% text-only). Per-class, fusion also underperforms text-only specifically on the
neutral class. See `MODALITY_CONFLICT_ANALYSIS.md` for the full breakdown — this
nuance is the actual answer to our research question, not a blanket "multimodal wins."

**Q: What's overfitting, and did you check for it?**
Overfitting is when a model learns patterns specific to the training data that don't
generalize — showing as much better training performance than validation/test
performance. We used a held-out validation set to select the best checkpoint (by
macro-F1) during training, and report final numbers only on a separate, untouched
test set — standard practice to detect and avoid reporting an overfit result.

**Q: What does "false positive" and "false negative" mean here, and why does it
matter for a mental-health-adjacent project?**
Since this is a 3-class problem, we analyze per-class rather than one binary
positive/negative. The more consequential error type in a mental-health-adjacent
framing is a true negative-sentiment post misclassified as positive (a "missed
concern"); a true positive misclassified as negative is more like a false alarm. We
found non-zero rates of both for every model — which is exactly why we don't claim
this system is reliable enough for any real monitoring/intervention use. See
`ETHICS_AND_PRIVACY.md` §5.

## Limitations and future work

**Q: What are the biggest limitations of your work?**
(1) The dataset has sentiment labels, not mental-health-specific labels — we use
sentiment as a documented proxy, not a diagnosis. (2) It's English-language,
2015-16-era Twitter data — doesn't generalize to other languages/platforms/eras
without further work. (3) A small amount of duplicate-content leakage exists in the
published split we used. (4) Our from-scratch CNN baseline was constrained by this
machine's CPU-only hardware (see below).

**Q: Why does your from-scratch CNN (Model B1) use fewer epochs / lower resolution
than planned, and what did it actually score?**
We hit a real, measured compute constraint: training even a small CNN from scratch at
full 224×224 resolution on this CPU-only (no GPU), 8GB-RAM machine was impractically
slow — one run was killed by the OS for memory pressure (which we found and fixed: a
dataset-loading bug), another exceeded our background-process time budget without
finishing 3 epochs. We responded by reducing image resolution (96×96) and training for
3 epochs, documenting exactly why in `REPRODUCIBILITY.md`. The completed result:
47.6% accuracy / 42.9% macro-F1 — lower accuracy than the frozen-ResNet18 model (B2,
52.7%), but an almost-identical macro-F1 (42.9% vs. 42.8%). We also observed real
training instability (validation macro-F1 dropped between epochs 1 and 2 before
recovering by epoch 3), which we report as-is rather than hide. Overall this supports
our literature-grounded design choice to prefer the frozen pretrained ResNet18 (B2)
for the main comparison: it reaches comparable or better results with far less
training instability and compute cost.

**Q: What would you do differently with more time/compute (e.g., a GPU)?**
Fine-tune DistilBERT and ResNet18 end-to-end rather than using them frozen; try a
proper cross-modal attention fusion architecture; run cross-validation / multiple
seeds for confidence intervals on the reported metrics; extend to a larger or
multilingual dataset.

**Q: What's your actual research contribution, stated carefully?**
A real, same-dataset, same-metrics comparison of text-only, image-only, and two
multimodal fusion strategies on a substantially-sized real social-media dataset — with
an honest, non-overclaimed finding that fusion's benefit concentrates specifically on
cases of text/image disagreement, plus a documented account of what this comparison
actually costs to run under realistic CPU-only resource constraints. We do not claim
to be first in this space (the literature is active and well-populated — see
`RESEARCH_GAP.md`), and we do not claim clinical validity.
