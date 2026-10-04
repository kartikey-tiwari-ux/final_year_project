# AI-Driven Multimodal Sentiment Analysis for Mental Health Monitoring on Social Media: A Resource-Constrained Comparative Study

**Authors:** Kartikey Tiwari, Akshat Sharma, Karan Sharma, Lalit Sharma
**Affiliation:** Department of Computer Science and Engineering, Ajay Kumar Garg Engineering College, Ghaziabad (Dr. A.P.J. Abdul Kalam Technical University, Lucknow)
**Supervisor:** Dr. Avdhesh Gupta

## Abstract

Social media posts often combine text and images, but most sentiment-analysis systems
for mental-health-related monitoring rely on text alone, potentially losing
complementary emotional signal carried by attached images. This paper reports a
controlled, same-dataset, same-metrics comparison of text-only, image-only, and
multimodal (text+image) sentiment classification on MVSA-Single, a real, publicly
documented Twitter image-text sentiment dataset (4,511 labeled pairs after standard
cleaning; 3-class: positive/neutral/negative). We evaluate a traditional TF-IDF +
Logistic Regression text baseline, a frozen-DistilBERT text model, a CNN trained from
scratch and a frozen-ResNet18 image model, and two multimodal fusion strategies
(feature-concatenation and late/decision-level fusion). The best multimodal model
(feature-concatenation fusion) achieves 64.4% test accuracy and 56.3% macro-F1,
outperforming the best unimodal model (frozen DistilBERT: 61.6% accuracy, 54.6%
macro-F1) and the image-only model (52.7% accuracy, 42.8% macro-F1). A modality-
conflict analysis shows the gain concentrates specifically on the 53.3% of test posts
where text and image models disagree (fusion: 57.1% accuracy on this subset vs. 50.0%
for text-only and 33.3% for image-only), with a small regression on posts where
modalities already agree. We report this work entirely from real experiments run on
constrained consumer CPU hardware (no GPU, 8GB RAM), documenting the resource
constraints encountered and the engineering decisions made in response, as a
contribution to resource-constrained reproducibility in this research area. Consistent
with the non-clinical scope of this study, all outputs are the dataset's own sentiment
labels, not diagnostic claims.

**Keywords:** Multimodal Sentiment Analysis, Mental Health Monitoring, Social Media
Analytics, Natural Language Processing, Deep Learning, DistilBERT, ResNet18, Feature
Fusion, MVSA, Resource-Constrained Machine Learning

## 1. Introduction

See `PROJECT_REQUIREMENTS_ANALYSIS.md` for the full background/motivation, derived
from the originating project synopsis. In short: text-only or image-only sentiment
analysis of social media posts each miss information the other modality carries
(synopsis Ch. 1.1.2); this project investigates whether and when combining them
improves sentiment-based classification, framed explicitly as mental-health-*related*
sentiment monitoring support — not clinical diagnosis (see §8, Ethics).

## 2. Related Work

Full annotated review in `LITERATURE_REVIEW.md` (14 verified sources). Key threads:
text-based mental-health classification increasingly uses domain-adapted transformers
(MentalBERT — Ji et al. 2022) but a 2024 systematic review (Cao et al.) documents
widespread representativeness and evaluation-methodology issues across this
literature; image-emotion recognition benefits substantially from transfer learning
over from-scratch training on modest datasets (Agung et al. 2024); multimodal fusion
has a mature taxonomy (Gandhi et al. 2023; Das & Singh 2023) but — per our research
gap analysis — lacks a controlled, adequately-sized, same-metrics text-vs-image-vs-
multimodal comparison specifically on real social-media data (the closest located
prior work, Vasanthi & Viswanatham 2026, evaluates its social-media condition on only
100 pairs).

## 3. Research Gap and Questions

Full analysis in `RESEARCH_GAP.md`. Primary research question (from
`PROJECT_REQUIREMENTS_ANALYSIS.md`): *Can an AI-driven multimodal framework combining
textual and visual information improve sentiment-based mental health monitoring on
social media compared with unimodal approaches?*

## 4. Dataset

**MVSA-Single** (Niu, Zhu, Pang & El Saddik, MMM 2016): real Twitter image-text pairs,
human-annotated 3-class sentiment (positive/neutral/negative). This project uses the
published train/dev/test split from Li, Xu, Zhu & Zhao's CLMLF (Findings of ACL:
NAACL 2022) — 3,611/450/450 samples — for comparability with that and other work using
the same fold, rather than an arbitrary self-computed split. Images and text were
obtained via a HuggingFace community mirror after the original OneDrive distribution
proved dead (404) at acquisition time; full provenance chain, label-semantics
verification (empirically confirmed 0=positive/1=neutral/2=negative by inspecting
sample texts per label), and licensing caveats are documented in
`DATASET_SELECTION.md`. Class distribution (train): positive 2,147 (59.5%), negative
1,088 (30.1%), neutral 376 (10.4%) — a real, measured imbalance that motivates
macro-F1 as the primary metric. A duplicate-hash leakage audit found a small amount of
cross-split duplication inherited from the published split itself (56/4,511 texts,
10/4,511 images, from Twitter retweets) — disclosed in `REPRODUCIBILITY.md` as a
limitation of using a third-party split, assessed as unlikely to materially affect
results at this scale.

## 5. Methodology

Full architecture rationale in `PROJECT_ARCHITECTURE.md`. All models share the same
train/val/test split and evaluation code (`src/evaluation/metrics.py`).

**Text pipeline:** cleaning (URL/mention normalization, emoji→text demojizing,
lowercasing) → (A1) TF-IDF (1-2 grams, 10k features) + class-weighted Logistic
Regression, or (A2) frozen DistilBERT (`distilbert-base-uncased`) mean-pooled
embeddings + class-weighted Logistic Regression.

**Image pipeline:** resize/normalize → (B1) a small 4-layer CNN trained from scratch,
or (B2) frozen ImageNet-pretrained ResNet18 penultimate-layer embeddings + class-
weighted Logistic Regression.

**Multimodal fusion (C):** primary — feature-level concatenation of the A2 text
embedding (768-dim) and B2 image embedding (512-dim) → a 2-hidden-layer MLP with
dropout, trained jointly with class-weighted cross-entropy loss, selecting the best
checkpoint by validation macro-F1. Ablation — late (decision-level) fusion: independent
A2/B2 classifiers' predicted probabilities combined by simple averaging and by a
validation-tuned weighted average.

**Why frozen pretrained encoders, not full fine-tuning:** justified directly by this
project's measured hardware constraints (CPU-only, 8GB RAM) and by the literature
(§2) — transfer learning with a frozen/lightly-adapted encoder is both the
literature-supported choice for modest dataset sizes and the only practically
tractable choice on this hardware; see `REPRODUCIBILITY.md` for measured per-sample
embedding-extraction timing (DistilBERT: 49.5ms/sample; ResNet18: 536ms/sample on
first pass, dropping to a few ms/sample on a cache-warmed second pass).

## 6. Experimental Setup

Fixed seed (42) throughout (`src/utils/seed.py`). Evaluation: accuracy, precision/
recall/F1 (macro and weighted), confusion matrix — macro-F1 prioritized given class
imbalance (§4). All numbers in §7 are read directly from `results/*.json`, generated
by the scripts under `src/text_model/`, `src/image_model/`, and `src/multimodal/`.

## 7. Results

| Model | Accuracy | Macro-F1 | Weighted-F1 |
|---|---|---|---|
| A1 — TF-IDF + LogReg | 60.9% | 52.0% | 61.9% |
| A2 — Frozen DistilBERT + LogReg | 61.6% | 54.6% | 63.5% |
| B1 — CNN from scratch | 47.6% | 42.9% | 49.6% |
| B2 — Frozen ResNet18 + LogReg | 52.7% | 42.8% | 54.3% |
| **C — Concat Fusion** | **64.4%** | **56.3%** | **65.4%** |
| C — Late Fusion (tuned) | 64.0% | 56.4% | 65.6% |

Full confusion matrices and per-class breakdown: `ERROR_ANALYSIS.md`. Notably, B1
(trained from scratch, at a reduced 96×96 resolution and 3 epochs due to measured
CPU-only training-time constraints — see §5 and `REPRODUCIBILITY.md`) scores close to
B2 on macro-F1 (42.9% vs. 42.8%) despite a clearly lower accuracy (47.6% vs. 52.7%),
an interesting nuance: B1's training instability (its validation macro-F1 dropped
between epochs 1 and 2 before recovering by epoch 3 — a real, reported instability,
not smoothed over) suggests its best-epoch macro-F1 may partly reflect noisy,
under-converged training rather than a robust feature-learning advantage. This result
is consistent with, not contradictory to, the literature-grounded expectation
(§2, §5) that frozen transfer-learned features are the more reliable choice under
this project's real hardware constraints — B2 achieves comparable or better results
with a fraction of the training instability and compute.

**Model comparison answers the primary research question affirmatively but with
nuance, not a blanket win:** multimodal fusion outperforms every unimodal model on
every aggregate metric, but the improvement is not uniform across classes or subsets —
see §7.1.

### 7.1 Modality Conflict Analysis (full detail: `MODALITY_CONFLICT_ANALYSIS.md`)

Text and image models disagree on 53.3% of test posts (240/450). On exactly this
subset, fusion clearly outperforms both unimodal models (57.1% vs. 50.0% text-only,
33.3% image-only) — the clearest evidence that fusion helps specifically by resolving
genuine cross-modal disagreement, not just by averaging noise. On the 46.7% of posts
where modalities already agree, fusion shows a small regression versus text-only
(72.9% vs. 74.8%). Per-class, fusion helps substantially on the majority (positive)
class but underperforms text-only specifically on the minority neutral class (42.6%
vs. 48.9%) — attributed to neutral's severe data scarcity (10.4% of training data)
leaving little signal for the fusion head to learn a reliable joint boundary.

### 7.2 Ablation: Fusion Strategy

Feature-concatenation (early/joint) fusion and validation-tuned late (decision-level)
fusion perform almost identically (64.4%/56.3% vs. 64.0%/56.4% accuracy/macro-F1) —
neither strategy dominates the other on this dataset/scale, both clearly beat
unimodal. The late-fusion weight search independently converges on w_text=0.80,
corroborating text's stronger standalone performance (§7).

## 8. Discussion, Ethics, and Limitations

**Non-clinical scope:** all model outputs are the dataset's own sentiment labels
(positive/neutral/negative) — never relabeled as clinical categories. See
`ETHICS_AND_PRIVACY.md` for the full discussion of privacy, consent, licensing, bias,
and misuse-risk considerations specific to this dataset and application framing.

**Key limitations (not minimized):** (1) MVSA-Single's labels are general sentiment,
not mental-health-specific — a documented, unavoidable scoping limitation given no
open, legitimately-licensed multimodal mental-health dataset was found (§4,
`DATASET_SELECTION.md`); (2) English-language, 2015-16-era Twitter content — results
do not generalize to other languages, platforms, or eras without further validation;
(3) a small (~1-2%) cross-split duplication exists in the published split used;
(4) CPU-only compute constraints limited the from-scratch CNN baseline's training
budget and resolution — itself a finding relevant to resource-constrained
reproducibility (§5, `REPRODUCIBILITY.md`), not just a caveat.

## 9. Conclusion and Future Work

Within a real, resource-constrained CPU-only environment, this project provides
genuine (not assumed) evidence that multimodal fusion improves sentiment
classification on real social-media image-text pairs — concentrated specifically on
posts where modalities disagree, with measurable trade-offs elsewhere. Future work
(explicitly out of scope here per `PROJECT_REQUIREMENTS_ANALYSIS.md`): multilingual
extension, cross-modal attention fusion at scale, and validation against a genuinely
mental-health-labeled multimodal dataset if/when one becomes openly available.

## References

See `REFERENCES.md` / `research_paper/references.bib` for the full citation list.

## Acknowledgements

Dataset: Niu, Zhu, Pang & El Saddik (2016); split/labels: Li, Xu, Zhu & Zhao (2022,
CLMLF). Pretrained models: DistilBERT and ResNet18 via Hugging Face Transformers and
torchvision respectively.
