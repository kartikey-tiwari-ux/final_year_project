# Final Year Project Report

**AI-Driven Multimodal Sentiment Analysis for Mental Health Monitoring on Social Media**

Group 15 — Kartikey Tiwari (2300271540063), Akshat Sharma (2300271540018), Karan
Sharma (2300271540062), Lalit Sharma (2300271540066)
Department of Computer Science and Engineering, Ajay Kumar Garg Engineering College,
Ghaziabad, affiliated to Dr. A.P.J. Abdul Kalam Technical University, Lucknow.
Supervisor: Dr. Avdhesh Gupta. Session 2023–2027.

This report consolidates the project's work. Each section below is intentionally
concise and points to the detailed, source-of-truth document in the project root for
full depth — avoiding duplicating the same content twice and risking it drifting out
of sync.

## Abstract
See `research_paper/paper.md` §Abstract for the full abstract — reproduced in summary:
this project builds and compares text-only, image-only, and multimodal (text+image)
sentiment classifiers on a real, publicly documented Twitter dataset (MVSA-Single),
finding that multimodal fusion improves classification (64.4% accuracy / 56.3%
macro-F1 vs. 61.6% / 54.6% for the best unimodal model), with the benefit concentrated
specifically on posts where text and image signals disagree. All work was conducted
under real CPU-only hardware constraints, documented throughout.

## 1. Introduction, Motivation, Problem Statement, Objectives
Full detail: `PROJECT_REQUIREMENTS_ANALYSIS.md`. Summary: social media posts combine
text and images, each carrying sentiment information the other may miss; this project
investigates, with real experiments, whether and when combining both improves
sentiment-based classification relevant to mental-health monitoring research — framed
explicitly as non-diagnostic throughout (see §10).

## 2. Literature Review
Full detail: `LITERATURE_REVIEW.md` (14 real, individually verified sources across
text-based mental-health analysis, image-based emotion recognition, and multimodal
sentiment fusion, each with methodology/dataset/results/limitations/relevance).

## 3. Research Gap
Full detail: `RESEARCH_GAP.md`. Summary: existing literature establishes strong
unimodal methods and a mature multimodal fusion taxonomy, but no located prior work
performs a controlled, same-dataset, same-metrics comparison of text-only vs.
image-only vs. multimodal classification on *adequately-sized real social-media* data
— the closest comparable study evaluates its social-media condition on only 100
pairs. This project closes that specific, bounded gap.

## 4. Existing System vs. Proposed System
**Existing:** most sentiment/mental-health-monitoring research tools rely on text
alone (per Literature Review Theme 1), losing visual-content signal.
**Proposed:** a framework that processes text and image through separate pipelines,
fuses their learned representations, and is evaluated against both unimodal
alternatives on identical data/metrics — so the value of fusion is measured, not
assumed.

## 5. Requirements
Full detail: `PROJECT_REQUIREMENTS_ANALYSIS.md` (mandatory requirements vs.
optional/extensible features, explicitly scoped for this project's real compute
environment).

## 6. System Architecture and Methodology
Full detail: `PROJECT_ARCHITECTURE.md`. Summary: Text pipeline (cleaning → TF-IDF or
frozen DistilBERT embedding → classifier); Image pipeline (resize/normalize → small
from-scratch CNN or frozen ResNet18 embedding → classifier); Multimodal pipeline
(concatenated embeddings → MLP classifier, with late-fusion as a comparison
ablation). Every architectural choice is justified against the literature (§2) and
this project's measured hardware constraints (CPU-only, 8GB RAM — see §9).

## 7. Dataset
Full detail: `DATASET_SELECTION.md`, `ETHICS_AND_PRIVACY.md`. Summary: MVSA-Single
(Niu et al., 2016), 4,511 labeled Twitter image-text pairs after standard cleaning,
3-class sentiment (positive 59.5% / negative 30.1% / neutral 10.4% of training data).
Published CLMLF train/dev/test split used for comparability. Real acquisition
obstacle encountered and resolved: the official distribution link was dead; an
alternative, verified source was used instead and is fully documented, including the
provenance chain and its limitations.

## 8. Preprocessing
Text: URL/mention normalization, emoji→text demojizing, lowercasing
(`src/preprocessing/text_cleaning.py`, unit-tested). Images: resize to 224×224,
ImageNet normalization for the pretrained branch, missing/corrupt file filtering with
counts logged, not silently dropped (`src/preprocessing/image_preprocessing.py`).

## 9. Implementation
Full detail: `REPRODUCIBILITY.md`. All code under `src/`; every model trainable via a
single script (`python -m src.<module>.<script>`), each saving real metrics to
`results/*.json`. Real engineering obstacles encountered and how they were resolved
(all disclosed, not hidden): (a) a dataset-loading bug caused a training process to be
killed for memory pressure on this machine's 8GB RAM — found (eager image preloading)
and fixed (lazy per-batch loading); (b) CPU-only conv-layer training proved too slow
at full image resolution for a from-scratch CNN — resolved by caching decoded pixels
once and reducing that specific baseline's training resolution, with the reasoning
documented; (c) a quick re-implementation of the fusion model's training loop was
found to produce inconsistent results from the actual trained model — fixed by
sharing one training function between both call sites rather than letting two copies
drift apart.

## 10. Experiments and Results
Full detail: `research_paper/paper.md` §7, `ERROR_ANALYSIS.md`,
`MODALITY_CONFLICT_ANALYSIS.md`. Headline:

| Model | Accuracy | Macro-F1 |
|---|---|---|
| A1 — TF-IDF + LogReg | 60.9% | 52.0% |
| A2 — Frozen DistilBERT + LogReg | 61.6% | 54.6% |
| B1 — CNN from scratch (96×96, 3 epochs — see §9) | 47.6% | 42.9% |
| B2 — Frozen ResNet18 + LogReg | 52.7% | 42.8% |
| **C — Concat Fusion** | **64.4%** | **56.3%** |
| C — Late Fusion (tuned) | 64.0% | 56.4% |

Multimodal fusion beats every unimodal model on every aggregate metric. Breaking this
down further (`MODALITY_CONFLICT_ANALYSIS.md`): the gain concentrates on the 53.3% of
test posts where text and image models disagree (fusion 57.1% vs. text 50.0%, image
33.3% on that subset); on the 46.7% where they agree, fusion shows a small (1.9pp)
regression versus text-only. Per-class, fusion helps substantially on the majority
(positive) class but underperforms text-only on the minority neutral class.

## 11. Testing
Full detail: `TESTING.md`. 32 automated tests across 6 modules (seed reproducibility,
text cleaning, data splitting, evaluation metrics, dataset loading, image
preprocessing), all actually executed and passing — including one real bug the tests
caught and fixed (a missing `labels=` argument in scikit-learn's
`classification_report` that crashed on class-imbalanced batches).

## 12. Explainability
Full detail: `PROJECT_ARCHITECTURE.md` (Explainability section),
`results/explainability_examples/`. Text: gradient×input token saliency. Image:
Grad-CAM via forward/backward hooks. Both implemented directly against PyTorch/
Transformers APIs with no extra heavy dependency, run on real test examples, with
limitations documented (e.g., saliency scores can be disproportionately pulled toward
punctuation/number tokens — an observed, disclosed limitation of this lightweight
method, not glossed over).

## 13. Application
`app/streamlit_app.py` — a demo page (text/image input → text-only, image-only, and
fusion predictions with confidence and a modality-comparison table) and a research
dashboard (model comparison table, confusion matrices, class distribution), both
carrying a persistent, explicit non-diagnostic disclaimer.

## 14. Ethics and Societal Impact
Full detail: `ETHICS_AND_PRIVACY.md`. Covers: privacy (public-data-only, no
re-identification attempts, no raw-data redistribution), consent (disclosed
limitation inherent to public-social-media research datasets), licensing (MVSA's
cite-and-contact convention, disclosed rather than assumed to be formally open),
bias/representativeness (English/Twitter/2015-16-era skew), false-positive/
false-negative framing specific to a mental-health-adjacent context, misuse/stigma
risk, and responsible-deployment boundaries (research/coursework artifact only, never
deployed against real identifiable individuals).

## 15. SDG Alignment
Per the original synopsis (`source_documents/Group-15_Project_Synopsis.pdf` Ch. 3),
maintained without exaggeration: **Primary — SDG 3** (Good Health and Well-being: AI
support for mental-health-related monitoring *research*, not diagnosis). **Secondary
— SDG 9** (Industry, Innovation and Infrastructure: applying modern NLP/CV/multimodal
techniques to a socially relevant problem). **Secondary — SDG 10** (Reduced
Inequalities: the premise of scalable digital analysis, caveated by this project's
own finding of English/Twitter-specific bias that would need addressing before any
broader claim of equitable accessibility).

## 16. Limitations
1. Sentiment labels, not mental-health-specific labels (no open, legitimately
   licensed multimodal mental-health dataset was found — a genuine, disclosed gap in
   available resources, not an oversight).
2. English-language, Twitter-only, 2015-16-era data — no claim of generalization
   beyond this.
3. A small (~1–2%) cross-split content duplication exists in the published split
   used, inherited from Twitter retweets, not introduced by this project.
4. CPU-only hardware constrained the from-scratch CNN baseline's training budget and
   resolution — itself documented as a relevant finding for resource-constrained
   reproducibility, not just an apology.
5. Only one random seed's worth of results reported for the neural models (fusion,
   CNN) due to time/compute budget — variance across seeds not characterized.

## 17. Future Scope
Multilingual extension; cross-modal attention/transformer fusion at full scale (GPU
available); validation against a genuinely mental-health-labeled multimodal dataset
if one becomes openly available; multi-seed variance/statistical-significance
testing; conversational support and personalized-recommendation extensions (as named
as future directions in the original synopsis, explicitly out of this project's
scope).

## 18. Conclusion
This project set out to answer, with real evidence rather than assumption, whether
combining text and image improves sentiment-based classification relevant to
mental-health monitoring research. The answer is a qualified yes: multimodal fusion
measurably outperforms either modality alone overall, with that benefit concentrated
specifically on posts where the two modalities disagree — a nuanced, defensible
finding produced under genuine resource constraints, documented honestly throughout,
consistent with this project's non-clinical scope and academic-integrity commitments
(`ORIGINALITY.md`).

## References
See `REFERENCES.md`.
