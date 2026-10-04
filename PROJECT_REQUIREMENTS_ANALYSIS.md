# Project Requirements Analysis

Source: `source_documents/Group-15_Project_Synopsis.pdf` (primary specification), `source_documents/Task2_Literature_Review_and_Objective.docx`, `source_documents/Claude_Code_Max_Final_Year_Project_Prompt.pdf` (build instructions).

## Project Title
AI-Driven Multimodal Sentiment Analysis for Mental Health Monitoring on Social Media

## Team
- Kartikey Tiwari (2300271540063) — Literature and dataset preparation
- Akshat Sharma (2300271540018) — NLP and text modelling
- Karan Sharma (2300271540062) — Image processing, multimodal fusion, evaluation
- Lalit Sharma (2300271540066) — Documentation
- Supervisor: Dr. Avdhesh Gupta, Dept. of CSE, Ajay Kumar Garg Engineering College, Ghaziabad (AKTU)
- Session: 2023–2027

## Problem Statement
Existing sentiment and mental-health analysis systems rely primarily on text from social media, losing emotional/contextual information carried in images, memes, and other visual content. The project develops an AI-driven multimodal framework that jointly analyses text and images for sentiment-based mental health monitoring, and compares this multimodal approach against individual text-only and image-only approaches.

## Background / Motivation
- A social media post may combine text and image to express an emotional state; a text-only or image-only system each lose part of that signal.
- NLP provides sentiment/emotion signals from text; Computer Vision provides complementary signals from images; multimodal AI combines both to address the limitation of unimodal analysis.

## Primary Research Question
Can an AI-driven multimodal framework combining textual and visual information improve sentiment-based mental health monitoring on social media compared with unimodal approaches?

## Secondary Research Questions
1. How effectively can textual information alone identify mental-health-related sentiment in social media posts?
2. What additional information can visual content provide for such classification?
3. Does combining textual and visual features improve classification performance compared with individual modalities?

## Research Objectives
1. Study existing AI-based approaches for social-media sentiment and mental-health analysis.
2. Develop suitable preprocessing methods for textual and visual data.
3. Develop a multimodal framework combining text and image representations for classification.
4. Compare multimodal performance against text-only and image-only models.
5. Evaluate using accuracy, precision, recall, and F1-score (macro/weighted), confusion matrices, and (where applicable) ROC/PR-AUC.

## Project Scope
- Computational analysis of textual and visual information from social media content: preprocessing, feature extraction, multimodal feature fusion, classification, performance evaluation.
- Focus: identifying **mental-health-related sentiment or patterns**, NOT clinical diagnosis.
- The system is a research/monitoring support tool, not a replacement for qualified mental-health professionals. This boundary must be maintained everywhere: code, UI, README, paper, report, presentation, model output labels.
- Out of scope: real clinical validation, deployment to real users, scraping private/live social media accounts.

## Literature Themes (from synopsis Ch. 2)
1. **Text-Based Mental Health Analysis** — NLP/ML on social posts; traditional (word representations, statistical features) vs. modern (deep learning, transformers e.g. BERT). Limitation: ignores visual content.
2. **Image-Based Emotion Analysis** — CNN/deep learning visual feature extraction from images/memes. Limitation: ignores textual meaning.
3. **Multimodal Sentiment and Mental Health Analysis** — feature fusion, attention mechanisms, multimodal learning architectures combining text+image. Challenge: effectively combining modalities, handling conflicting signals between modalities.

## Research Gap (synopsis Ch. 2.2, to be expanded with real literature in `LITERATURE_REVIEW.md` / `RESEARCH_GAP.md`)
Existing research shows the value of text-based and multimodal approaches, but effective integration of text+image remains challenging (complementary vs conflicting emotional cues), and there is a need for consistent, rigorous comparison of multimodal vs unimodal models on the same data/metrics. This project addresses that gap with a practical, evaluated AI framework.

## Proposed Framework (synopsis Ch. 4)
- **Text pipeline:** raw text → cleaning → tokenization → NLP representation → text embedding.
- **Image pipeline:** image → resize/normalize → CNN or Vision Transformer → visual embedding.
- **Multimodal pipeline:** text embedding + image embedding → fusion → classification head → output.
- Research approach: experimental and comparative (dataset prep, preprocessing, feature extraction, model development, multimodal fusion, performance evaluation), with unimodal baselines for meaningful comparison.

## Required Modalities
Text (posts/captions, possibly emoji) and Image (photos/memes accompanying posts).

## Required Comparison (mandatory, core experiment)
- **Model A — Text only**
- **Model B — Image only**
- **Model C — Multimodal (text + image)**
The multimodal model is **not assumed** to win; if it underperforms, that result must be reported and investigated, not hidden or adjusted.

## Dataset Requirements
Must be: authentic (real social-media-sourced or academically released), have legitimate provenance, contain paired text+image samples with usable labels, sufficient sample count, acceptable licensing for research/educational use, ethically sourced (no scraping private accounts, no PII exposure), and computationally feasible to download/process on this project's hardware (CPU-only, limited disk — see Reproducibility notes). Labels must be used exactly as defined by the dataset's authors; no relabeling as clinical diagnoses.

## Evaluation Metrics
Accuracy, Precision, Recall, F1-score, Macro-F1, Weighted-F1, Confusion Matrix; ROC-AUC/PR-AUC where applicable (binary/one-vs-rest). Macro-F1 and per-class metrics prioritized under class imbalance.

## Expected Research Contribution
1. Systematic comparison of text-only, image-only, and multimodal approaches on one dataset with common metrics.
2. Evidence on whether/when multimodal fusion improves sentiment-based mental-health-relevant classification.
3. Analysis of fusion strategy, error patterns, and (where feasible) explainability of predictions.
Contribution is bounded by what the actual experiments show — no claims of novelty beyond the evidence produced.

## SDG Alignment (synopsis Ch. 3 — maintain without exaggeration)
- **Primary:** SDG 3 — Good Health and Well-being (AI-based computational support for mental-health-related monitoring/research, not diagnosis).
- **Secondary:** SDG 9 — Industry, Innovation and Infrastructure (applying modern NLP/CV/multimodal learning to a socially relevant problem).
- **Secondary:** SDG 10 — Reduced Inequalities (scalable, accessible digital analysis — caveated: must avoid bias from language/culture/digital-access differences).

## Expected Challenges (synopsis Ch. 5.4 + practical constraints of this build)
- Limited availability of suitable labelled multimodal (text+image) datasets for the mental-health domain specifically.
- Variation/noise in real social media content; class imbalance.
- Computational constraints: this machine has **no GPU** and (at project start) **critically limited disk space** — addressed by using lightweight/efficient pretrained models (e.g., DistilBERT-scale text encoders, small CNN/MobileNet-scale or frozen pretrained image encoders), a modest dataset subset, and CPU-feasible training, documented honestly in `REPRODUCIBILITY.md`.
- Conflicting text/image emotional signals complicating fusion.
- Time constraints limiting exhaustive experimentation.

## Timeline (from synopsis, academic milestones — informational; actual build proceeds continuously)
| Period | Milestone |
|---|---|
| Jul–Aug 2026 | Literature review finalized; synopsis submitted |
| Sept 2026 | Detailed system/algorithm design; tool & dataset setup |
| Oct 2026 | Implementation/prototyping (Phase 1) |
| Nov 2026 | Implementation completion; experimentation & validation |
| Dec 2026 | Results analysis; manuscript drafting |
| Jan 2027 | Manuscript/patent draft submission |
| Feb–Mar 2027 | Peer-review response, revisions, publication, final report |

## Expected Publication / Research Outcome
Technical research report + academic project demonstration; if results are sufficient, a manuscript prepared for a suitable conference/journal (synopsis mentions IEEE ICCCNT as an example target venue), with proper acknowledgement of datasets/papers used. No publication claim is made by this build itself — only the groundwork (paper draft in `research_paper/`) is produced here.

---

## MANDATORY REQUIREMENTS
- [ ] Text-only, Image-only, and Multimodal models all implemented and compared on the same dataset/splits/metrics.
- [ ] Authentic, legitimately-sourced, properly licensed multimodal dataset (no synthetic/fabricated posts presented as real).
- [ ] Dataset provenance, label semantics, and licensing documented (`DATASET_SELECTION.md`).
- [ ] Ethics & privacy documented (`ETHICS_AND_PRIVACY.md`): no PII, no private-account scraping.
- [ ] Data leakage prevention (train/val/test split discipline, no duplicate leakage, no test-set tuning).
- [ ] Real literature review with verifiable citations (`LITERATURE_REVIEW.md`) — no fabricated papers.
- [ ] Defensible, non-overclaimed research gap (`RESEARCH_GAP.md`).
- [ ] Evaluation beyond accuracy: precision/recall/F1 (macro+weighted), confusion matrix.
- [ ] Explicit, maintained non-clinical framing everywhere (code, UI, docs, paper, report, slides).
- [ ] Reproducibility artifacts: requirements file, config files, seeds, experiment logs (`REPRODUCIBILITY.md`).
- [ ] Working demonstration app (text/image/multimodal prediction + confidence + comparison + non-diagnostic disclaimer).
- [ ] Tests for preprocessing/pipelines, actually executed (`TESTING.md`).
- [ ] Originality documentation (`ORIGINALITY.md`) — no verbatim copying of existing repos/notebooks/papers.
- [ ] Final audit (`FINAL_PROJECT_AUDIT.md`) verifying all of the above against what was *actually done*, not assumed.

## OPTIONAL / EXTENSIBLE FEATURES (attempt if time/compute allow; explicitly scoped down given CPU-only, low-disk environment)
- Ablation study across fusion strategies (early/late/attention-based), beyond the minimum text/image/text+image ablation.
- Explainability: token-importance/attention for text, Grad-CAM/saliency for images, cross-modal contribution analysis.
- Modality-conflict analysis (posts where text and image suggest different emotional states), if the chosen dataset supports it.
- Statistical validity extras: cross-validation, repeated runs with variance, significance testing — only where the dataset size and compute budget make it meaningful.
- Full research paper (`research_paper/`), complete final-year report, presentation deck content, viva-prep Q&A — produced after real experiments exist, populated only with real results.
- Multilingual analysis, conversational support, personalized recommendations (explicitly future-work extensions per the synopsis, not in scope for this build).
