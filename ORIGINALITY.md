# Originality

This document tracks what is independently developed in this project versus what is
built on existing libraries, pretrained models, or published research — per the
project's academic integrity commitment (see also the Declaration in
`source_documents/Group-15_Project_Synopsis.pdf`).

## Independently developed

- The overall multimodal pipeline design and its integration (text pipeline + image
  pipeline + fusion + classification head) as implemented in `src/`, written for this
  project rather than copied from a tutorial, GitHub repo, or paper's released code.
- The specific comparison experiment design (Model A vs B vs C, plus the fusion-strategy
  ablation) and evaluation harness in `src/evaluation/`.
- The explainability analysis code (attention/token-importance extraction for text,
  Grad-CAM-style visualization for images) — implemented directly against PyTorch/
  Transformers APIs, not copied from a third-party explainability demo.
- The demo application and dashboard in `app/`.
- All project documentation (this file and its siblings), the literature review's
  synthesis/analysis, the research gap framing, and the final report/paper's writing.

## Built on existing work (used and acknowledged, not claimed as original)

- **Pretrained models:** (final list populated once `PROJECT_ARCHITECTURE.md` is
  finalized — e.g. a HuggingFace `transformers` text encoder, a `torchvision` image
  encoder.) Using a pretrained encoder via transfer learning is standard, disclosed
  practice, not a violation of originality.
- **Libraries:** PyTorch, torchvision, Transformers (Hugging Face), scikit-learn,
  pandas, numpy, matplotlib/seaborn, Streamlit, pytest — all used via their public
  APIs per their respective licenses.
- **Dataset:** sourced from a publicly released academic dataset (see
  `DATASET_SELECTION.md` for exact provenance and citation) — not collected by this
  project, and not presented as such.
- **Research inspiration:** methodological ideas (e.g. fusion strategy types, evaluation
  practices) draw on the papers catalogued in `LITERATURE_REVIEW.md`, each cited there
  with its own title/authors/venue. No paper's code or text is copied verbatim.

## What is explicitly NOT claimed

- Global novelty ("no one has ever done multimodal sentiment analysis before") — this
  project builds on an active research area; `RESEARCH_GAP.md` states the specific,
  bounded gap this project addresses, not a sweeping originality claim.
- Clinical validity or diagnostic capability — see the non-clinical framing maintained
  throughout (`README.md`, `ETHICS_AND_PRIVACY.md`).

## Verification practice

Code in this repository is written and tested incrementally (see `TESTING.md` for what
was actually executed). Where a technique is adapted from a specific published method
(e.g. a particular fusion architecture), the originating paper is cited in code
comments/docstrings at the point of use, not just in this file.
