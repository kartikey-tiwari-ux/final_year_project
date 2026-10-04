# AI-Driven Multimodal Sentiment Analysis for Mental Health Monitoring on Social Media

Final Year B.Tech Project (CSE) — Group 15, Ajay Kumar Garg Engineering College, Ghaziabad (AKTU), Session 2023–2027.

> **This is a research and monitoring-support tool, not a clinical diagnostic system.**
> It classifies sentiment/emotion-related patterns in social media text and images. It
> does **not** diagnose depression, anxiety, or any mental illness, and is not a
> substitute for a qualified mental-health professional. See `ETHICS_AND_PRIVACY.md`.

## What this project does

Compares three approaches to sentiment/emotion classification on social media posts
that contain both text and an image:

- **Model A — Text only**
- **Model B — Image only**
- **Model C — Multimodal (text + image fusion)**

The central research question: *does combining textual and visual information improve
sentiment-based classification compared with either modality alone, and under what
conditions?* See `PROJECT_REQUIREMENTS_ANALYSIS.md` for the full research questions and
objectives, and `RESEARCH_GAP.md` / `LITERATURE_REVIEW.md` for the grounding.

## Project documentation map

| Document | Contents |
|---|---|
| `PROJECT_REQUIREMENTS_ANALYSIS.md` | Extracted requirements from the project synopsis |
| `LITERATURE_REVIEW.md` | Real, verified papers reviewed across text/image/multimodal themes |
| `RESEARCH_GAP.md` | What's known, what's missing, what this project investigates |
| `DATASET_SELECTION.md` | Candidate datasets evaluated, final choice, and justification |
| `ETHICS_AND_PRIVACY.md` | Privacy, consent, licensing, bias, misuse considerations |
| `PROJECT_ARCHITECTURE.md` | Text/image/multimodal pipeline design |
| `REPRODUCIBILITY.md` | Environment, seeds, configs, how to reproduce results |
| `ORIGINALITY.md` | What's original vs. built on existing libraries/pretrained models |
| `TESTING.md` | What's tested and the results of actually running the tests |
| `FINAL_PROJECT_AUDIT.md` | Checklist verifying claims against what was actually done |

## Repository structure

```
project/
├── src/                  # Source code
│   ├── data/              # Dataset loading
│   ├── preprocessing/     # Text cleaning, image preprocessing
│   ├── text_model/        # Model A
│   ├── image_model/       # Model B
│   ├── multimodal/        # Model C (fusion)
│   ├── evaluation/        # Metrics, confusion matrices
│   ├── explainability/    # Attention/Grad-CAM style analysis
│   └── utils/
├── data/                 # raw/processed/splits (not committed — see DATASET_SELECTION.md)
├── experiments/configs/  # YAML configs per experiment run
├── models/               # Saved model checkpoints (not committed)
├── results/              # Metrics, logs from actual runs
├── visualizations/       # Generated plots/figures
├── notebooks/            # Exploratory notebooks
├── app/                  # Demo application (Streamlit)
├── tests/                # Automated tests
├── research_paper/       # Paper draft, built from real results only
└── source_documents/     # Original synopsis/task PDFs this project is based on
```

## Status

This project is under active development. See `FINAL_PROJECT_AUDIT.md` for the current
checklist of what's actually implemented vs. planned.

## Setup

```powershell
python -m venv venv
./venv/Scripts/pip install -r requirements.txt
```

See `REPRODUCIBILITY.md` for full environment details, including the hardware
constraints this project was built under (CPU-only, limited local disk) and how that
shaped model/dataset choices.
