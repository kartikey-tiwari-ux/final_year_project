# Final Project Audit

Living checklist, updated as work actually completed (not aspirational). Checked items
link to the artifact that proves them.

- [x] Synopsis requirements understood — `PROJECT_REQUIREMENTS_ANALYSIS.md`
- [x] Text-only model implemented — Model A1 (`src/text_model/baseline_tfidf.py`, `results/model_a1_tfidf_logreg.json`) and A2 (`src/text_model/distilbert_classifier.py`, `results/model_a2_distilbert_frozen.json`)
- [x] Image-only model implemented — Model B1 (`src/image_model/baseline_cnn.py`, `results/model_b1_cnn_scratch.json`) and B2 (`src/image_model/resnet_classifier.py`, `results/model_b2_resnet18_frozen.json`)
- [x] Multimodal model implemented — concat fusion (`src/multimodal/fusion_concat.py`, `results/model_c_concat_fusion.json`) and late fusion ablation (`src/multimodal/fusion_late.py`, `results/model_c_late_fusion.json`)
- [x] Authentic dataset selected — `DATASET_SELECTION.md` (MVSA-Single)
- [x] Dataset provenance documented — `DATASET_SELECTION.md` "Actual acquisition" (including the real OneDrive-link-dead detour and its resolution)
- [x] Dataset labels verified — empirically spot-checked against real downloaded data (`REPRODUCIBILITY.md` label semantics spot-check), not assumed from the source paper alone
- [x] Dataset licensing checked — `DATASET_SELECTION.md` / `ETHICS_AND_PRIVACY.md` (MVSA: no formal license, cite+contact convention, disclosed; provenance update also disclosed after the official link turned out to be dead)
- [x] Ethics documented — `ETHICS_AND_PRIVACY.md`
- [x] Dataset actually downloaded — 4,869 real image+text pairs (HuggingFace mirror, content-verified) + 4,511 labels/split (CLMLF repo), merged into `data/processed/mvsa_single_manifest.csv`, zero missing files
- [x] Data leakage checked — `src/data/check_leakage.py` run for real: found and disclosed 56/4,511 duplicate texts and 10/4,511 duplicate images across splits (inherited from the published CLMLF split's retweet content, not introduced by this project) — see `REPRODUCIBILITY.md` and `results/leakage_audit.txt`
- [x] Experiments executed — all 6 model variants (A1, A2, B1, B2, C-concat, C-late) trained and evaluated on identical splits
- [x] Results generated from real data — every number in every doc traces to a `results/*.json` file produced by an actually-executed script
- [x] Metrics calculated correctly — `src/evaluation/metrics.py`, unit-tested (`tests/test_metrics.py`), one real bug caught and fixed (missing `labels=` param crashing on imbalanced batches)
- [x] Ablation completed — text-only vs. image-only vs. multimodal (required minimum), plus concat-fusion vs. late-fusion (simple and validation-tuned weighted) as the fusion-strategy ablation
- [x] Error analysis completed — `ERROR_ANALYSIS.md` (confusion matrices, per-class breakdown, class-imbalance analysis, polarity-confusion analysis, false-positive/negative framing)
- [x] Modality conflict analysis completed — `MODALITY_CONFLICT_ANALYSIS.md` (real finding: fusion's gain concentrates on the 53.3% of test posts where text/image disagree)
- [x] Explainability implemented — text saliency (`src/explainability/text_saliency.py`) and image Grad-CAM (`src/explainability/image_gradcam.py`), run on real test examples (`src/explainability/generate_examples.py`, `results/explainability_examples/`), limitations documented
- [x] Application functional — `app/streamlit_app.py` (demo + research dashboard); launched for real (`streamlit run`), server started cleanly with no startup errors, HTTP 200 + health check both passed; interactive click-through (Predict button, model-loading path) not exercised in this automated session — see note below
- [x] Tests executed — 32/32 passing (`TESTING.md`), across seed, text-cleaning, splits, metrics, dataset-loading, and image-preprocessing modules
- [x] Core planning documentation complete — requirements, literature review, research gap, dataset selection, ethics, architecture, reproducibility, originality
- [x] Research paper generated — `research_paper/paper.md`, `references.bib` (real results only, 14 verified citations)
- [x] Project report generated — `docs/FINAL_PROJECT_REPORT.md`
- [x] Viva preparation generated — `VIVA_PREPARATION.md`
- [x] Presentation content generated — `PRESENTATION_CONTENT.md`
- [x] No fabricated results — every reported number sourced from an actually-run script's saved output
- [x] No fabricated citations — `LITERATURE_REVIEW.md` verification note; all 14 sources web-verified
- [x] No fake dataset — real MVSA data, full provenance chain documented including the detour around a dead official link
- [x] No hardcoded credentials — `.env.example` pattern used, `.env` gitignored

## Known limitations (carried forward honestly, not hidden)
- MVSA's official OneDrive link was confirmed dead; this project used a verified
  third-party mirror + a separate paper's published split instead (fully disclosed,
  `DATASET_SELECTION.md`).
- Small (~1–2%) cross-split content duplication exists in the published split used,
  inherited from Twitter retweets (`REPRODUCIBILITY.md`, `results/leakage_audit.txt`).
- Model B1 (CNN from scratch) was trained at reduced resolution (96×96, down from
  224×224) and only 3 epochs, due to measured CPU-only compute constraints on this
  8GB-RAM, no-GPU machine — two earlier full-scale attempts failed (one OOM-killed,
  one exceeded a runtime budget), both root-caused and documented rather than retried
  blindly (`REPRODUCIBILITY.md`).
- Only a single random seed was used for the neural models (fusion MLP, CNN); run-to-
  run variance across seeds was not characterized given the time/compute budget.
- The demo app was launched for real and confirmed to start cleanly (HTTP 200, health
  check OK, no startup traceback), but the interactive flow (typing text, uploading an
  image, clicking Analyze, which lazy-loads the models) was not click-tested through a
  browser in this automated session. Recommended before a live presentation/viva demo:
  run `./venv/Scripts/streamlit run app/streamlit_app.py` and click through both pages
  once, including an actual Analyze click, to confirm the model-loading path works
  interactively too.
- Sentiment labels (not literal mental-health labels) are used throughout as a
  documented proxy — see `ETHICS_AND_PRIVACY.md` for why this boundary is maintained.
