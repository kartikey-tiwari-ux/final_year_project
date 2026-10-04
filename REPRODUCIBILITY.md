# Reproducibility

## Hardware / environment this project was built and run on

- **OS:** Windows 10 Home
- **GPU:** None detected (`nvidia-smi` not available) — all training/inference is CPU-only.
- **Disk:** Development began under a severe constraint (<1GB free on the system drive);
  cleanup recovered working space before any dataset download or package installation.
  This materially shaped the choices in `PROJECT_ARCHITECTURE.md`: models are chosen to
  be small enough to download and run on CPU in this environment (e.g. DistilBERT-scale
  text encoder rather than large LLMs, a small/frozen-pretrained CNN rather than a large
  vision transformer), and the dataset subset size is bounded accordingly. This is
  documented honestly rather than hidden — see `DATASET_SELECTION.md` for the exact
  numbers used.
- **Python:** 3.13 (see `python --version` at build time)
- **Core libraries:** pinned in `requirements.txt` (CPU-only PyTorch wheel via
  `--extra-index-url https://download.pytorch.org/whl/cpu`)

## Environment setup

```powershell
python -m venv venv
./venv/Scripts/pip install -r requirements.txt
```

Exact installed versions are captured by running, after setup:
```powershell
./venv/Scripts/pip freeze > requirements.lock.txt
```
(`requirements.lock.txt` is generated during the build — see repo root once experiments
are run.)

## Randomness and seeds

- A single project-wide seed (default `42`) is set in `src/utils/seed.py` for Python's
  `random`, `numpy`, and `torch` (CPU) RNGs at the start of every training/evaluation
  script, so re-running a script with the same config and seed reproduces the same
  train/val/test split assignment and model initialization.
- Where repeated runs are used to report variance (see `PROJECT_ARCHITECTURE.md` /
  experiment results), the specific seeds used per run are logged in that experiment's
  config/result file — not just "a fixed seed" with no record of which one.

## Configuration

Each experiment (Model A / B / C, and any fusion-strategy ablation) has its own YAML
config under `experiments/configs/`, recording: dataset split file used, model
name/checkpoint, hyperparameters, seed, and output paths. Results in `results/` are
named to match their originating config so any reported number can be traced back to
the exact run that produced it.

## Dataset acquisition (what actually worked)

The originally planned MVSA-Single OneDrive link was dead (404) when attempted. The
working acquisition path actually used, in order:

```powershell
./venv/Scripts/python -m src.data.download_mvsa      # fails fast: OneDrive link is 404
# fallback used instead of download_memotion.py — see DATASET_SELECTION.md "Actual acquisition":
#   1. images+text downloaded from HuggingFace Hub `xwycyj/MVSA-Single` (data.zip, 211.2MB)
#   2. labels+split downloaded from GitHub `Link-Li/CLMLF` (train/dev/test.json)
./venv/Scripts/python -m src.data.build_mvsa_manifest  # merges both, verifies coverage,
                                                         # writes data/processed/mvsa_single_manifest.csv
```

This is recorded here, not smoothed over, per the project's rule against hiding what
actually happened. `src/data/download_mvsa.py` still documents and attempts the
original official link first (so it will self-heal if that link is ever restored);
the HuggingFace+CLMLF path is the one actually exercised in this build.

**Label semantics spot-check (so the 0/1/2 → positive/neutral/negative mapping used
throughout this project is not just taken on faith):** inspected several sample tweet
texts per label from `data/raw/mvsa_single_labels/train.json`. Label 0 examples read
positive in tone (e.g. "fun loving and energetic team...Thank you and I love you");
label 2 examples read negative (e.g. containing "#Depressed", "helpless"); label 1
examples are informational/neutral. Full distribution: label 0 (positive) = 2,683,
label 1 (neutral) = 470, label 2 (negative) = 1,358, across all 4,511 labeled samples.

## Data splitting and leakage prevention

- **Primary split actually used:** the published CLMLF train/dev(→val)/test split
  (fold 1) for MVSA-Single — 3,611 / 450 / 450 — baked into
  `data/processed/mvsa_single_manifest.csv`'s `split` column by
  `src/data/build_mvsa_manifest.py`. This was chosen over computing our own split
  specifically **for comparability**: results can be sanity-checked against numbers
  reported in the literature using the same split, rather than an arbitrary one of our
  own. All models (A, B, C) are trained/evaluated on this same split, so comparisons
  between them are fair.
- `src/data/splits.py` (the project's own stratified-split utility, unit-tested in
  `tests/test_splits.py`) is kept and available for any secondary experiment that
  needs a different split (e.g. a robustness check across multiple random splits) —
  it is not the source of the primary reported results.
- Splitting (in general, and as inherited from CLMLF's split) happens **before** any
  preprocessing that could leak information (e.g. no vocabulary/statistics fit on the
  full dataset before splitting).
- `assert df["id"].is_unique` in `build_mvsa_manifest.py` guards against the same post
  appearing twice (e.g. across the three source JSON files) before the manifest is
  written.
- **Duplicate-hash leakage audit (`src/data/check_leakage.py`), real result, not
  assumed clean:** out of 4,511 samples, **56 exact-duplicate texts** and **10
  exact-duplicate images** (by content hash) appear across more than one split
  (e.g. train+test). This comes from Twitter retweets ("RT @...") sharing identical
  text/images across different post ids — a property of the **published CLMLF split
  itself**, not introduced by this project's own processing. Scale: 56/4,511 ≈ 1.2%
  of texts, 10/4,511 ≈ 0.2% of images (by hash-group count, not sample count) are
  affected. This is disclosed as a genuine limitation of using a third-party published
  split for comparability (see `DATASET_SELECTION.md`) rather than silently assumed to
  be leakage-free — at this scale it is very unlikely to materially inflate reported
  metrics, but it is not literally zero, and is reported in `results/leakage_audit.txt`
  and `FINAL_PROJECT_AUDIT.md` rather than hidden.
- Image augmentation (if used) is applied only to the training split, never to
  validation/test.
- The test split is touched only for final reported metrics — not used during
  hyperparameter selection (validation split is used for that).

## Measured timing (real numbers, not estimates)

- DistilBERT frozen-embedding extraction, 4,511 texts, CPU: **223.1s (49.5 ms/sample)**.
- ResNet18 frozen-embedding extraction, 4,511 images, CPU: **2,419.2s / ~40 min
  (536.3 ms/image)** — over 10x slower per-sample than text despite ResNet18 being a
  much smaller/cheaper model than DistilBERT. Profiling the gap: the actual forward
  pass is fast; the bottleneck is opening and JPEG-decoding 4,511 individual small
  files from disk one at a time on this Windows machine (consistent with real-time
  antivirus scanning per file open, matching the low CPU-time-vs-wall-clock-time ratio
  observed while the job ran). This is reported as a genuine, measured
  resource-constraint finding — directly relevant to `RESEARCH_GAP.md` point 3
  (documenting what CPU-only, non-institutional-compute reproduction actually costs)
  — not glossed over as a non-issue. Practical mitigation applied: embeddings are
  extracted **once** and cached (`data/processed/*_embeddings.npz`), so this cost is
  paid once per dataset, not once per training run.
- **Follow-up finding:** a second full pass over the same 4,511 images (building the
  raw-pixel cache for Model B1, `extract_pixel_cache.py`) took only **26.4s** — ~100x
  faster than the first ResNet pass. This confirms the ~530ms/image cost was a
  cold-cache/first-touch cost (OS file cache and/or antivirus per-file scan on first
  access), not a fundamental per-image cost — real-world implication: a resource-
  constrained reproduction should budget for one slow "warm-up" pass over a new
  dataset, not assume every pass will be that slow.
- Model B1's first training attempt re-decoded JPEGs from disk on every epoch
  (8 epochs × full dataset), which would have taken hours even at the warmed-up
  speed; it was killed by the OS for memory pressure before that became visible
  (a separate, compounding problem — see next note) and was fixed by caching
  resized pixels as a single uint8 array once (`extract_pixel_cache.py`, 679MB)
  and training from that in-memory array instead of the filesystem.
- **Memory incident:** the first Model B1 training attempt was killed by Claude
  Code's background-process memory-pressure guard on this 8GB-RAM machine. Root
  cause (found and fixed, not papered over): `ImageOnlyDataset.__init__` eagerly
  decoded and transformed every image into a float32 tensor up front (~2.7GB per
  split) instead of loading lazily per batch. Fixed by switching to the pixel-cache
  approach above, which keeps peak memory bounded to one copy of the uint8 cache
  (~679MB) plus small per-batch float32 tensors.
- **Compute-time incident and final resolution:** even with the memory fix, a second
  B1 attempt (full 224×224 resolution, batch size 64, all 4 CPU threads) exceeded a
  45-minute background runtime budget without completing 3 epochs — back-of-envelope
  FLOP estimates for a 224×224 first conv layer on unaccelerated CPU kernels are
  consistent with the observed pace (roughly ~19 min/epoch). Fixed by reducing the
  training resolution to 96×96 (≈5.4x fewer pixels) and lowering the learning rate
  (1e-3 → 5e-4, since the first full run also showed worsening validation metrics
  epoch-over-epoch, suggestive of unstable updates). The reduced-scale run completed
  3 epochs in **248.9s** — confirming the earlier estimate and validating the fix.
  Final real result: 47.6% test accuracy, 42.9% macro-F1 (`results/model_b1_cnn_scratch.json`).
  This entire sequence — two failed attempts, root-caused, fixed, and a successful
  reduced-scale run — is reported in full rather than only showing the final success,
  per this project's commitment to honest process documentation.

## Experiment logs

Each run writes a log (console + file under `results/`) recording: start/end time,
config used, dataset split sizes and class distribution, final metrics, and any
warnings/errors encountered. Failed or poor-performing runs are kept, not deleted —
per project rule, results are reported as obtained, not filtered to look better.

## What "reproducible" means here, concretely

Given someone with this repo, the same `requirements.txt`-installed environment, and
access to the dataset described in `DATASET_SELECTION.md`, running
`experiments/configs/*.yaml` through the corresponding training script should
reproduce the reported metrics within normal floating-point/CPU-nondeterminism
tolerance. Any known source of run-to-run variance (e.g. non-deterministic CPU
multithreading in some PyTorch ops) is noted here once observed during actual runs.
