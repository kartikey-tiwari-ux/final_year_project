# Testing

Per project rule: tests listed here are only ones **actually executed**, with their
real pass/fail output — not an aspirational list.

## Test suite layout
- `tests/test_seed.py` — reproducibility of `src/utils/seed.set_seed()`.
- `tests/test_text_cleaning.py` — URL/mention stripping, emoji demojizing, lowercasing,
  whitespace handling, edge cases (None/empty input) in `src/preprocessing/text_cleaning.py`.
- `tests/test_splits.py` — stratified split sizing, determinism given a seed,
  different-seed divergence, overlap/leakage detection, and rejection of
  too-small classes in `src/data/splits.py`.
- `tests/test_metrics.py` — perfect/zero-score sanity checks, confusion matrix shape,
  per-class key presence in `src/evaluation/metrics.py` (caught a real bug: missing
  `labels=[0,1,2]` in `classification_report` crashed on batches missing a class).
- `tests/test_mvsa_dataset.py` — manifest loads with expected columns, the real
  confirmed dataset size (4,511) and split sizes (3,611/450/450), label-mapping
  consistency, and that every image path actually exists on disk.
- `tests/test_image_preprocessing.py` — missing/corrupt file handling (returns None,
  doesn't raise), valid image loading, transform output shape and dtype.

## Run

```powershell
./venv/Scripts/python -m pytest -v
```

## Results

Latest full run: `./venv/Scripts/python -m pytest -v`, Python 3.13.15, pytest 9.1.1.

```
collected 32 items
... (all 32 tests across 6 files)
32 passed in 4676.19s (1:17:56)
```

All 32 tests pass. Re-run in isolation (no competing background jobs) for an accurate
timing figure:

```
32 passed in 8.34s
```

confirming the earlier 4676s figure was CPU contention from Model B1's concurrent
training job (see REPRODUCIBILITY.md), not a real cost of the tests themselves.
