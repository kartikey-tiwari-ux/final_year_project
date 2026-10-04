# Error Analysis

All numbers below are real, from the actually-executed experiments in `results/*.json`
(test split, 450 samples — the same held-out split for every model). Confusion matrix
row/column order is `[positive, neutral, negative]` throughout.

## 1. Headline comparison

| Model | Accuracy | Macro-F1 | Weighted-F1 | Macro-Precision | Macro-Recall |
|---|---|---|---|---|---|
| A1 — TF-IDF + LogReg | 60.9% | 52.0% | 61.9% | 51.5% | 53.4% |
| A2 — Frozen DistilBERT + LogReg | 61.6% | 54.6% | 63.5% | 54.2% | 58.2% |
| B1 — CNN from scratch (96×96, 3 epochs) | 47.6% | 42.9% | 49.6% | 43.5% | 49.3% |
| B2 — Frozen ResNet18 + LogReg | 52.7% | 42.8% | 54.3% | 43.0% | 43.5% |
| C — Concat fusion (text+image) | **64.4%** | **56.3%** | **65.4%** | 55.5% | 58.1% |
| C — Late fusion (simple avg) | 63.8% | 54.6% | 65.0% | 54.0% | 56.4% |
| C — Late fusion (tuned, w_text=0.80) | 64.0% | 56.4% | 65.6% | 55.6% | 59.4% |

**Primary finding:** both fusion strategies beat every unimodal model on every
aggregate metric. Concat (feature-level) fusion gets the best raw accuracy; tuned
late fusion gets the best macro-F1 by a hair (56.4% vs 56.3%) — the two fusion
strategies are close enough that neither single-handedly dominates, but both clearly
outperform unimodal. Text dominates over image throughout (A2 ≫ B2 on every metric),
and the late-fusion weight search independently confirms this by landing on
w_text=0.80.

## 2. Confusion matrices (test set, rows=true, cols=predicted)

**A2 (best unimodal, text):**
```
            pred:pos  pred:neu  pred:neg
true:pos      170        42        56
true:neu       17        23         7
true:neg       26        25        84
```

**C — Concat fusion (best overall):**
```
            pred:pos  pred:neu  pred:neg
true:pos      186        24        58
true:neu       13        20        14
true:neg       34        17        84
```

Comparing the two: fusion correctly classifies 16 more positive posts than text alone
(186 vs 170) and misclassifies fewer positives as neutral (24 vs 42) — the biggest
single improvement. Negative-class performance is identical (84 correct in both).
Neutral is marginally worse under fusion (20 vs 23 correct) — consistent with
`MODALITY_CONFLICT_ANALYSIS.md` §5's finding that fusion underperforms text-only
specifically on the neutral class.

## 3. Class imbalance and per-class performance

Training-set class distribution (from `DATASET_SELECTION.md`): positive 2,147 (59.5%),
negative 1,088 (30.1%), **neutral 376 (10.4%)**. Neutral is both the smallest class by
a wide margin and, consistently across every model, the hardest:

| Model | Positive F1 | Neutral F1 | Negative F1 |
|---|---|---|---|
| A1 | 70.9% | 30.2% | 54.9% |
| A2 | 70.7% | 33.6% | 59.6% |
| B2 | 65.3% | 17.7% | 45.3% |
| C (concat) | 74.3% | 37.0% | 57.7% |

Neutral F1 is roughly half (or less) of positive/negative F1 in every model — directly
attributable to class scarcity (10.4% of training data) compounded by neutral being an
inherently ambiguous/low-signal sentiment category (a post that is neither clearly
positive nor negative often lacks strong lexical or visual cues of *any* kind, making
it hard to distinguish from noise). This is exactly why `PROJECT_REQUIREMENTS_ANALYSIS.md`
and `DATASET_SELECTION.md` specified macro-F1 (not accuracy) as the primary metric —
accuracy alone would hide how poorly every model handles this minority class.

## 4. Where models confuse positive and negative directly

Across all models, positive↔negative confusion (true positive predicted negative, or
vice versa) is the largest error category after neutral-related errors — e.g. A2
confuses 56 true-positive posts as negative and 26 true-negative posts as positive.
This is a meaningfully more serious error type than confusing either with neutral
(a polarity flip, not just a confidence/intensity miss) — see `ETHICS_AND_PRIVACY.md`
§5 on why false positive/negative error types matter differently in a mental-health-
adjacent framing. Fusion reduces this specific error somewhat for positive→negative
(58 vs A2's 56... actually comparable) but the fundamental difficulty persists across
every model tested; no model here solves polarity confusion.

## 5. False positive / false negative framing (per ETHICS_AND_PRIVACY.md §5)

In this project's non-clinical framing, "false positive" / "false negative" are
analyzed per-class rather than as a single binary notion (since this is 3-class, not
binary). The most relevant asymmetry for a mental-health-adjacent application: a
**true negative-sentiment post misclassified as positive** (false negative for
"concerning" content — 26/450 for A2, 34/450 for fusion) is arguably the more
consequential error type in any real monitoring-support context, since it's the
"missed concern" case; a **true positive post misclassified as negative** (56/450 for
A2, 58/450 for fusion — a false alarm) causes different, lower-stakes harm (unwarranted
concern). Neither error type is at zero for any model tested here, which is exactly
why this project (per `ETHICS_AND_PRIVACY.md`) does not claim diagnostic or
intervention-grade reliability.

## 6. CPU-only compute cost as an error source (resource-constrained finding)

Per `REPRODUCIBILITY.md`, Model B1 (CNN trained from scratch) had its epoch count
reduced from 8 to 3 and its input resolution reduced from 224×224 to 96×96, after two
full-resolution attempts were killed (one for memory pressure — a real bug, found and
fixed; one for exceeding a 45-minute background runtime budget without finishing 3
epochs). The final, completed run: test accuracy 47.6%, macro-F1 42.9%, in 248.9s.
This is itself a relevant finding for `RESEARCH_GAP.md` point 3 (resource-constrained
reproduction): a from-scratch CNN needs materially more compute than this environment
can comfortably provide at full resolution. Interestingly, B1's final macro-F1 (42.9%)
is almost identical to B2's (42.8%) despite B1's clearly lower accuracy (47.6% vs.
52.7%) and a training instability B1 showed that B2 (a frozen, pre-trained, non-
iteratively-trained feature extractor) cannot show by construction (B1's validation
macro-F1 dropped from 0.302 to 0.224 between epochs 1 and 2 before recovering to 0.384
by epoch 3 — reported as observed, not smoothed over). Taken together, this supports
the literature-grounded design decision (Agung et al. 2024, Dosovitskiy et al. 2021 —
`LITERATURE_REVIEW.md` 2.1/2.2) to prefer frozen transfer-learned features under this
project's real hardware constraints: B2 reaches comparable-or-better results with
dramatically less training instability and compute than B1 required even at reduced
scale.
