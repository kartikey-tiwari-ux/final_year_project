| Model | Acc. | Macro-F1 | Weighted-F1 |
|---|---|---|---|
| A1 -- TF-IDF + LogReg (text) | 42.6% | 33.3% | 43.7% |
| A2 -- Frozen DistilBERT + LogReg (text) | 39.4% | 33.4% | 42.1% |
| B1 -- CNN from scratch (image, sharp) | 61.1% | 25.3% | 46.4% |
| B2 -- Frozen ResNet18 + LogReg (image, sharp) | 38.6% | 31.0% | 42.2% |
| B2-Blur -- Frozen ResNet18 + LogReg (image, blurred) | 37.1% | 30.7% | 39.6% |
| C -- Concat Fusion (text + sharp image) | 52.6% | 36.6% | 50.6% |
| C-Blur -- Concat Fusion (text + blurred image) | 41.7% | 33.0% | 44.4% |
