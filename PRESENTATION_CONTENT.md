# Presentation Content (~18 slides)

## Slide 1 — Title
**AI-Driven Multimodal Sentiment Analysis for Mental Health Monitoring on Social Media**
Group 15 | Kartikey Tiwari, Akshat Sharma, Karan Sharma, Lalit Sharma
Supervisor: Dr. Avdhesh Gupta | AKGEC, Ghaziabad (AKTU) | Session 2023–2027

## Slide 2 — Problem
- Social media posts often combine **text + image**
- Text-only or image-only sentiment analysis each miss signal the other carries
- Does combining both actually improve sentiment classification? We tested it for real.

## Slide 3 — Motivation
- Mental-health awareness is increasingly digital-first
- Social media is a large, real source of expressed sentiment
- AI can support — not replace — research/monitoring in this space (non-clinical framing, always)

## Slide 4 — Existing Approaches
- Text-based: NLP/transformers (e.g. MentalBERT) — strong, but ignore visual content
- Image-based: CNN emotion recognition — strong, but ignore textual meaning
- Multimodal: an active, well-populated research area (not new) — taxonomy of fusion strategies exists

## Slide 5 — Research Gap
- No located prior work does a **controlled, same-dataset, same-metrics** text-vs-image-vs-multimodal comparison on **adequately-sized real social-media data**
- Closest comparable work tests its social-media condition on only **100 pairs**
- Most multimodal literature is resource-unconstrained (GPU-scale) — we report what this costs on CPU-only hardware

## Slide 6 — Objectives
1. Build and compare Text-only, Image-only, and Multimodal models on the same data/metrics
2. Determine whether/when fusion helps
3. Analyze modality conflict and errors honestly
4. Document resource-constrained reproduction

## Slide 7 — Dataset
- **MVSA-Single**: 4,511 real Twitter image+text pairs (after standard cleaning), 3-class sentiment (positive/neutral/negative)
- Published train/dev/test split (3,611/450/450) from CLMLF (NAACL 2022) for comparability
- Class imbalance: 59.5% positive, 30.1% negative, 10.4% neutral
- Real acquisition challenge: official link was dead (404) — found and verified an alternative path

## Slide 8 — Proposed Architecture
- Text pipeline → embedding → classifier
- Image pipeline → embedding → classifier
- Fusion: concatenate embeddings → joint classifier
- (Diagram: see `PROJECT_ARCHITECTURE.md`)

## Slide 9 — Text Pipeline
- Clean: strip URLs, normalize mentions, demojize emoji to text
- A1: TF-IDF + Logistic Regression (traditional baseline)
- A2: Frozen DistilBERT embeddings + Logistic Regression (transformer model)

## Slide 10 — Image Pipeline
- Resize/normalize
- B1: small CNN trained from scratch (baseline)
- B2: frozen pretrained ResNet18 embeddings + Logistic Regression (modern/transfer-learning model)

## Slide 11 — Multimodal Fusion
- Primary: feature-level concatenation (text 768-dim + image 512-dim) → MLP classifier
- Ablation: late (decision-level) fusion — combine independent models' predicted probabilities
- Why not full cross-modal attention: CPU-only compute budget; literature validates it mainly on non-social-media (video/audio) benchmarks

## Slide 12 — Experimental Setup
- Fixed seed (42), same split for every model, macro-F1-prioritized evaluation (class imbalance)
- CPU-only, no GPU — model choices and training budgets shaped by this real constraint

## Slide 13 — Results
| Model | Accuracy | Macro-F1 |
|---|---|---|
| A1 TF-IDF+LogReg | 60.9% | 52.0% |
| A2 DistilBERT | 61.6% | 54.6% |
| B1 CNN from scratch | 47.6% | 42.9% |
| B2 ResNet18 | 52.7% | 42.8% |
| **C Fusion (concat)** | **64.4%** | **56.3%** |

Multimodal fusion beats every unimodal model on every metric. (B1 vs B2: pretrained
features win on accuracy; near-tied on macro-F1 but with real training instability
for B1 — supports our transfer-learning design choice.)

## Slide 14 — Modality Conflict Analysis
- Text and image models **disagree on 53.3%** of test posts
- On disagreement cases: Fusion 57.1% vs Text 50.0% vs Image 33.3% — fusion's gain concentrates here
- On agreement cases: small regression (Fusion 72.9% vs Text 74.8%)
- Honest finding, not a blanket "multimodal always wins"

## Slide 15 — Explainability
- Text: gradient×input token saliency (no extra library)
- Image: Grad-CAM via forward/backward hooks
- Multimodal: modality-ablation (zero one branch, observe prediction shift)
- Exploratory — not a rigorously validated attribution method (documented limitation)

## Slide 16 — Application / Dashboard
- Streamlit demo: text/image input → text-only, image-only, and fusion predictions + confidence
- Research dashboard: model comparison, confusion matrices, class distribution
- Persistent non-clinical disclaimer throughout

## Slide 17 — Research Contribution & Limitations
**Contribution:** real, same-conditions unimodal-vs-multimodal comparison on adequately-sized social-media data; honest modality-conflict analysis; documented resource-constrained reproduction.
**Limitations:** sentiment labels (not clinical), English/Twitter/2015-16-era only, small published-split duplication, from-scratch CNN baseline compute-constrained.

## Slide 18 — Future Work & Conclusion
- Future: multilingual extension, cross-modal attention at scale, genuinely mental-health-labeled multimodal data if it becomes available
- Conclusion: multimodal fusion provides a real, measurable, but conditional improvement — concentrated where modalities disagree — answering the project's central research question with evidence, not assumption
