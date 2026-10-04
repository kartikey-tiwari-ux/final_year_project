# Literature Review

All entries below were located and verified via live web search/fetch (not recalled from memory and not fabricated). Each entry lists what could be confirmed; where a detail (e.g., exact limitations section) could not be confirmed from the available abstract/page content, that is stated rather than guessed.

---

## Theme 1: Text-Based Mental Health Analysis

### 1.1 Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory.* Neural Computation, 9(8), 1735–1780.
- **Methodology:** Introduces the LSTM recurrent neural network architecture with gated memory cells to address vanishing/exploding gradients in standard RNNs over long sequences.
- **Dataset:** Synthetic sequence-learning benchmark tasks (not social-media text).
- **Results:** LSTM solves long time-lag tasks that classical RNNs and earlier approaches could not.
- **Limitations:** Foundational architecture paper, not evaluated on any sentiment/mental-health task.
- **Relevance:** Foundational sequence model historically used for text-based sentiment/mental-health classification before transformer dominance; cited directly by the project synopsis's own reference list (Task 2 document).

### 1.2 Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* Proceedings of NAACL-HLT 2019, pp. 4171–4186. (arXiv:1810.04805)
- **Methodology:** Pretrains a deep bidirectional Transformer with a masked-language-model objective and next-sentence prediction, then fine-tunes for downstream tasks.
- **Dataset:** BooksCorpus + English Wikipedia (pretraining); evaluated on GLUE, SQuAD v1.1/v2.0, MultiNLI (fine-tuning benchmarks).
- **Results:** New state of the art at publication: GLUE 80.5%, MultiNLI 86.7%, SQuAD v1.1 Test F1 93.2, SQuAD v2.0 Test F1 83.1.
- **Limitations:** General-domain pretraining; not specialized for social-media or mental-health text (motivates domain-adapted variants, see 1.3).
- **Relevance:** BERT is the base architecture for most modern transformer-based text classifiers used in this project's Model A (text-only) baseline.

### 1.3 Ji, S., Zhang, T., Ansari, L., Fu, J., Tiwari, P., & Cambria, E. (2022). *MentalBERT: Publicly Available Pretrained Language Models for Mental Healthcare.* Proceedings of LREC 2022.
- **Methodology:** Domain-adaptive pretraining (continued pretraining) of BERT-base and RoBERTa on a large corpus of mental-health-related Reddit posts (e.g., r/depression, r/SuicideWatch, r/Anxiety), producing MentalBERT and MentalRoBERTa.
- **Dataset:** Reddit mental-health subreddit corpus (pretraining); evaluated on multiple downstream mental-health classification benchmarks (depression, stress, suicidal ideation, multi-label disorder detection).
- **Results:** Domain-adapted models outperform general-purpose BERT/RoBERTa and biomedical/clinical variants (BioBERT, ClinicalBERT) on most downstream mental-health tasks.
- **Limitations:** Performance depends on dataset quality/representativeness; generalization across different platforms/populations needs further study; class imbalance common to mental-health datasets remains a challenge.
- **Relevance:** Directly relevant prior art for the text-only pipeline; illustrates the value (and limits) of domain-specific pretraining for mental-health-related text classification — a design choice this project must weigh against compute/disk constraints.

### 1.4 Cao, Y., Dai, J., Wang, Z., Zhang, Y., Shen, X., Liu, Y., & Tian, Y. (2024). *Machine Learning Approaches for Mental Illness Detection on Social Media: A Systematic Review of Biases and Methodological Challenges.* arXiv:2410.16204.
- **Methodology:** Systematic review of peer-reviewed literature (PubMed, IEEE Xplore, Google Scholar; studies published after 2010) using the PROBAST (Prediction model Risk Of Bias ASsessment Tool) framework; 47 papers analyzed.
- **Dataset:** N/A (meta-review of other studies' datasets).
- **Results:** Twitter dominates as a data source (63.8% of studies); >90% English-only; 80% used non-probability sampling (representativeness concerns); only 23% explicitly handled linguistic features like negation; 27.7% had inconsistent hyperparameter tuning; 17% had inadequate data partitioning (overfitting risk); 74.5% used appropriate metrics for imbalance, meaning ~25% did not.
- **Limitations:** The review itself calls for diversified data sources, standardized preprocessing, and more transparent reporting across the field.
- **Relevance:** Directly informs this project's methodological choices (train/val/test discipline, imbalance-aware metrics, documenting sampling/representativeness limits) and the Research Gap below.

---

## Theme 2: Image-Based Emotion Analysis

### 2.1 Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., et al. (2021). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.* ICLR 2021. (arXiv:2010.11929)
- **Methodology:** Applies the Transformer architecture directly to sequences of image patches (Vision Transformer, ViT), without convolutional inductive bias, pretrained on large image datasets then fine-tuned.
- **Dataset:** ImageNet-21k / JFT-300M (pretraining); ImageNet, CIFAR, VTAB (evaluation).
- **Results:** Matches or exceeds CNN state of the art (e.g., ResNet-based models) when pretrained on sufficiently large data, at lower training compute than comparable CNNs.
- **Limitations:** Requires large-scale pretraining data to be competitive; less effective than CNNs when trained from scratch on small datasets — directly relevant to this project's CPU-only, small-dataset constraint (favors a smaller/frozen pretrained CNN over training a ViT from scratch).
- **Relevance:** Candidate architecture family for the image-only pipeline; its data-hungriness is a key reason this project is expected to prefer a lighter pretrained CNN/frozen-encoder approach (documented in `PROJECT_ARCHITECTURE.md` / `REPRODUCIBILITY.md`).

### 2.2 Agung, E. S., Rifai, A. P., & Wijayanto, T. (2024). *Image-based facial emotion recognition using convolutional neural network on emognition dataset.* Scientific Reports, 14. DOI: 10.1038/s41598-024-65276-x.
- **Methodology:** CNN-based facial emotion recognition using transfer learning (fine-tuned Inception-V3 and MobileNet-V2) versus a from-scratch CNN tuned via the Taguchi method; pipeline includes video-to-frame extraction, face cropping, cleaning, augmentation.
- **Dataset:** Emognition dataset, ten discrete emotions (amusement, awe, enthusiasm, liking, surprise, anger, disgust, fear, sadness, neutral); 2,535 facial images after preprocessing.
- **Results:** Inception-V3 transfer-learning model: 96% accuracy, 0.95 average F1; outperformed MobileNet-V2 (89%) and the from-scratch model (87%).
- **Limitations:** Relies on static images rather than video sequences, so dynamic facial movement is not captured; authors note a single static image may not fully represent the expression range for an emotion class, limiting real-world generalizability.
- **Relevance:** Demonstrates that transfer learning from a pretrained CNN substantially outperforms training from scratch on a modest image dataset — directly supports this project's planned use of a frozen/lightly fine-tuned pretrained CNN for the image-only pipeline under compute constraints. Also a useful cautionary note: this is facial-emotion recognition specifically, not general social-media image sentiment (memes, scenes, objects), a scope difference worth flagging when interpreting image-only results here.

### 2.3 Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization.* ICCV 2017.
- **Methodology:** Produces a coarse localization map highlighting image regions important to a CNN's prediction, using gradients flowing into the final convolutional layer — requires no architecture change or retraining.
- **Dataset:** Evaluated across standard image classification/captioning/VQA benchmarks (ImageNet-trained CNNs, etc.).
- **Results:** Widely adopted (one of the most cited explainability techniques, ~23,500+ citations per secondary sources found during this search).
- **Limitations:** Coarse (low-resolution) localization relative to the input; explanation quality depends on the chosen convolutional layer; does not explain non-CNN architectures without adaptation.
- **Relevance:** Primary candidate method for this project's image-side explainability (Grad-CAM over the image-only and multimodal models' visual branch), per the master build prompt's explainability requirement.

---

## Theme 3: Multimodal Sentiment and Mental Health Analysis

### 3.1 Niu, T., Zhu, S., Pang, L., & El Saddik, A. (2016). *Sentiment Analysis on Multi-View Social Data.* Proceedings of the International Conference on Multimedia Modeling (MMM 2016).
- **Methodology:** Introduces the MVSA dataset — image–text pairs from Twitter manually annotated with sentiment labels — and studies whether jointly modeling text and image views improves sentiment classification versus single-view approaches.
- **Dataset:** MVSA-Single (5,129 image-text pairs, single annotator) and MVSA-Multiple (19,600 pairs, three annotators), collected from Twitter, labeled positive/neutral/negative.
- **Results:** Joint text+image modeling improves over single-view (text-only or image-only) baselines.
- **Limitations:** General Twitter sentiment, not mental-health-specific; label noise from single-annotator subset (MVSA-Single); sentiment polarity only (no finer-grained emotion categories like stress/anxiety/loneliness).
- **Relevance:** MVSA is a standard, publicly available, real (non-synthetic) multimodal social-media benchmark — a strong candidate dataset family for this project, evaluated in detail in `DATASET_SELECTION.md`.

### 3.2 Gandhi, A., Adhvaryu, K., Poria, S., Cambria, E., & Hussain, A. (2023). *Multimodal sentiment analysis: A systematic review of history, datasets, multimodal fusion methods, applications, challenges and future directions.* Information Fusion, 91, 424–444.
- **Methodology:** Systematic literature review covering the historical development of multimodal sentiment analysis, available datasets, fusion method taxonomy, applications, and open challenges.
- **Dataset:** N/A (meta-review).
- **Results:** Consolidates and categorizes fusion approaches (feature-level/early, decision-level/late, hybrid, attention-based) and highlights dataset and benchmark landscape.
- **Limitations:** As a survey, does not itself produce new experimental results; coverage reflects the literature available at time of writing.
- **Relevance:** Primary reference for this project's fusion-strategy taxonomy and for situating the chosen fusion method (`PROJECT_ARCHITECTURE.md`) within the broader field.

### 3.3 Das, R., & Singh, T. D. (2023). *Multimodal Sentiment Analysis: A Survey of Methods, Trends, and Challenges.* ACM Computing Surveys, 55(13s). DOI: 10.1145/3586075.
- **Methodology:** Comprehensive survey of sentiment-analysis approaches across unimodal-to-multimodal evolution, covering applications, challenges, and available resources.
- **Dataset:** N/A (meta-review).
- **Results:** Documents the field's shift from unimodal to multimodal methods and the trends driving it.
- **Limitations:** Survey-level; no new experiments.
- **Relevance:** Cross-checks and complements Gandhi et al. (3.2) as a second independent survey, strengthening confidence in the fusion-method taxonomy used in this project's design.

### 3.4 Jiang, M., & Ji, S. (2022). *Cross-Modality Gated Attention Fusion for Multimodal Sentiment Analysis.* arXiv:2208.11893.
- **Methodology:** Proposes CMGA, a cross-modality gated-attention fusion model with a forget-gate mechanism to suppress noisy/redundant cross-modal signals while capturing shared and modality-unique information.
- **Dataset:** CMU-MOSI and CMU-MOSEI (multimodal — text, audio, video — sentiment benchmarks, not static image+text social posts).
- **Results:** Reported to outperform baseline fusion models on MOSI/MOSEI (specific numeric scores not confirmable from the abstract alone).
- **Limitations:** Evaluated on video/audio/text benchmarks, not static social-media image+text; abstract does not state explicit limitations.
- **Relevance:** Relevant as an attention-based fusion design reference, but note the modality mismatch (video+audio+text vs. this project's image+text) — a caution against directly transplanting MOSI/MOSEI-tuned architectures without adaptation.

### 3.5 Alam, F., Ofli, F., & Imran, M. (2018). *CrisisMMD: Multimodal Twitter Datasets from Natural Disasters.* Proceedings of the 12th International AAAI Conference on Web and Social Media (ICWSM 2018). (arXiv:1805.00713)
- **Methodology:** Collects and manually annotates tweets with paired images across seven major 2017 natural disasters, labeling informativeness and humanitarian categories.
- **Dataset:** Several thousand manually annotated tweet–image pairs (CrisisMMD).
- **Results:** Establishes CrisisMMD as a benchmark for multimodal crisis-informatics classification.
- **Limitations:** Domain is disaster/crisis response, not sentiment or mental health — labels (informativeness, humanitarian category) are not emotion/sentiment labels.
- **Relevance:** Considered and **rejected** as a primary dataset for this project during dataset selection (see `DATASET_SELECTION.md`) specifically because its labels are not sentiment/emotion-related; included here to document that this alternative was evaluated, not overlooked.

### 3.6 Vasanthi, P., & Viswanatham, V. M. (2026). *Multimodal sentiment analysis: hybrid classification model with image and text feature descriptors.* Scientific Reports, 16, Article 13987. DOI: 10.1038/s41598-026-42912-2. (Received 18 Aug 2025; Accepted 27 Feb 2026; Published 18 Mar 2026.)
- **Note on citation correction:** The project's own Task 2 document cited this as "Vasanthi, P., & Viswanatham, V. M. (2026). Multimodal sentiment analysis," with no venue. This literature review confirms the paper is real, published in Scientific Reports (not a placeholder/future-dated error) — Scientific Reports uses continuous-publication volume/issue numbering, which is why the year is 2026; this has been verified directly from the publisher page and PMC (PMC13133387), not assumed.
- **Methodology:** Preprocessing (tokenization, stemming, YOLO-based object detection for images), feature extraction (N-grams and NDC-based TF-IDF plus emoji features for text; improved multitexon and SLBT descriptors for images), and a hybrid classifier combining an optimized Deep Maxout network with a Modified-Sigmoid-based Bi-GRU, tuned via an "Innovative Beluga Whale Optimization Algorithm" (IBwOA); transfer learning used for the Bi-GRU.
- **Dataset:** Yelp Restaurant Photo Classification (~1,042 images), Apple Products Image Dataset (1,514 images), and a small Twitter set (100 images, 100 texts).
- **Results:** Accuracy 0.947–0.964 across datasets; F-measure 0.927; MCC 0.904; reported computation time 497.55s (favorable vs. compared methods).
- **Limitations:** The Twitter (social-media) evaluation set is very small (100 image-text pairs) relative to the other two product/review datasets; no explicit limitations section was retrievable from the available page content; results are from benchmark/product-review image datasets more than from a large-scale, sentiment-labeled social-media corpus.
- **Relevance:** The authors' own cited inspiration paper; confirmed genuine, but its multimodal fusion is tested on a very small social-media subset — reinforces the Research Gap point that rigorous multimodal-vs-unimodal comparison specifically on substantial, sentiment-labeled social-media image+text data (not small Twitter samples or unrelated product-review images) remains underexplored, which this project's planned use of MVSA (thousands of labeled Twitter pairs) directly addresses.

### 3.7 Li, Z., Xu, B., Zhu, C., & Zhao, T. (2022). *CLMLF: A Contrastive Learning and Multi-Layer Fusion Method for Multimodal Sentiment Detection.* Findings of the Association for Computational Linguistics: NAACL 2022, pp. 2282–2294. (aclanthology.org/2022.findings-naacl.175)
- **Methodology:** Proposes CLMLF — token-level multi-layer fusion of text and image features combined with a supervised contrastive learning objective (two contrastive learning tasks: label-based and data-augmentation-based) to help the model learn sentiment-related commonality across modalities.
- **Dataset:** MVSA-Single and MVSA-Multiple (standard Twitter image-text sentiment benchmarks — same dataset family selected for this project).
- **Results:** Reported to outperform prior fusion baselines on both MVSA-Single and MVSA-Multiple (specific numeric scores not re-verified here beyond the abstract/repo).
- **Limitations:** Evaluated only on the MVSA family; abstract does not detail cross-dataset generalization.
- **Relevance — direct and practical, not just thematic:** this project's actual MVSA-Single train/dev/test split (3,611/450/450, fold 1 of CLMLF's published 10-fold split) and the confirmed 0=positive/1=neutral/2=negative label mapping come directly from this paper's public code release (`github.com/Link-Li/CLMLF`), used after the original MVSA OneDrive distribution link proved dead (404) — see `DATASET_SELECTION.md` "Actual acquisition" for the full account. Using a published, citable split (rather than an arbitrary self-computed one) makes this project's results comparable to CLMLF's and other papers using the same fold.

---

## Synthesis by Theme

**Theme 1 (Text-only):** Transformer-based models (BERT and domain-adapted variants like MentalBERT) are now the standard for text-based mental-health-related classification and consistently outperform earlier RNN/LSTM-based and classical approaches, but the field-wide systematic review (Cao et al., 2024) shows this strength is undercut by narrow data sources (Twitter/English-dominant), non-probability sampling, and inconsistent handling of class imbalance — meaning reported text-only accuracy numbers across the literature should be read with those caveats, and this project's own text-only baseline must be evaluated with imbalance-aware metrics and documented sampling limits rather than compared naively to headline numbers from other papers.

**Theme 2 (Image-only):** Pretrained-and-fine-tuned CNNs (e.g., Inception-V3) substantially outperform from-scratch CNNs or data-hungry architectures like ViT when the available image dataset is modest — directly shaping this project's choice to use a frozen or lightly fine-tuned pretrained CNN rather than training a Vision Transformer from scratch under CPU-only, limited-disk constraints. Grad-CAM is the literature's standard, architecture-agnostic tool for visualizing what the image branch is attending to, with the caveat that its explanations are coarse and layer-dependent. Most image-emotion literature located here is facial-expression-centric, not general social-media imagery (memes/scenes/objects) — a scope gap this project's image pipeline must account for when interpreting results.

**Theme 3 (Multimodal):** Two independent systematic surveys (Gandhi et al., 2023; Das & Singh, 2023) confirm a mature taxonomy of fusion strategies (early/feature-level, late/decision-level, attention-based/hybrid) and a general literature consensus that multimodal fusion *can* outperform unimodal baselines — but the strongest, most-cited multimodal sentiment benchmarks (CMU-MOSI/MOSEI, used by Jiang & Ji, 2022) are video+audio+text, not static social-media image+text, and the most directly comparable prior work this project's own team cited (Vasanthi & Viswanatham, 2026) tested its social-media (Twitter) condition on only 100 image-text pairs. The MVSA dataset family (Niu et al., 2016) is the literature's standard real, publicly documented, substantially-sized (5,129–19,600 pairs) Twitter image+text sentiment resource, making it the most defensible choice for this project's core text-vs-image-vs-multimodal comparison (full justification in `DATASET_SELECTION.md`).

---

## Verification Note
All 14 sources above were located via live web search and, where possible, directly fetched and cross-checked against the original/publisher page, PubMed/PMC, arXiv, ACL Anthology, or the paper's own public code repository (not reconstructed from memory). No DOI, page number, venue, or result figure in this document was invented; where a detail could not be confirmed (e.g., CMGA's exact benchmark scores), that is stated explicitly rather than estimated. Entry 3.7 (CLMLF) was added after the dataset-acquisition phase, once its code repository became this project's actual source for the MVSA-Single train/dev/test split and label mapping — not a hypothetical literature entry.
