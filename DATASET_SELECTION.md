# Dataset Selection

## Status of this document
**Update — data acquisition actually completed.** The planning-phase research below
(candidate comparison, original acquisition plan) is kept as-written for transparency,
with a correction appended at the end of the MVSA-Single section describing what
actually happened when the data pipeline was built: the originally planned OneDrive
link was dead (404), and a different, verified acquisition path was used instead. The
original research below was based on web research (official project pages, the
dataset's citing paper, Kaggle listing pages, secondary academic sources); the
**Actual acquisition** subsection reports what was verified by actually downloading
and inspecting the real files.

## Requirement recap
Need: real (not synthetic), paired **text + image** social-media content, usable sentiment/emotion labels, legitimate provenance, acceptable licensing/citation terms, ethically sourced (no private-account scraping, minimal PII exposure), and small enough to fit this project's hardware: **no GPU, ~11GB free disk at research time** (budgeted as if only 6–8GB is safely usable for data + Python environment, leaving headroom for installed libraries and model checkpoints).

## Candidates investigated

### 1. MVSA (MVSA-Single / MVSA-Multiple) — **selected (MVSA-Single)**
- **Authenticity/Provenance:** Real tweets (text + attached image) collected from Twitter, manually sentiment-annotated. Introduced by Niu, Zhu, Pang & El Saddik, *"Sentiment Analysis on Multi-View Social Data,"* MultiMedia Modeling (MMM) 2016, pp. 15–27 (also published as a Memorial University / City University of Hong Kong MCR Lab resource).
- **Original publication:** Peer-reviewed conference paper (MMM 2016), widely cited (100+ follow-up papers use MVSA as a standard benchmark for multimodal sentiment fusion — directly matches this project's text-vs-image-vs-multimodal comparison design).
- **Samples:** MVSA-Single is reported as **4,869** image-text pairs by the original authors; several downstream papers report **5,129** raw pairs or **4,511** pairs after removing low-agreement/invalid pairs during standard preprocessing. This discrepancy is a known, documented artifact of different papers' cleaning steps, not a sign of a different dataset — **the exact count used in this project will be reported after we load and clean the data ourselves**, not assumed in advance. MVSA-Multiple (3 annotators per tweet) has ~19,598 pairs; not selected as primary (adds label-aggregation complexity and is ~4x larger) but may be used later for a robustness check if time/disk permit.
- **Text availability:** Yes — original tweet text per pair.
- **Image availability:** Yes — one image per pair (MVSA-Single).
- **Label quality/semantics:** Single human annotator per tweet (MVSA-Single) assigns **one of {positive, neutral, negative}** sentiment to the pair as a whole (not separate text/image labels in the version we will use for the headline task). This is an overall-sentiment label, not a clinical or diagnostic label — consistent with this project's framing of "mental-health-related sentiment," not diagnosis.
- **Class distribution:** Reported in the literature as imbalanced (positive-leaning), to be confirmed empirically on load; macro-F1 and per-class metrics will be prioritized accordingly per `PROJECT_REQUIREMENTS_ANALYSIS.md`.
- **Missing modalities:** Pairs with missing/corrupt image or empty text are known to exist in redistributed copies and must be filtered during preprocessing (documented in `REPRODUCIBILITY.md` once implemented).
- **Licensing:** **No formal open-data license (e.g., no CC BY) is published on the official project page.** Usage is by academic custom: cite Niu et al. (2016) and contact the maintainer (Dr. Shiai Zhu) for issues. This is a genuine limitation — documented here rather than asserting a license that doesn't exist. Used here strictly for non-commercial academic research/coursework, with citation, consistent with how the dataset has been used in 100+ published papers.
- **Reproducibility:** Fixed dataset (not a live scrape); citation available; no single "official" train/val/test split is distributed, so this project will define and document its own fixed, seeded split (see `REPRODUCIBILITY.md`).
- **Ethical/privacy issues:** Tweets were public at time of collection (2016-era paper); still carries typical social-media dataset risks (usernames potentially inferable from image content, personal photos). Mitigation documented in `ETHICS_AND_PRIVACY.md` (no re-publication of raw content, no attempt to re-identify users, aggregate reporting only).
- **Suitability:** Strong — general social-media sentiment (not just memes), is the field's standard benchmark for exactly the text/image/multimodal fusion comparison this project is required to run, giving us real published numbers to sanity-check against.
- **Computational feasibility:** Small (thousands of tweet-sized images, each typically tens to a few hundred KB) — expected total well under 2GB, fits comfortably in the disk budget. To be confirmed at actual download time.
- **Acquisition path:**
  - Primary: official project page `https://mcrlab.net/research/mvsa-sentiment-analysis-on-multi-view-social-data/` → OneDrive link for MVSA-single: `https://portland-my.sharepoint.com/:u:/g/personal/shiaizhu2-c_my_cityu_edu_hk/Ebcsf1kUpL9Do_u4UfNh7CgBC19i6ldyYbDZwr6lVbkGQQ` (no account/DUA required to view; standard OneDrive share link).
  - Backup/cross-check: BaiduYun link on the same page (likely impractical to use outside China without a Baidu account — treat as fallback only).
  - A Kaggle mirror exists for **MVSA-Multiple** (`kaggle.com/datasets/vincemarcs/mvsamultiple`) if the OneDrive link becomes unavailable and the project needs to fall back to the multi-annotator variant (would require a label-aggregation step, e.g. majority vote, to be documented if used).
  - **Action for the data-pipeline phase:** attempt the OneDrive download first; if blocked (link rot, access restriction), fall back to Memotion 7k (candidate #2 below), which has a guaranteed, friction-free Kaggle path. This contingency must be recorded in `REPRODUCIBILITY.md` with whichever path actually worked.

### 2. Memotion Dataset 7k (SemEval-2020 Task 8) — strong alternative / fallback
- **Authenticity/Provenance:** Real, publicly-shared internet memes (image with overlaid text caption, OCR-extracted). Introduced by Sharma et al., *"SemEval-2020 Task 8: Memotion Analysis — the Visuo-Lingual Metaphor!"* (ACL Anthology, 2020.semeval-1.99).
- **Samples:** ~6,992 meme images with paired OCR text (reported as 6,990 train + 1,879 for test/trial across task phases in different write-ups; the Kaggle "7k" package is the commonly-used ~6,992-image release).
- **Label semantics (verbatim per task definition):** Task A — overall sentiment ∈ {positive, negative, neutral}. Task B — binary presence of {humour, sarcasm, offensive, motivational}. Task C — intensity/degree of the same four categories. **Important scoping note:** humour/sarcasm/offensive labels are about meme-genre properties, not mental-health-relevant affect — if this dataset is used, only the **overall sentiment** (Task A) and possibly **motivational** label would be used for this project's sentiment framing; humour/sarcasm/offensive are out of scope for the mental-health-adjacent framing and must not be repurposed as such.
- **Licensing:** Reported (secondary sources) as **CC BY 4.0** — to be confirmed directly on the Kaggle listing page at download time, and cited via the SemEval-2020 Task 8 paper regardless.
- **Computational feasibility:** ~743.8MB total — comfortably fits the disk budget, smallest of the serious candidates.
- **Acquisition path:** Kaggle, `kaggle.com/datasets/williamscott701/memotion-dataset-7k` — reliable, scriptable via the Kaggle API (`kaggle datasets download -d williamscott701/memotion-dataset-7k`), no OneDrive/BaiduYun friction.
- **Why not primary:** Weaker thematic fit — meme culture (humour/sarcasm-centric) is a narrower, more stylistically unusual genre of social media content than general posts, and the dataset's multimodal-fusion literature lineage (papers benchmarking general text+image sentiment fusion) is thinner than MVSA's. Kept as the designated fallback specifically because its acquisition is lower-risk (Kaggle API vs. an OneDrive link that could go stale).

### 3. B-T4SA / T4SA (Twitter for Sentiment Analysis) — rejected
- Large-scale (470,586 images in the balanced B-T4SA subset alone), weakly-labeled (labels derived from a text-sentiment teacher model, not human annotation) Twitter image+text dataset from `t4sa.it`.
- **Rejected on computational feasibility:** the image package alone is reported at **63GB** — more than 5x this project's entire disk budget. Even a small random subset would require first downloading the full 63GB archive (no partial-download API found), which is infeasible here. Also weaker label quality (weak/distant supervision vs. MVSA's human annotation) is a secondary reason.

### 4. CrisisMMD — rejected (wrong domain)
- Real, well-documented multimodal Twitter dataset (QCRI/CrisisNLP), ~16,058 tweets / ~18,082 images, from Alam et al., *"CrisisMMD: Multimodal Twitter Datasets from Natural Disasters"* (ICWSM 2018).
- **Rejected on suitability:** labels are disaster-response categories — "Informative/Not informative," humanitarian categories (affected individuals, infrastructure damage, rescue/volunteering, etc.) and damage severity. These describe crisis-response relevance, not emotional/sentiment/mental-health-adjacent content. Repurposing these labels as sentiment or mental-health proxies would misrepresent what the dataset's own authors measured, which this project's rules explicitly forbid. Also larger (~18K images) than needed given a better-fitting alternative exists.

### 5. Clinically-labeled mental-health text datasets (CLPsych shared-task data, eRisk, Reddit depression/self-harm corpora) — ruled out for this project's multimodal requirement
- These exist and are the most direct "mental health" label sources in the literature, but (a) they are almost all **text-only** (no paired images), failing the core multimodal requirement, and (b) the ones with the strongest clinical grounding are typically **DUA-gated** (CLPsych shared tasks require a signed data use agreement and task registration; not obtainable within this session). Noted here for the literature review's context (these are the datasets closest to true "mental health" labels) but not usable as this project's primary dataset given the multimodal + open-access constraints.

## Final decision
**Primary dataset: MVSA-Single** (≈4,869–5,129 Twitter text-image pairs, 3-class sentiment: positive/neutral/negative), with **Memotion Dataset 7k** as the documented fallback if MVSA's OneDrive distribution is inaccessible when the data pipeline is actually built.

## Actual acquisition (what really happened — verified, not planned)

The official OneDrive link (`portland-my.sharepoint.com/.../Ebcsf1kUpL9Do...`) returned
**404 Not Found** when actually fetched (confirmed via direct HTTP request, both with
and without a `download=1` suffix) — the predicted link-rot risk materialized. Rather
than falling back to the weaker-fit Memotion dataset, two independently-citable
real sources were located, verified, and combined:

1. **Images + original tweet text (full 4,869 pairs):** HuggingFace Hub dataset
   `xwycyj/MVSA-Single` (https://huggingface.co/datasets/xwycyj/MVSA-Single), a
   community re-upload of the same Niu et al. (2016) MVSA-Single data — verified by
   downloading `data.zip` (211.2 MB) and confirming it contains exactly **4,869**
   `<id>.jpg` + `<id>.txt` pairs, matching the original paper's reported count exactly.
2. **Labels + a published train/dev/test split (4,511 of the 4,869 pairs):**
   GitHub repo `Link-Li/CLMLF` (https://github.com/Link-Li/CLMLF), the official code
   release for **Li, Z., Xu, B., Zhu, C., & Zhao, T. (2022). "CLMLF: A Contrastive
   Learning and Multi-Layer Fusion Method for Multimodal Sentiment Detection."
   Findings of the ACL: NAACL 2022, pp. 2282–2294** (verified via ACL Anthology,
   `aclanthology.org/2022.findings-naacl.175`). Their `dataset/data/MVSA-single/
   10-flod-1/{train,dev,test}.json` (fold 1 of a published 10-fold split) gives
   3,611 / 450 / 450 labeled samples (4,511 total) — matching the commonly-cited
   "4,511 pairs after removing low-agreement pairs" cleaning convention for
   MVSA-Single noted in the planning research above. **Zero missing files**: every
   one of the 4,511 labeled ids was verified present in the HuggingFace mirror's
   image+text archive before building the manifest.
3. **Label semantics — empirically verified, not assumed:** labels are integers
   0/1/2. Distribution across all 4,511 samples: `0: 2,683`, `2: 1,358`, `1: 470`.
   Manually inspecting sample tweet texts per label (see `REPRODUCIBILITY.md`)
   confirms **0 = positive, 1 = neutral, 2 = negative** — label-0 examples read
   positive ("fun loving and energetic team...Thank you and I love you"), label-2
   examples read negative (`#Depressed`, "helpless"), label-1 examples are
   informational/neutral in tone. This matches the standard MVSA-Single convention
   used across the citing literature, confirmed here rather than taken on faith.
4. **Provenance caveat (disclosed, not hidden):** the images+text now come through a
   third-party HuggingFace re-upload rather than the original authors' own
   distribution channel, since that channel is dead. The *content* was verified
   (exact pair count match, label-semantic spot check, real Twitter-style text),
   but this is one more hop in the provenance chain than originally planned — noted
   here and in `ETHICS_AND_PRIVACY.md`. The dataset is still cited primarily as
   Niu, Zhu, Pang & El Saddik (MMM 2016); the HuggingFace mirror and the CLMLF
   repo/paper are cited as the actual technical source of the files used.
5. **Final processed artifact:** `data/processed/mvsa_single_manifest.csv` —
   columns `id, text, image_path, label, label_int, split` — built by
   `src/data/build_mvsa_manifest.py`, which performs the id-coverage check and
   writes this file. Class distribution confirms the predicted imbalance (positive
   majority): train `{positive: 2147, negative: 1088, neutral: 376}`, val/test each
   `{positive: 268, negative: 135, neutral: 47}` — carried forward into
   `PROJECT_ARCHITECTURE.md`'s macro-F1-first evaluation choice.

**Memotion 7k fallback was not needed and was not downloaded.**

## Key limitation to carry forward
Neither candidate has labels that say "mental health" — both have **sentiment/emotion labels** (positive/neutral/negative, or humour/sarcasm/motivational/offensive). This project treats sentiment as a **proxy signal relevant to mental-health-related monitoring**, exactly as the synopsis itself frames it ("mental-health-related sentiment," never a diagnosis). Model outputs will be reported using the dataset's own label names (e.g., "negative sentiment"), never relabeled as a clinical condition (e.g., never output "depression" or "anxiety" as a class name) — this boundary is mandatory per `PROJECT_REQUIREMENTS_ANALYSIS.md` and must be enforced in code, UI, and all documentation.
