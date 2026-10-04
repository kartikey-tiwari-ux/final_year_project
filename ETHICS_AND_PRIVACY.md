# Ethics and Privacy

Scope: this document covers the ethical and privacy considerations for the selected primary dataset (**MVSA-Single**, with **Memotion Dataset 7k** as fallback — see `DATASET_SELECTION.md`), and for the project as a whole.

## 1. Privacy

- **Source of data:** MVSA-Single consists of tweets that were **public** at the time of collection (dataset introduced 2016). Memotion 7k consists of **publicly shared internet memes**. Neither this project nor the original dataset authors scraped private accounts, private messages, or access-restricted content.
- **PII exposure risk:** Tweet text and images can still contain incidental personal information — usernames mentioned in text, faces in photos, recognizable locations, etc. — even though the dataset authors did not collect from private accounts.
- **This project's mitigations:**
  - No attempt will be made to re-identify, look up, or contact any individual whose post appears in the dataset.
  - No raw dataset content (verbatim tweet text, original images) will be redistributed outside the dataset's own terms; the project's own repository will not re-host the raw dataset files (only derived, aggregate artifacts — e.g., trained model weights, aggregate metrics, small illustrative examples used strictly for explainability figures in the report, consistent with academic fair-use/citation norms).
  - Any example post shown in the report, paper, slides, or demo app (e.g., to illustrate a correct/incorrect prediction) will be chosen to avoid displaying identifying personal details where reasonably avoidable, and will be presented purely as a research illustration, not as commentary on a real identifiable person's mental state.
  - The demonstration application (see `PROJECT_ARCHITECTURE.md` / `app/`) accepts **user-supplied text/image input for demo purposes only**, processes it in-memory for the single prediction request, and must not log, store, or transmit user-submitted demo input anywhere beyond that single request/response cycle.

## 2. Consent

- Neither MVSA nor Memotion involved individualized informed consent from each post's author (standard for public-social-media research datasets built from already-public posts, consistent with how such datasets have been used in 100+ published academic papers). This is a known, accepted limitation of public-social-media-derived research datasets and is disclosed here rather than glossed over.
- Because of this, **this project's outputs are research artifacts about aggregate patterns in a benchmark dataset — not statements about any specific identifiable person.**

## 3. Licensing

- **MVSA:** No formal open-data license (e.g., CC BY) is published by the dataset maintainers. Usage convention (confirmed via the official project page) is: cite the originating paper (Niu, Zhu, Pang & El Saddik, MMM 2016) and contact the maintainer with issues. This project will use MVSA strictly for non-commercial academic coursework/research, with citation, consistent with its established use in the published literature. If any ambiguity arises about permitted use, this project defaults to the most conservative interpretation (no redistribution of raw data, citation given, non-commercial use only).
- **Provenance update (disclosed):** the official OneDrive distribution was found dead (404) when this project actually acquired the data. The images/text used were instead obtained from a third-party HuggingFace re-upload (`xwycyj/MVSA-Single`), content-verified against the original paper's reported sample count (4,869 — exact match) before use; labels and the train/dev/test split came from the public code release of a separate peer-reviewed paper (CLMLF, Findings of ACL: NAACL 2022) that itself uses MVSA-Single under the same academic-citation convention. This project does not re-host or redistribute these files — see `DATASET_SELECTION.md` "Actual acquisition" for the full chain. The same non-commercial, citation-only, no-redistribution posture applies regardless of which specific host served the files.
- **Memotion 7k (fallback):** Reported as CC BY 4.0 (to be re-confirmed on the live Kaggle page at download time); requires attribution to the SemEval-2020 Task 8 paper (Sharma et al., 2020) regardless of exact license text.
- Both: this project will cite the dataset's originating paper in `REFERENCES.md`, `research_paper/references.bib`, and the final report, as required by `ORIGINALITY.md`.

## 4. Potential harm, bias, and representation

- **Language/cultural bias:** Both candidate datasets are predominantly English-language, Western-platform (Twitter) social media content from a specific collection period (MVSA: ~2015-16; Memotion: ~2020). Models trained on them will **not generalize** to other languages, cultures, dialects, or more recent internet slang/meme conventions without further validation — this must be stated as an explicit limitation in the report and paper, not implied away.
- **Annotator bias:** Sentiment labels reflect the subjective judgment of the dataset's own human annotators (one annotator per tweet for MVSA-Single), not a clinical consensus or validated psychological instrument. Downstream model outputs inherit this subjectivity.
- **Platform/demographic skew:** Twitter and meme-sharing userbases are not representative of the general population (skew toward particular age groups, internet-literacy levels, and self-selection into public posting). Any claim about "social media sentiment" must be scoped to "sentiment as expressed in this specific dataset," not generalized to all people or all mental-health presentations.
- **Class imbalance:** Sentiment classes are expected to be imbalanced (positive-leaning, per prior literature on MVSA); this risks a model that performs poorly on minority classes (often the more "concerning" sentiment categories, e.g. negative/sad), which is precisely where errors matter most for a mental-health-adjacent use case. This motivates prioritizing macro-F1 and per-class metrics over raw accuracy, as already required in `PROJECT_REQUIREMENTS_ANALYSIS.md`.

## 5. False positives / false negatives in a mental-health-adjacent framing

- **False positive** (model flags "negative/concerning sentiment" when the post isn't actually concerning): risk of mislabeling ordinary content, potentially stigmatizing or causing unwarranted concern if ever shown to a real person — mitigated here because this project does **not** deploy to real users or real accounts; it only evaluates on held-out dataset samples.
- **False negative** (model misses genuinely concerning sentiment): in a real deployment this would be the more serious failure mode (missed support opportunity); this project's non-diagnostic framing explicitly prevents it from being used as a sole signal for any real intervention decision.
- Both error types will be reported transparently via confusion matrices and per-class metrics (`PROJECT_REQUIREMENTS_ANALYSIS.md` evaluation requirements) rather than hidden behind an aggregate accuracy figure.

## 6. Misuse potential and stigma risk

- A system that labels social-media content with sentiment/emotion categories could, if misapplied, be used to profile, surveil, or stigmatize individuals (e.g., an employer or acquaintance screening someone's public posts). This project exists **only as an academic research/coursework artifact**: it is not deployed against real, identifiable individuals' live accounts, is not offered as a monitoring service, and is not validated for any real screening/intervention use.
- The non-clinical boundary is enforced throughout: the model's output vocabulary is restricted to the dataset's own sentiment/emotion label names (e.g., "negative," "neutral," "positive," or Memotion's humour/sarcasm/motivational/offensive labels where used) — **never** clinical terms like "depression," "anxiety disorder," or "at risk," which the dataset was never labeled for and which only a qualified professional can assess.

## 7. Responsible deployment notes

- This project is a **research/monitoring-support demonstration**, not a product. The companion application (`app/`) must display a clear, persistent disclaimer: *"This tool analyses sentiment/emotion patterns in text and images for research and educational purposes. It is not a diagnostic tool and cannot identify mental health conditions. It is not a substitute for professional mental health assessment or care."*
- No real-time scraping of live social media accounts is implemented anywhere in this project; all experiments run on the static, pre-collected benchmark dataset.
- Any future extension toward real deployment (explicitly out of scope here, per `PROJECT_REQUIREMENTS_ANALYSIS.md`'s scope section) would require: informed consent from monitored individuals, clinical validation with mental-health professionals, bias/fairness auditing across languages and demographics, a human-in-the-loop review process, and ethics board approval — none of which this coursework project attempts to provide.

## 8. Summary table

| Concern | Status for MVSA-Single / Memotion 7k |
|---|---|
| Private-account scraping | None — both sourced from already-public posts |
| PII in raw data | Possible (usernames/faces incidental to public posts) — not re-exposed by this project beyond standard academic citation/illustration use |
| Individual informed consent | Not obtained by original authors (standard limitation for public-social-media research datasets) — disclosed, not hidden |
| Formal open license | MVSA: none published (cite + contact-author convention). Memotion: reported CC BY 4.0 (verify at download) |
| Demographic/language bias | Present (English, Twitter/meme-culture skew, dated collection period) — disclosed as a generalization limitation |
| Clinical-label risk | Mitigated — labels used exactly as the dataset authors defined them; never relabeled as diagnoses |
| Real-world deployment risk | None in this project's scope — static benchmark evaluation only, with a mandatory non-diagnostic disclaimer in the demo app |
