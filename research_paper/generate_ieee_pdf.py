"""
Generates an IEEE two-column conference-paper-format PDF from the project's
research_paper/paper.md content. Run with the project's venv:

    ./venv/Scripts/python.exe research_paper/generate_ieee_pdf.py

Output: research_paper/paper_IEEE.pdf
"""
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame, FrameBreak, NextPageTemplate,
    Paragraph, Spacer, Table, TableStyle, KeepTogether,
)
from reportlab.lib import colors

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "paper_IEEE.pdf"

PAGE_W, PAGE_H = letter
MARGIN_L = MARGIN_R = 0.75 * inch
MARGIN_TOP = 0.75 * inch
MARGIN_BOTTOM = 0.75 * inch
COL_GAP = 0.25 * inch

USABLE_W = PAGE_W - MARGIN_L - MARGIN_R
COL_W = (USABLE_W - COL_GAP) / 2.0
COL1_X = MARGIN_L
COL2_X = MARGIN_L + COL_W + COL_GAP

HEADER_H = 2.55 * inch
FULL_H = PAGE_H - MARGIN_TOP - MARGIN_BOTTOM
COL_H_P1 = FULL_H - HEADER_H

# ---------------------------------------------------------------- styles ---
S_TITLE = ParagraphStyle(
    "Title", fontName="Times-Bold", fontSize=20, leading=24,
    alignment=TA_CENTER, spaceAfter=10,
)
S_AUTHORS = ParagraphStyle(
    "Authors", fontName="Times-Roman", fontSize=11, leading=13,
    alignment=TA_CENTER, spaceAfter=2,
)
S_AFFIL = ParagraphStyle(
    "Affil", fontName="Times-Italic", fontSize=10, leading=12,
    alignment=TA_CENTER, spaceAfter=2,
)
S_LEAD = ParagraphStyle(
    "Lead", fontName="Times-Roman", fontSize=9, leading=11,
    alignment=TA_JUSTIFY, spaceAfter=8,
)
S_HEAD = ParagraphStyle(
    "Head", fontName="Times-Bold", fontSize=10, leading=12,
    alignment=TA_CENTER, spaceBefore=10, spaceAfter=6,
)
S_SUBHEAD = ParagraphStyle(
    "SubHead", fontName="Times-Italic", fontSize=10, leading=12,
    alignment=TA_LEFT, spaceBefore=6, spaceAfter=3,
)
S_BODY = ParagraphStyle(
    "Body", fontName="Times-Roman", fontSize=10, leading=12.5,
    alignment=TA_JUSTIFY, firstLineIndent=12, spaceAfter=0,
)
S_CAPTION = ParagraphStyle(
    "Caption", fontName="Times-Roman", fontSize=8, leading=10,
    alignment=TA_CENTER, spaceBefore=6, spaceAfter=4,
)
S_TABLECELL = ParagraphStyle(
    "TableCell", fontName="Times-Roman", fontSize=7, leading=8.5,
    alignment=TA_LEFT,
)
S_TABLECELL_C = ParagraphStyle(
    "TableCellC", fontName="Times-Roman", fontSize=7, leading=8.5,
    alignment=TA_CENTER,
)
S_TABLECELL_B = ParagraphStyle(
    "TableCellB", fontName="Times-Bold", fontSize=7, leading=8.5,
    alignment=TA_CENTER,
)
S_REF = ParagraphStyle(
    "Ref", fontName="Times-Roman", fontSize=9, leading=10.5,
    alignment=TA_JUSTIFY, leftIndent=12, firstLineIndent=-12,
)

# -------------------------------------------------------------- frames -----
header_frame = Frame(MARGIN_L, PAGE_H - MARGIN_TOP - HEADER_H, USABLE_W, HEADER_H,
                      id="header", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
col1_p1 = Frame(COL1_X, MARGIN_BOTTOM, COL_W, COL_H_P1, id="col1p1",
                 leftPadding=0, rightPadding=6, topPadding=0, bottomPadding=0)
col2_p1 = Frame(COL2_X, MARGIN_BOTTOM, COL_W, COL_H_P1, id="col2p1",
                 leftPadding=6, rightPadding=0, topPadding=0, bottomPadding=0)

col1_later = Frame(COL1_X, MARGIN_BOTTOM, COL_W, FULL_H, id="col1later",
                    leftPadding=0, rightPadding=6, topPadding=0, bottomPadding=0)
col2_later = Frame(COL2_X, MARGIN_BOTTOM, COL_W, FULL_H, id="col2later",
                    leftPadding=6, rightPadding=0, topPadding=0, bottomPadding=0)


def draw_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Times-Roman", 8)
    canvas.drawCentredString(PAGE_W / 2.0, MARGIN_BOTTOM - 16, str(doc.page))
    canvas.restoreState()


doc = BaseDocTemplate(
    str(OUT), pagesize=letter,
    leftMargin=MARGIN_L, rightMargin=MARGIN_R, topMargin=MARGIN_TOP, bottomMargin=MARGIN_BOTTOM,
    title="AI-Driven Multimodal Sentiment Analysis for Mental Health Monitoring on Social Media",
    author="Kartikey Tiwari, Akshat Sharma, Karan Sharma, Lalit Sharma",
)
doc.addPageTemplates([
    PageTemplate(id="First", frames=[header_frame, col1_p1, col2_p1], onPage=draw_page_number),
    PageTemplate(id="Later", frames=[col1_later, col2_later], onPage=draw_page_number),
])

story = []

# ------------------------------------------------------------- header ------
story.append(Paragraph(
    "AI-Driven Multimodal Sentiment Analysis for Mental Health Monitoring on "
    "Social Media: A Resource-Constrained Comparative Study", S_TITLE))
story.append(Paragraph("Kartikey Tiwari, Akshat Sharma, Karan Sharma, Lalit Sharma", S_AUTHORS))
story.append(Paragraph("Department of Computer Science and Engineering", S_AFFIL))
story.append(Paragraph("Ajay Kumar Garg Engineering College, Ghaziabad", S_AFFIL))
story.append(Paragraph("Dr. A.P.J. Abdul Kalam Technical University, Lucknow", S_AFFIL))
story.append(Paragraph("<i>Supervisor: Dr. Avdhesh Gupta</i>", S_AFFIL))

story.append(NextPageTemplate("Later"))
story.append(FrameBreak())

# -------------------------------------------------------------- body -------

ABSTRACT = (
    "<b><i>Abstract&mdash;</i></b>Social media posts often combine text and images, but "
    "most sentiment-analysis systems for mental-health-related monitoring rely on text "
    "alone, potentially losing complementary emotional signal carried by attached images. "
    "This paper reports a controlled, same-dataset, same-metrics comparison of text-only, "
    "image-only, and multimodal (text+image) sentiment classification on MVSA-Single, a "
    "real, publicly documented Twitter image-text sentiment dataset (4,511 labeled pairs "
    "after standard cleaning; 3-class: positive/neutral/negative). We evaluate a "
    "traditional TF-IDF + Logistic Regression text baseline, a frozen-DistilBERT text "
    "model, a CNN trained from scratch, and a frozen-ResNet18 image model, together with "
    "two multimodal fusion strategies (feature-concatenation and late/decision-level "
    "fusion). The best multimodal model (feature-concatenation fusion) achieves 64.4% "
    "test accuracy and 56.3% macro-F1, outperforming the best unimodal model (frozen "
    "DistilBERT: 61.6% accuracy, 54.6% macro-F1) and the image-only model (52.7% "
    "accuracy, 42.8% macro-F1). A modality-conflict analysis shows the gain concentrates "
    "specifically on the 53.3% of test posts where text and image models disagree "
    "(fusion: 57.1% accuracy on this subset vs. 50.0% for text-only and 33.3% for "
    "image-only), with a small regression on posts where modalities already agree. We "
    "report this work entirely from real experiments run on constrained consumer CPU "
    "hardware (no GPU, 8GB RAM), documenting the resource constraints encountered and the "
    "engineering decisions made in response, as a contribution to resource-constrained "
    "reproducibility in this research area. Consistent with the non-clinical scope of "
    "this study, all outputs are the dataset's own sentiment labels, not diagnostic "
    "claims."
)
story.append(Paragraph(ABSTRACT, S_LEAD))

INDEX_TERMS = (
    "<b><i>Index Terms&mdash;</i></b>Multimodal sentiment analysis, mental health "
    "monitoring, social media analytics, natural language processing, deep learning, "
    "DistilBERT, ResNet18, feature fusion, MVSA, resource-constrained machine learning."
)
story.append(Paragraph(INDEX_TERMS, S_LEAD))

# I. INTRODUCTION
story.append(Paragraph("I. INTRODUCTION", S_HEAD))
story.append(Paragraph(
    "Text-only or image-only sentiment analysis of social media posts each miss "
    "information that the other modality carries; the full background and motivation, "
    "derived from the originating project synopsis, is provided in the accompanying "
    "technical report (PROJECT_REQUIREMENTS_ANALYSIS.md). This project investigates "
    "whether, and under what conditions, combining text and image information improves "
    "sentiment-based classification, framed explicitly as mental-health-related sentiment "
    "monitoring support &mdash; not clinical diagnosis (see Section VIII).", S_BODY))

# II. RELATED WORK
story.append(Paragraph("II. RELATED WORK", S_HEAD))
story.append(Paragraph(
    "A full annotated review of 14 verified sources is provided in the accompanying "
    "literature review (LITERATURE_REVIEW.md). Key threads: text-based mental-health "
    "classification increasingly uses domain-adapted transformers such as MentalBERT "
    "[1], but a 2024 systematic review [2] documents widespread representativeness and "
    "evaluation-methodology issues across this literature. Image-emotion recognition "
    "benefits substantially from transfer learning over from-scratch training on modest "
    "datasets [3]. Multimodal fusion has a mature taxonomy [4], [5], but &mdash; per our "
    "research-gap analysis &mdash; lacks a controlled, adequately sized, same-metrics "
    "text-vs-image-vs-multimodal comparison specifically on real social-media data; the "
    "closest located prior work [6] evaluates its social-media condition on only 100 "
    "pairs.", S_BODY))

# III. RESEARCH GAP AND QUESTIONS
story.append(Paragraph("III. RESEARCH GAP AND QUESTIONS", S_HEAD))
story.append(Paragraph(
    "A full analysis is given in RESEARCH_GAP.md. The primary research question (from "
    "PROJECT_REQUIREMENTS_ANALYSIS.md) is: <i>can an AI-driven multimodal framework "
    "combining textual and visual information improve sentiment-based mental health "
    "monitoring on social media compared with unimodal approaches?</i>", S_BODY))

# IV. DATASET
story.append(Paragraph("IV. DATASET", S_HEAD))
story.append(Paragraph(
    "MVSA-Single [7] consists of real Twitter image-text pairs with human-annotated "
    "3-class sentiment labels (positive/neutral/negative). This project uses the "
    "published train/dev/test split from CLMLF [8] &mdash; 3,611/450/450 samples "
    "&mdash; for comparability with that and other work using the same fold, rather "
    "than an arbitrary self-computed split. Images and text were obtained via a "
    "HuggingFace community mirror after the original OneDrive distribution proved "
    "unavailable (HTTP 404) at acquisition time; the full provenance chain, "
    "label-semantics verification (empirically confirmed as 0 = positive, 1 = neutral, "
    "2 = negative by inspecting sample texts per label), and licensing caveats are "
    "documented in DATASET_SELECTION.md. The training-set class distribution is "
    "positive 2,147 (59.5%), negative 1,088 (30.1%), and neutral 376 (10.4%) &mdash; a "
    "real, measured imbalance that motivates macro-F1 as the primary evaluation metric. "
    "A duplicate-hash leakage audit found a small amount of cross-split duplication "
    "inherited from the published split itself (56/4,511 texts and 10/4,511 images, "
    "arising from Twitter retweets); this is disclosed in REPRODUCIBILITY.md as a "
    "limitation of using a third-party split and is assessed as unlikely to materially "
    "affect results at this scale.", S_BODY))

# V. METHODOLOGY
story.append(Paragraph("V. METHODOLOGY", S_HEAD))
story.append(Paragraph(
    "The full architecture rationale is given in PROJECT_ARCHITECTURE.md. All models "
    "share the same train/val/test split and evaluation code "
    "(src/evaluation/metrics.py).", S_BODY))

story.append(Paragraph("A. Text Pipeline", S_SUBHEAD))
story.append(Paragraph(
    "Cleaning (URL/mention normalization, emoji-to-text demojizing, lowercasing), "
    "followed by either (A1) TF-IDF (1&ndash;2 grams, 10k features) with class-weighted "
    "Logistic Regression, or (A2) frozen DistilBERT (distilbert-base-uncased) "
    "mean-pooled embeddings with class-weighted Logistic Regression.", S_BODY))

story.append(Paragraph("B. Image Pipeline", S_SUBHEAD))
story.append(Paragraph(
    "Resize and normalize, followed by either (B1) a small 4-layer CNN trained from "
    "scratch, or (B2) frozen ImageNet-pretrained ResNet18 penultimate-layer embeddings "
    "with class-weighted Logistic Regression.", S_BODY))

story.append(Paragraph("C. Multimodal Fusion", S_SUBHEAD))
story.append(Paragraph(
    "Primary strategy (C): feature-level concatenation of the A2 text embedding "
    "(768-dim) and B2 image embedding (512-dim), fed into a 2-hidden-layer MLP with "
    "dropout, trained jointly with class-weighted cross-entropy loss and selecting the "
    "best checkpoint by validation macro-F1. Ablation: late (decision-level) fusion, in "
    "which independent A2/B2 classifiers' predicted probabilities are combined by "
    "simple averaging and by a validation-tuned weighted average.", S_BODY))

story.append(Paragraph("D. Why Frozen Pretrained Encoders", S_SUBHEAD))
story.append(Paragraph(
    "This choice is justified directly by this project's measured hardware constraints "
    "(CPU-only, 8GB RAM) and by the literature (Section II) &mdash; transfer learning "
    "with a frozen or lightly adapted encoder is both the literature-supported choice "
    "for modest dataset sizes and the only practically tractable choice on this "
    "hardware. REPRODUCIBILITY.md reports measured per-sample embedding-extraction "
    "timing (DistilBERT: 49.5 ms/sample; ResNet18: 536 ms/sample on first pass, "
    "dropping to a few ms/sample on a cache-warmed second pass).", S_BODY))

# VI. EXPERIMENTAL SETUP
story.append(Paragraph("VI. EXPERIMENTAL SETUP", S_HEAD))
story.append(Paragraph(
    "A fixed seed (42) is used throughout (src/utils/seed.py). Evaluation metrics are "
    "accuracy, precision/recall/F1 (macro and weighted), and the confusion matrix; "
    "macro-F1 is prioritized given the class imbalance described in Section IV. All "
    "numbers reported in Section VII are read directly from results/*.json, generated "
    "by the scripts under src/text_model/, src/image_model/, and src/multimodal/.",
    S_BODY))

# VII. RESULTS
story.append(Paragraph("VII. RESULTS", S_HEAD))

table_header = [Paragraph(h, S_TABLECELL_B) for h in ["Model", "Acc.", "Macro-F1", "Wtd-F1"]]
rows_data = [
    ("A1 &mdash; TF-IDF + LogReg", "60.9%", "52.0%", "61.9%", False),
    ("A2 &mdash; Frozen DistilBERT + LogReg", "61.6%", "54.6%", "63.5%", False),
    ("B1 &mdash; CNN from scratch", "47.6%", "42.9%", "49.6%", False),
    ("B2 &mdash; Frozen ResNet18 + LogReg", "52.7%", "42.8%", "54.3%", False),
    ("C &mdash; Concat Fusion", "64.4%", "56.3%", "65.4%", True),
    ("C &mdash; Late Fusion (tuned)", "64.0%", "56.4%", "65.6%", False),
]
table_rows = [table_header]
for name, acc, f1m, f1w, best in rows_data:
    style = S_TABLECELL_B if best else S_TABLECELL
    style_c = S_TABLECELL_B if best else S_TABLECELL_C
    table_rows.append([
        Paragraph(name, style), Paragraph(acc, style_c),
        Paragraph(f1m, style_c), Paragraph(f1w, style_c),
    ])

results_table = Table(table_rows, colWidths=[COL_W * 0.52, COL_W * 0.16, COL_W * 0.16, COL_W * 0.16])
results_table.setStyle(TableStyle([
    ("LINEABOVE", (0, 0), (-1, 0), 1.0, colors.black),
    ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
    ("LINEBELOW", (0, -1), (-1, -1), 1.0, colors.black),
    ("TOPPADDING", (0, 0), (-1, -1), 2),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ("LEFTPADDING", (0, 0), (-1, -1), 2),
    ("RIGHTPADDING", (0, 0), (-1, -1), 2),
]))

story.append(KeepTogether([
    Paragraph("TABLE I<br/>RESULTS: TEST-SET PERFORMANCE BY MODEL", S_CAPTION),
    results_table,
]))

story.append(Spacer(1, 6))
story.append(Paragraph(
    "Full confusion matrices and a per-class breakdown are given in "
    "ERROR_ANALYSIS.md. Notably, B1 (trained from scratch, at a reduced 96&times;96 "
    "resolution and 3 epochs due to measured CPU-only training-time constraints "
    "&mdash; see Section V and REPRODUCIBILITY.md) scores close to B2 on macro-F1 "
    "(42.9% vs. 42.8%) despite a clearly lower accuracy (47.6% vs. 52.7%). This is an "
    "interesting nuance: B1's training instability (its validation macro-F1 dropped "
    "between epochs 1 and 2 before recovering by epoch 3 &mdash; a real, reported "
    "instability, not smoothed over) suggests its best-epoch macro-F1 may partly "
    "reflect noisy, under-converged training rather than a robust feature-learning "
    "advantage. This result is consistent with, not contradictory to, the "
    "literature-grounded expectation (Sections II, V) that frozen transfer-learned "
    "features are the more reliable choice under this project's real hardware "
    "constraints &mdash; B2 achieves comparable or better results with a fraction of "
    "the training instability and compute.", S_BODY))
story.append(Paragraph(
    "Model comparison answers the primary research question affirmatively, but with "
    "nuance rather than a blanket win: multimodal fusion outperforms every unimodal "
    "model on every aggregate metric, but the improvement is not uniform across "
    "classes or subsets &mdash; see Section VII-A.", S_BODY))

story.append(Paragraph("A. Modality Conflict Analysis", S_SUBHEAD))
story.append(Paragraph(
    "Full detail is given in MODALITY_CONFLICT_ANALYSIS.md. Text and image models "
    "disagree on 53.3% of test posts (240/450). On exactly this subset, fusion clearly "
    "outperforms both unimodal models (57.1% vs. 50.0% text-only, 33.3% image-only) "
    "&mdash; the clearest evidence that fusion helps specifically by resolving genuine "
    "cross-modal disagreement, not just by averaging noise. On the 46.7% of posts where "
    "modalities already agree, fusion shows a small regression versus text-only (72.9% "
    "vs. 74.8%). Per class, fusion helps substantially on the majority (positive) class "
    "but underperforms text-only specifically on the minority neutral class (42.6% vs. "
    "48.9%) &mdash; attributed to neutral's severe data scarcity (10.4% of training "
    "data), leaving little signal for the fusion head to learn a reliable joint "
    "boundary.", S_BODY))

story.append(Paragraph("B. Ablation: Fusion Strategy", S_SUBHEAD))
story.append(Paragraph(
    "Feature-concatenation (early/joint) fusion and validation-tuned late "
    "(decision-level) fusion perform almost identically (64.4%/56.3% vs. 64.0%/56.4% "
    "accuracy/macro-F1) &mdash; neither strategy dominates the other on this "
    "dataset/scale, and both clearly beat unimodal models. The late-fusion weight "
    "search independently converges on w_text = 0.80, corroborating text's stronger "
    "standalone performance (Section VII).", S_BODY))

# VIII. DISCUSSION
story.append(Paragraph("VIII. DISCUSSION, ETHICS, AND LIMITATIONS", S_HEAD))
story.append(Paragraph("A. Non-Clinical Scope", S_SUBHEAD))
story.append(Paragraph(
    "All model outputs are the dataset's own sentiment labels (positive/neutral/"
    "negative) and are never relabeled as clinical categories. ETHICS_AND_PRIVACY.md "
    "gives the full discussion of privacy, consent, licensing, bias, and misuse-risk "
    "considerations specific to this dataset and application framing.", S_BODY))
story.append(Paragraph("B. Key Limitations", S_SUBHEAD))
story.append(Paragraph(
    "(1) MVSA-Single's labels are general sentiment, not mental-health-specific "
    "&mdash; a documented, unavoidable scoping limitation given that no open, "
    "legitimately licensed multimodal mental-health dataset was found (Section IV, "
    "DATASET_SELECTION.md). (2) The dataset is English-language, 2015&ndash;16-era "
    "Twitter content; results do not generalize to other languages, platforms, or eras "
    "without further validation. (3) A small (approximately 1&ndash;2%) cross-split "
    "duplication exists in the published split used. (4) CPU-only compute constraints "
    "limited the from-scratch CNN baseline's training budget and resolution &mdash; "
    "itself a finding relevant to resource-constrained reproducibility (Section V, "
    "REPRODUCIBILITY.md), not just a caveat. None of these limitations are minimized.",
    S_BODY))

# IX. CONCLUSION
story.append(Paragraph("IX. CONCLUSION AND FUTURE WORK", S_HEAD))
story.append(Paragraph(
    "Within a real, resource-constrained CPU-only environment, this project provides "
    "genuine, not assumed, evidence that multimodal fusion improves sentiment "
    "classification on real social-media image-text pairs &mdash; concentrated "
    "specifically on posts where modalities disagree, with measurable trade-offs "
    "elsewhere. Future work, explicitly out of scope here per "
    "PROJECT_REQUIREMENTS_ANALYSIS.md, includes multilingual extension, cross-modal "
    "attention fusion at scale, and validation against a genuinely mental-health-"
    "labeled multimodal dataset if and when one becomes openly available.", S_BODY))

# ACKNOWLEDGMENT
story.append(Paragraph("ACKNOWLEDGMENT", S_HEAD))
story.append(Paragraph(
    "Dataset: Niu, Zhu, Pang, and El Saddik (2016) [7]; split and labels: Li, Xu, Zhu, "
    "and Zhao (2022, CLMLF) [8]. Pretrained models: DistilBERT and ResNet18, accessed "
    "via Hugging Face Transformers and torchvision, respectively.", S_BODY))

# REFERENCES
story.append(Paragraph("REFERENCES", S_HEAD))
references = [
    "S. Ji, T. Zhang, L. Ansari, J. Fu, P. Tiwari, and E. Cambria, \"MentalBERT: "
    "Publicly available pretrained language models for mental healthcare,\" in "
    "<i>Proc. 13th Conf. Language Resources and Evaluation (LREC)</i>, 2022.",

    "Y. Cao, J. Dai, Z. Wang, Y. Zhang, X. Shen, Y. Liu, and Y. Tian, \"Machine "
    "learning approaches for mental illness detection on social media: A systematic "
    "review of biases and methodological challenges,\" <i>arXiv preprint "
    "arXiv:2410.16204</i>, 2024.",

    "E. S. Agung, A. P. Rifai, and T. Wijayanto, \"Image-based facial emotion "
    "recognition using convolutional neural network on Emognition dataset,\" "
    "<i>Scientific Reports</i>, vol. 14, 2024, doi: 10.1038/s41598-024-65276-x.",

    "A. Gandhi, K. Adhvaryu, S. Poria, E. Cambria, and A. Hussain, \"Multimodal "
    "sentiment analysis: A systematic review of history, datasets, multimodal fusion "
    "methods, applications, challenges and future directions,\" <i>Information "
    "Fusion</i>, vol. 91, pp. 424&ndash;444, 2023.",

    "R. Das and T. D. Singh, \"Multimodal sentiment analysis: A survey of methods, "
    "trends, and challenges,\" <i>ACM Computing Surveys</i>, vol. 55, no. 13s, 2023, "
    "doi: 10.1145/3586075.",

    "P. Vasanthi and V. M. Viswanatham, \"Multimodal sentiment analysis: Hybrid "
    "classification model with image and text feature descriptors,\" <i>Scientific "
    "Reports</i>, vol. 16, p. 13987, 2026, doi: 10.1038/s41598-026-42912-2.",

    "T. Niu, S. Zhu, L. Pang, and A. El Saddik, \"Sentiment analysis on multi-view "
    "social data,\" in <i>Proc. Int. Conf. Multimedia Modeling (MMM)</i>, 2016, "
    "pp. 15&ndash;27.",

    "Z. Li, B. Xu, C. Zhu, and T. Zhao, \"CLMLF: A contrastive learning and "
    "multi-layer fusion method for multimodal sentiment detection,\" in <i>Findings "
    "of the Assoc. for Computational Linguistics: NAACL 2022</i>, "
    "pp. 2282&ndash;2294.",
]
for i, ref in enumerate(references, start=1):
    story.append(Paragraph(f"[{i}] {ref}", S_REF))

doc.build(story)
print(f"Wrote {OUT}")
