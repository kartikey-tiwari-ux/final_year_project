"""
Generates an IEEE two-column conference-paper-format PDF from this project's
paper.md content (research_paper_v2). Run with the main project's venv:

    ../venv/Scripts/python.exe src/generate_ieee_pdf.py

Output: paper_IEEE.pdf
"""
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame, FrameBreak, NextPageTemplate,
    Paragraph, Spacer, Table, TableStyle, KeepTogether, Image,
)
from reportlab.lib import colors

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_IEEE.pdf"
FIG = ROOT / "figures" / "sharp_vs_blur.png"

PAGE_W, PAGE_H = letter
MARGIN_L = MARGIN_R = 0.75 * inch
MARGIN_TOP = 0.75 * inch
MARGIN_BOTTOM = 0.75 * inch
COL_GAP = 0.25 * inch

USABLE_W = PAGE_W - MARGIN_L - MARGIN_R
COL_W = (USABLE_W - COL_GAP) / 2.0
COL1_X = MARGIN_L
COL2_X = MARGIN_L + COL_W + COL_GAP

HEADER_H = 2.85 * inch
FULL_H = PAGE_H - MARGIN_TOP - MARGIN_BOTTOM
COL_H_P1 = FULL_H - HEADER_H

# ---------------------------------------------------------------- styles ---
S_TITLE = ParagraphStyle(
    "Title", fontName="Times-Bold", fontSize=15, leading=18,
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
    title="Embedded-Text Leakage in Multimodal Meme Sentiment Classification",
    author="Kartikey Tiwari, Akshat Sharma, Karan Sharma, Lalit Sharma",
)
doc.addPageTemplates([
    PageTemplate(id="First", frames=[header_frame, col1_p1, col2_p1], onPage=draw_page_number),
    PageTemplate(id="Later", frames=[col1_later, col2_later], onPage=draw_page_number),
])

story = []

# ------------------------------------------------------------- header ------
story.append(Paragraph(
    "Embedded-Text Leakage in Multimodal Meme Sentiment Classification: "
    "A Resource-Constrained Study of When Fusion Really Helps", S_TITLE))
story.append(Paragraph("Kartikey Tiwari, Akshat Sharma, Karan Sharma, Lalit Sharma", S_AUTHORS))
story.append(Paragraph("Department of Computer Science and Engineering", S_AFFIL))
story.append(Paragraph("Ajay Kumar Garg Engineering College, Ghaziabad", S_AFFIL))
story.append(Paragraph("Dr. A.P.J. Abdul Kalam Technical University, Lucknow", S_AFFIL))
story.append(Paragraph("<i>Supervisor: Dr. Avdhesh Gupta</i>", S_AFFIL))

story.append(NextPageTemplate("Later"))
story.append(FrameBreak())

# -------------------------------------------------------------- body -------

ABSTRACT = (
    "<b><i>Abstract&mdash;</i></b>Prior controlled comparisons of text-only, "
    "image-only, and multimodal sentiment classification on social media report that "
    "fusion helps because text and images carry complementary signal, using "
    "photographic images where caption and picture are produced independently. "
    "Internet memes break that independence: meme text is rendered as pixels on the "
    "image, so an image-only model can, in principle, re-derive the text signal a "
    "text-only model already sees. We ask whether fusion gains reported for "
    "photographic images still hold for memes, and how much of any gain is "
    "attributable to this embedded-text channel. We run a controlled, same-split "
    "comparison on a stratified 3,500-sample subset (2,800/350/350) of the public "
    "Memotion dataset (SemEval-2020 Task 8; 6,992 memes total), collapsing 5-class "
    "sentiment to 3 classes. We evaluate TF-IDF+LogisticRegression and "
    "frozen-DistilBERT text models, a from-scratch CNN and frozen-ResNet18 image "
    "model, and a concatenation-fusion MLP &mdash; each trained twice: on the "
    "original (\"sharp\") memes, and on the same memes after a strong Gaussian blur "
    "that destroys fine text-scale pixel detail (objectively confirmed via a 99.9% "
    "drop in mean Laplacian-variance sharpness on a 150-image sample) while "
    "preserving coarse colour and composition. On sharp images, fusion reaches 52.6% "
    "test accuracy / 36.6% macro-F1, clearly ahead of the best unimodal model (TF-IDF "
    "text: 42.6% accuracy; DistilBERT text: 33.4% macro-F1). Blurring barely changes "
    "the image-only model (&minus;1.4pp accuracy, &minus;0.2pp macro-F1), but "
    "collapses most of the fusion gain (&minus;10.9pp accuracy, &minus;3.6pp "
    "macro-F1): blurred-image fusion performs almost identically to text-only "
    "(+2.3pp accuracy, &minus;0.4pp macro-F1). This indicates the fusion benefit here "
    "is substantially mediated by the image channel's access to legible embedded "
    "text rather than independent visual sentiment cues &mdash; with direct "
    "implications for systems that assume image and text are independent evidence "
    "sources when images may themselves contain the text. All experiments ran on "
    "constrained consumer CPU hardware (no GPU, 8GB RAM)."
)
story.append(Paragraph(ABSTRACT, S_LEAD))

INDEX_TERMS = (
    "<b><i>Index Terms&mdash;</i></b>Multimodal sentiment analysis, Internet memes, "
    "social media analytics, feature fusion, DistilBERT, ResNet18, embedded text, "
    "modality leakage, resource-constrained machine learning."
)
story.append(Paragraph(INDEX_TERMS, S_LEAD))

# I. INTRODUCTION
story.append(Paragraph("I. INTRODUCTION", S_HEAD))
story.append(Paragraph(
    "Multimodal sentiment analysis on social media typically assumes text and image "
    "are independently produced evidence sources whose combination can only help, or "
    "at worst add noise, precisely because they are independent. That assumption is "
    "reasonable for a tweet consisting of a caption plus an attached photograph. It is "
    "not obviously reasonable for an Internet meme, where the \"image\" frequently "
    "<i>is</i> the text: bold caption text is rendered directly onto the picture. A "
    "convolutional image encoder operating on raw pixels does not know, a priori, that "
    "a region of high-contrast, high-frequency pixels is text rather than visual "
    "texture &mdash; it may simply learn to exploit it, whether or not that was the "
    "intended visual signal. Consider a meme pairing a stock photograph with the "
    "bold caption \"WHEN YOU FINALLY SUBMIT THE PROJECT REPORT\" rendered directly "
    "on the image: a text-only model reads the caption and infers relief/positive "
    "sentiment from the words themselves; an \"image-only\" model, denied the "
    "caption field, may still recover much of the same signal simply by reading the "
    "same words off the pixels &mdash; it is not actually limited to the "
    "photograph's visual content the way an image-only model would be for an "
    "attached but textually separate photograph. This paper asks: (RQ1) does "
    "multimodal fusion still "
    "outperform unimodal baselines for meme sentiment classification, replicating the "
    "pattern reported for photographic images; and (RQ2) how much of any fusion gain "
    "depends specifically on the image channel reading embedded text, versus "
    "independent visual cues? We answer RQ2 with a direct ablation: destroy legible "
    "text-scale detail with an objectively validated Gaussian blur, re-extract image "
    "features, and re-train every image-dependent model, holding everything else "
    "fixed.", S_BODY))

# II. RELATED WORK
story.append(Paragraph("II. RELATED WORK", S_HEAD))
story.append(Paragraph(
    "Controlled text-vs-image-vs-multimodal comparisons on real social-media data "
    "are rare. The closest located prior work performs a same-dataset, same-metrics "
    "comparison of text-only, image-only, and multimodal sentiment classification on "
    "photographic Twitter image-text pairs, and reports that multimodal fusion "
    "outperforms either unimodal model on every aggregate metric, with the gain "
    "concentrated specifically on the subset of posts where the text-only and "
    "image-only models disagree, and a small regression on the subset where they "
    "already agree. That result is a natural baseline expectation for the present "
    "study, since it establishes what a \"clean\" complementary-modality fusion "
    "pattern looks like when text and image are produced independently. Multimodal "
    "fusion more broadly has a mature survey literature covering fusion-architecture "
    "taxonomies, benchmark datasets, and open challenges [1], [2], but these surveys "
    "largely treat \"image\" as a visually distinct channel from \"text\" without "
    "examining cases where the two are pixel-level coupled, as in memes.", S_BODY))
story.append(Paragraph(
    "The Memotion / SemEval-2020 Task 8 dataset used here was introduced explicitly "
    "to study sentiment, humor, sarcasm, offensiveness, and motivational content in "
    "memes as combined text-image artifacts, with the shared task's participating "
    "systems overwhelmingly adopting some form of text-plus-image fusion architecture "
    "[3]. The Hateful Memes Challenge similarly argues that meme meaning often "
    "depends on a genuine text-image interaction (e.g., a caption that is benign "
    "alone but harmful paired with a specific image) that neither modality captures "
    "in isolation, and was explicitly designed with \"benign confounder\" examples to "
    "force models away from relying on either modality alone [4]. Neither line of "
    "work, to our knowledge, directly quantifies how much of a meme multimodal "
    "model's measured cross-modal gain is attributable to the image channel simply "
    "re-reading the same embedded text a text-only model already sees, versus "
    "genuinely independent visual content (composition, colour, facial expression). "
    "This is precisely the gap this paper's blur ablation targets. Separately, the "
    "scene-text/TextVQA literature establishes that recognizing text rendered inside "
    "images is a major, distinct capability from general visual-object recognition, "
    "and that models without it systematically fail on image-understanding tasks "
    "where the answer depends on in-image text [5] &mdash; directly motivating the "
    "hypothesis that a meme image encoder's apparent \"visual\" sentiment signal may "
    "substantially be this same capability in disguise. Our text and image encoders "
    "follow established lightweight transfer-learning choices for "
    "resource-constrained settings: DistilBERT, a distilled, faster, smaller variant "
    "of BERT retaining most of its language-understanding performance [6], for text; "
    "and ResNet18, a compact residual convolutional architecture originally "
    "introduced for large-scale image classification and now a standard frozen "
    "feature extractor for transfer learning [7], for images &mdash; both used "
    "frozen (inference-only) throughout.", S_BODY))

# III. RESEARCH QUESTIONS
story.append(Paragraph("III. RESEARCH QUESTIONS", S_HEAD))
story.append(Paragraph(
    "<b>RQ1:</b> Does feature-concatenation multimodal fusion outperform unimodal "
    "text-only and image-only sentiment classification on real Internet memes, as "
    "previously reported for photographic social-media images?", S_BODY))
story.append(Paragraph(
    "<b>RQ2:</b> How much of any fusion gain is attributable specifically to the "
    "image channel's access to legible embedded meme text, versus independent visual "
    "sentiment cues &mdash; tested by destroying legible text-scale detail via "
    "Gaussian blur and measuring the resulting change in each model's performance?",
    S_BODY))
story.append(Paragraph(
    "<b>RQ3 (comparability check):</b> Does the previously reported pattern that "
    "fusion's benefit concentrates on the text/image-disagreement subset replicate "
    "on meme data?", S_BODY))

# IV. DATASET
story.append(Paragraph("IV. DATASET", S_HEAD))
story.append(Paragraph(
    "We use Memotion (SemEval-2020 Task 8 [3]): real, publicly shared Internet memes, "
    "human-annotated for 5-class sentiment, humor, sarcasm, offensiveness, and "
    "motivational content, with OCR-extracted, human-corrected caption text per meme. "
    "We accessed the data via the Ahren09/MMSoc_Memotion Hugging Face mirror (6,992 "
    "memes; 5,593/699/700 train/validation/test), disclosing this third-party-mirror "
    "provenance explicitly. We merge the 5-class sentiment label to 3 classes "
    "(very_positive+positive&rarr;positive; neutral&rarr;neutral; "
    "negative+very_negative&rarr;negative) for comparability with standard 3-class "
    "social-media sentiment framings.", S_BODY))
story.append(Paragraph(
    "Given this project's CPU-only, 8GB-RAM hardware (Section V), we draw a "
    "stratified, per-class-proportional subsample of 2,800/350/350 "
    "train/validation/test memes (3,500 total) rather than using the full 6,992, "
    "keeping the experiment tractable while preserving each split's class "
    "proportions. Resulting distribution &mdash; train: positive 1,659 (59.3%), "
    "neutral 882 (31.5%), negative 259 (9.3%); validation: positive 209 (59.7%), "
    "neutral 111 (31.7%), negative 30 (8.6%); test: positive 214 (61.1%), neutral "
    "109 (31.1%), negative 27 (7.7%) &mdash; a real, substantial imbalance motivating "
    "macro-F1 as the primary metric.", S_BODY))

# V. METHODOLOGY
story.append(Paragraph("V. METHODOLOGY", S_HEAD))
story.append(Paragraph(
    "All models share the same split, fixed seed (42), and evaluation code. All "
    "image-dependent models are trained and evaluated in two parallel conditions, "
    "<i>sharp</i> and <i>blur</i>, from otherwise-identical pipelines.", S_BODY))

story.append(Paragraph("A. Text and Image Pipelines", S_SUBHEAD))
story.append(Paragraph(
    "Text: (A1) TF-IDF (1&ndash;2 grams, 10k features) + class-weighted Logistic "
    "Regression on the corrected caption text; (A2) frozen DistilBERT mean-pooled "
    "embeddings (768-dim) + class-weighted Logistic Regression. Image: (B1) a small "
    "4-layer CNN trained from scratch at 96&times;96 resolution; (B2) frozen "
    "ImageNet-pretrained ResNet18 penultimate-layer embeddings (512-dim) + "
    "class-weighted Logistic Regression, run on both the sharp image and, as "
    "B2-Blur, the blurred image.", S_BODY))

story.append(Paragraph("B. Blur Ablation", S_SUBHEAD))
story.append(Paragraph(
    "The blur condition applies a Gaussian blur (radius 14px, at native resolution "
    "before the standard 224&times;224 resize) to every image before feature "
    "extraction. We validate this destroys text-scale detail using an OCR-independent "
    "focus measure (variance of the image Laplacian) on a random 150-image test "
    "sample: mean sharp-image variance 2,021.6 vs. mean blurred-image variance 1.5 "
    "&mdash; a 99.9% reduction, i.e., fine high-frequency edge content that would "
    "carry legible character strokes is essentially eliminated, while coarse colour "
    "and compositional layout is preserved.", S_BODY))

story.append(Paragraph("C. Multimodal Fusion", S_SUBHEAD))
story.append(Paragraph(
    "Feature-level concatenation of the A2 text embedding (768-dim) and the B2 image "
    "embedding (512-dim, sharp or blurred) &rarr; a 2-hidden-layer MLP "
    "(256&rarr;128&rarr;3) with dropout (0.4, 0.3), class-weighted cross-entropy, "
    "Adam (lr=1e-3, weight decay=1e-4), batch size 32, up to 40 epochs with early "
    "stopping (patience 8) on validation macro-F1; the best-validation checkpoint is "
    "evaluated once on the untouched test split. <b>C</b> denotes sharp-image fusion; "
    "<b>C-Blur</b> denotes the identical procedure on the blurred-image embedding.",
    S_BODY))

story.append(Paragraph("D. Why Frozen Pretrained Encoders", S_SUBHEAD))
story.append(Paragraph(
    "As in prior resource-constrained work, this choice is driven directly by "
    "measured hardware constraints (CPU-only, 8GB RAM): batched frozen-DistilBERT "
    "extraction measured 46.5&ndash;67.6 ms/sample, and batched frozen-ResNet18 "
    "extraction measured 63.2&ndash;70.5 ms/sample (sharp and blurred conditions, all "
    "splits) &mdash; tractable only because the encoders are frozen rather than "
    "fine-tuned end to end.", S_BODY))

# VI. EXPERIMENTAL SETUP
story.append(Paragraph("VI. EXPERIMENTAL SETUP", S_HEAD))
story.append(Paragraph(
    "Fixed seed (42) throughout. Metrics: accuracy, macro-F1, weighted-F1, and "
    "per-class precision/recall/F1 with confusion matrices; macro-F1 is the primary "
    "metric given the class imbalance in Section IV. All numbers in Section VII are "
    "read directly from results/*.json produced by this project's own pipeline "
    "scripts, run end to end on CPU-only, 8GB-RAM hardware, with no numbers invented "
    "or adjusted post hoc.", S_BODY))

story.append(Paragraph("A. Hardware and Software Environment", S_SUBHEAD))
story.append(Paragraph(
    "All experiments ran on a single consumer laptop: CPU-only (no GPU), 8GB RAM, "
    "Windows 10. Software: Python 3.13, PyTorch 2.14 (CPU build), torchvision 0.29, "
    "Hugging Face Transformers 5.18, and scikit-learn 1.9. This is the same class of "
    "constrained hardware used in prior resource-constrained work in this area, "
    "reported here in full rather than abstracted away, since it directly motivated "
    "the frozen-encoder modelling choices in Section V and the stratified "
    "subsampling decision in Section IV.", S_BODY))

story.append(Paragraph("B. Fusion Training Behaviour", S_SUBHEAD))
story.append(Paragraph(
    "The fusion MLP (C) converged with early stopping at epoch 21 of a maximum 40 "
    "(best validation macro-F1 checkpoint retained); the blurred-image fusion model "
    "(C-Blur) trained longer before early stopping, at epoch 32 &mdash; consistent "
    "with the blurred-image embedding carrying a weaker, slower-to-fit signal for "
    "the fusion head to exploit, a further behavioural indicator (alongside the "
    "headline metrics) that the sharp-image embedding offers the fusion model more "
    "immediately usable signal than the blurred one.", S_BODY))

# VII. RESULTS
story.append(Paragraph("VII. RESULTS", S_HEAD))
story.append(Paragraph(
    "Table I reports test-set performance for every model, sharp and blurred image "
    "conditions (text-only models are listed once).", S_BODY))

table_header = [Paragraph(h, S_TABLECELL_B) for h in ["Model", "Acc.", "Macro-F1", "Wtd-F1"]]
rows_data = [
    ("A1 &mdash; TF-IDF + LogReg (text)", "42.6%", "33.3%", "43.7%", False),
    ("A2 &mdash; Frozen DistilBERT + LogReg (text)", "39.4%", "33.4%", "42.1%", False),
    ("B1 &mdash; CNN from scratch (sharp)", "61.1%", "25.3%", "46.4%", False),
    ("B2 &mdash; Frozen ResNet18 + LogReg (sharp)", "38.6%", "31.0%", "42.2%", False),
    ("B2-Blur &mdash; Frozen ResNet18 + LogReg (blur)", "37.1%", "30.7%", "39.6%", False),
    ("C &mdash; Concat Fusion (text + sharp image)", "52.6%", "36.6%", "50.6%", True),
    ("C-Blur &mdash; Concat Fusion (text + blurred image)", "41.7%", "33.0%", "44.4%", False),
]
table_rows = [table_header]
for name, acc, f1m, f1w, best in rows_data:
    style = S_TABLECELL_B if best else S_TABLECELL
    style_c = S_TABLECELL_B if best else S_TABLECELL_C
    table_rows.append([
        Paragraph(name, style), Paragraph(acc, style_c),
        Paragraph(f1m, style_c), Paragraph(f1w, style_c),
    ])

results_table = Table(table_rows, colWidths=[COL_W * 0.58, COL_W * 0.14, COL_W * 0.14, COL_W * 0.14])
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
    Paragraph("TABLE I<br/>RESULTS: TEST-SET PERFORMANCE BY MODEL AND IMAGE CONDITION", S_CAPTION),
    results_table,
]))

story.append(Spacer(1, 6))
story.append(Paragraph(
    "<b>RQ1 (does fusion help):</b> yes. Fusion on sharp images (C) clearly "
    "outperforms every unimodal model on both headline metrics: +10.0pp accuracy "
    "over the best unimodal accuracy (A1) and +3.2pp macro-F1 over the best unimodal "
    "macro-F1 (A2). This replicates, on a structurally different data type, the "
    "qualitative finding that multimodal fusion improves social-media sentiment "
    "classification over either modality alone.", S_BODY))
story.append(Paragraph(
    "<b>On B1:</b> B1 reaches the highest raw accuracy (61.1%) but the lowest "
    "macro-F1 (25.3%) &mdash; its confusion matrix shows collapse toward the "
    "majority (positive) class, a genuine training-instability failure under this "
    "hardware's constrained training budget rather than a real feature-learning "
    "advantage; its accuracy alone is not a meaningful comparison point, which is "
    "why macro-F1 is treated as primary. Table III makes this concrete: training "
    "loss stays pinned near ln(3)&asymp;1.099 (i.e., close to uniform-random "
    "output) for all 6 epochs, validation macro-F1 is identical at epochs 1, 2, and "
    "6 (0.249, matching pure majority-class prediction), and briefly collapses "
    "further at epoch 5 (0.161, accuracy 31.7%) before reverting &mdash; the model "
    "never escapes a narrow band of degenerate, near-constant predictions.", S_BODY))

b1_header = [Paragraph(h, S_TABLECELL_B) for h in ["Epoch", "Train Loss", "Val Acc.", "Val Macro-F1"]]
b1_rows_data = [
    (1, 1.100, 0.597, 0.249), (2, 1.099, 0.597, 0.249), (3, 1.099, 0.586, 0.252),
    (4, 1.098, 0.600, 0.256), (5, 1.102, 0.317, 0.161), (6, 1.098, 0.597, 0.249),
]
b1_rows = [b1_header]
for ep, loss, acc, f1 in b1_rows_data:
    b1_rows.append([Paragraph(str(ep), S_TABLECELL_C), Paragraph(f"{loss:.3f}", S_TABLECELL_C),
                     Paragraph(f"{acc:.3f}", S_TABLECELL_C), Paragraph(f"{f1:.3f}", S_TABLECELL_C)])
b1_table = Table(b1_rows, colWidths=[COL_W * 0.2, COL_W * 0.27, COL_W * 0.26, COL_W * 0.27])
b1_table.setStyle(TableStyle([
    ("LINEABOVE", (0, 0), (-1, 0), 1.0, colors.black),
    ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
    ("LINEBELOW", (0, -1), (-1, -1), 1.0, colors.black),
    ("TOPPADDING", (0, 0), (-1, -1), 2),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
]))
story.append(KeepTogether([
    Paragraph("TABLE III<br/>B1 (CNN FROM SCRATCH) PER-EPOCH TRAINING BEHAVIOUR", S_CAPTION),
    b1_table,
]))
story.append(Spacer(1, 6))

story.append(Paragraph("A. RQ2: The Blur Ablation", S_SUBHEAD))
story.append(Paragraph(
    "Fig. 1 shows accuracy and macro-F1 for every image-dependent configuration, "
    "sharp vs. blurred. Two results stand out. First, B2 is almost unaffected by "
    "blurring (accuracy 38.6%&rarr;37.1%, &minus;1.4pp; macro-F1 "
    "31.0%&rarr;30.7%, &minus;0.2pp): the image-only linear classifier was evidently "
    "not relying heavily on fine legible-text detail. Second, fusion is "
    "substantially affected (accuracy 52.6%&rarr;41.7%, &minus;10.9pp; macro-F1 "
    "36.6%&rarr;33.0%, &minus;3.6pp): most of fusion's advantage over text-only "
    "evaporates. C-Blur's accuracy is only 2.3pp above text-only (A2) and its "
    "macro-F1 is 0.4pp <i>below</i> text-only.", S_BODY))

story.append(KeepTogether([
    Image(str(FIG), width=COL_W, height=COL_W * (3.2 / 6.6)),
    Paragraph("Fig. 1. Sharp vs. blurred accuracy and macro-F1, by model.", S_CAPTION),
]))

story.append(Paragraph(
    "Taken together, this is the paper's central finding: the multimodal fusion "
    "gain measured here is substantially mediated by the image channel's access to "
    "legible embedded meme text, not by independent visual sentiment cues that a "
    "linear probe on frozen ResNet18 features can already detect. The fusion MLP "
    "evidently extracts residual text-correlated signal from the sharp-image "
    "embedding that the simpler B2 probe does not surface on its own &mdash; and "
    "that signal specifically, not general visual composition, is what the blur "
    "ablation removes.", S_BODY))

story.append(Paragraph("B. Confusion Matrix: Fusion, Sharp vs. Blurred", S_SUBHEAD))
story.append(Paragraph(
    "Table II shows the test-set confusion matrix for C (sharp) and C-Blur. The "
    "clearest shift under blurring is the positive class: C correctly recovers 159 "
    "of 214 positive memes (74.3% recall) but C-Blur recovers only 105 (49.1% "
    "recall), with many positive memes instead misclassified as neutral (72 under "
    "blur vs. 26 sharp) &mdash; i.e., once embedded text is illegible, the fusion "
    "model loses much of its ability to confidently identify positive memes "
    "specifically, which is consistent with positive-sentiment memes in this "
    "dataset leaning heavily on readable caption wording (e.g., explicit "
    "celebratory or congratulatory text) that a blurred image can no longer supply.",
    S_BODY))

cm_header = [Paragraph(h, S_TABLECELL_B) for h in ["True \\ Pred", "Neg.", "Neu.", "Pos."]]
cm_c = [[8, 4, 15], [21, 17, 71], [29, 26, 159]]
cm_cblur = [[6, 10, 11], [24, 35, 50], [37, 72, 105]]
cm_rownames = ["Neg.", "Neu.", "Pos."]

def cm_table(cm, caption):
    rows = [cm_header]
    for rn, row in zip(cm_rownames, cm):
        rows.append([Paragraph(rn, S_TABLECELL_B)] + [Paragraph(str(v), S_TABLECELL_C) for v in row])
    t = Table(rows, colWidths=[COL_W * 0.28] + [COL_W * 0.24] * 3)
    t.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 1.0, colors.black),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 1.0, colors.black),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
    ]))
    return [Paragraph(caption, S_CAPTION), t]

story.append(KeepTogether(
    [Paragraph("TABLE II<br/>TEST CONFUSION MATRICES: C (SHARP) VS. C-BLUR", S_CAPTION)]
    + cm_table(cm_c, "(a) C &mdash; fusion, sharp image")
    + [Spacer(1, 4)]
    + cm_table(cm_cblur, "(b) C-Blur &mdash; fusion, blurred image")
))
story.append(Spacer(1, 6))

story.append(Paragraph("C. RQ3: Modality Conflict Analysis", S_SUBHEAD))
story.append(Paragraph(
    "Text-only (A2) and image-only (B2, sharp) predictions disagree on 63.1% of "
    "test memes (221/350) &mdash; higher than the 53.3% disagreement rate reported "
    "for photographic tweet images, consistent with meme images and their embedded "
    "captions being a less independent signal pair. On this disagreement subset, "
    "fusion outperforms both unimodal models (51.6% vs. 37.1% text-only, 35.7% "
    "image-only). Unlike the photographic-image pattern &mdash; a small regression "
    "where modalities agree &mdash; fusion here also outperforms both unimodal "
    "models on the agreement subset (54.3% vs. 43.4% for both). We attribute this "
    "to the same cause as the blur-ablation finding: because meme \"image\" "
    "information is partly text-derived, the fusion head gains a more consistently "
    "useful signal than in the photographic-image case.", S_BODY))
story.append(Paragraph(
    "<b>Per-class:</b> fusion (C) substantially improves the majority positive "
    "class over text-only (F1 69.3% vs. 49.6%) and the minority negative class "
    "(F1 18.8% vs. 17.0%), but regresses on neutral (F1 21.8% vs. 33.8%, recall "
    "35.8%&rarr;15.6%) &mdash; the same majority-reinforcing, "
    "minority-adjacent-class cost pattern reported for fusion on photographic "
    "images, replicated here on a different dataset and task framing.", S_BODY))

story.append(KeepTogether([
    Image(str(ROOT / "figures" / "perclass_f1.png"), width=COL_W, height=COL_W * (3.0 / 6.6)),
    Paragraph("Fig. 2. Per-class F1: text-only vs. fusion sharp vs. fusion blurred.", S_CAPTION),
]))

# VIII. DISCUSSION
story.append(Paragraph("VIII. DISCUSSION, ETHICS, AND LIMITATIONS", S_HEAD))
story.append(Paragraph("A. Practical Implication", S_SUBHEAD))
story.append(Paragraph(
    "Any system fusing \"image\" and \"text\" modalities for meme-like content "
    "(memes, captioned infographics, text screenshots) should not assume the image "
    "channel is an independent evidence source. Reported multimodal gains on such "
    "content may substantially reflect the image model re-deriving the same text "
    "signal rather than genuine cross-modal understanding; evaluations that care "
    "about this distinction should include a leakage ablation of this kind.",
    S_BODY))
story.append(Paragraph("B. Non-Clinical, Non-Moderation Scope", S_SUBHEAD))
story.append(Paragraph(
    "This study measures sentiment-label prediction accuracy only; it makes no "
    "claims about detecting humor, sarcasm, offensiveness, or hateful content, "
    "despite the dataset providing such labels, and should not be read as a "
    "validated content-moderation tool.", S_BODY))
story.append(Paragraph("C. Threats to Validity", S_SUBHEAD))
story.append(Paragraph(
    "<b>Internal validity:</b> all reported numbers come from a single run per "
    "model at a fixed seed (42); we do not report variance across multiple seeds, "
    "so small differences between models (e.g., B2 vs. B2-Blur's 1.4pp accuracy "
    "gap) should be read with appropriate caution, while the large differences "
    "central to this paper's claims (C vs. C-Blur's 10.9pp accuracy gap) are "
    "unlikely to be seed noise given their size relative to the small-difference "
    "case observed elsewhere in the same table. <b>Construct validity:</b> the "
    "blur ablation is a proxy for \"text illegibility,\" not a direct text-region "
    "mask; Section VIII-D already discusses this as a limitation rather than a "
    "resolved concern. <b>External validity:</b> findings are specific to this "
    "dataset, label scheme, and model family, and should not be assumed to "
    "transfer to other meme corpora, languages, or encoder architectures without "
    "further testing.", S_BODY))

story.append(Paragraph("D. Key Limitations", S_SUBHEAD))
story.append(Paragraph(
    "(1) the 3,500-sample stratified subsample is smaller than the full 6,992-meme "
    "dataset; results may shift at full scale. (2) the 5-to-3-class sentiment "
    "collapse discards intensity information. (3) the Gaussian blur removes all "
    "fine high-frequency detail, not exclusively text; some measured effect could "
    "reflect loss of other fine visual detail, though the meme format's "
    "text-dominant high-frequency content makes text the most plausible primary "
    "contributor. (4) the dataset is English-language, non-codemixed memes from a "
    "fixed collection period and may not generalize to other languages, formats, or "
    "eras. (5) B1's training instability limits how much weight its result should "
    "carry. (6) CPU-only compute constrained the from-scratch CNN's training budget "
    "and resolution, documented rather than hidden.", S_BODY))

# IX. CONCLUSION
story.append(Paragraph("IX. CONCLUSION AND FUTURE WORK", S_HEAD))
story.append(Paragraph(
    "Within a real, resource-constrained CPU-only environment, this study finds "
    "multimodal fusion improves meme sentiment classification over unimodal "
    "baselines (RQ1), replicating the pattern reported for photographic social-media "
    "images. However, a direct blur ablation shows a substantial share of this gain "
    "is attributable to the image channel's access to legible embedded text rather "
    "than independent visual cues (RQ2) &mdash; the gain collapses to near "
    "text-only performance once that text is illegible. The modality-conflict "
    "pattern also differs from prior photographic-image findings: fusion helps on "
    "both the disagreement and agreement subsets here (RQ3). Future work: a direct "
    "OCR-region-masking leakage estimate rather than global blurring; evaluating "
    "whether this pattern holds for the dataset's humor/sarcasm/offensiveness "
    "labels; and testing whether an image encoder explicitly trained to ignore text "
    "regions recovers a meaningful independent-visual-signal multimodal gain.",
    S_BODY))

# CODE AND DATA AVAILABILITY
story.append(Paragraph("CODE AND DATA AVAILABILITY", S_HEAD))
story.append(Paragraph(
    "All pipeline scripts (data export, feature extraction, model training, "
    "analysis, and figure/PDF generation), the fixed random seed, and the raw "
    "results/*.json outputs underlying every number and figure in this paper are "
    "retained alongside this report for independent re-execution. The Memotion "
    "dataset is publicly available through the SemEval-2020 Task 8 organizers and "
    "through third-party mirrors, including the one used here; we did not "
    "redistribute any dataset files ourselves.", S_BODY))

# ACKNOWLEDGMENT
story.append(Paragraph("ACKNOWLEDGMENT", S_HEAD))
story.append(Paragraph(
    "Dataset: Sharma et al. (2020, SemEval-2020 Task 8 / Memotion) [3], accessed via "
    "the Ahren09/MMSoc_Memotion Hugging Face mirror. Pretrained models: DistilBERT "
    "and ResNet18, accessed via Hugging Face Transformers and torchvision "
    "respectively.", S_BODY))

# APPENDIX
story.append(Paragraph("APPENDIX: HYPERPARAMETER SUMMARY", S_HEAD))
appx_header = [Paragraph(h, S_TABLECELL_B) for h in ["Model", "Key Hyperparameters"]]
appx_rows_data = [
    ("A1 TF-IDF+LogReg", "1-2 grams, max 10k features, sublinear TF; "
     "LogReg: C=1.0, class_weight=balanced, max_iter=2000"),
    ("A2 DistilBERT+LogReg", "distilbert-base-uncased, mean-pooled, max_len=96; "
     "LogReg: C=1.0, class_weight=balanced"),
    ("B1 CNN (scratch)", "4 conv layers (16/32/64/64 ch.), 96x96 input, Adam "
     "lr=1e-3, batch=32, 6 epochs, dropout=0.3, class-weighted CE"),
    ("B2 ResNet18+LogReg", "ImageNet-pretrained, frozen, 224x224 input, "
     "512-dim penultimate layer; LogReg: C=1.0, class_weight=balanced"),
    ("B2-Blur (ablation)", "Identical to B2, but Gaussian blur radius=14px "
     "applied pre-resize"),
    ("C / C-Blur Fusion MLP", "Input 1280-dim (768+512), hidden "
     "256{→}128{→}3, dropout 0.4/0.3, Adam lr=1e-3, weight_decay=1e-4, "
     "batch=32, max 40 epochs, early stop patience=8 on val macro-F1"),
]
appx_rows = [appx_header]
for name, params in appx_rows_data:
    appx_rows.append([Paragraph(name, S_TABLECELL_B), Paragraph(params, S_TABLECELL)])
appx_table = Table(appx_rows, colWidths=[COL_W * 0.28, COL_W * 0.72])
appx_table.setStyle(TableStyle([
    ("LINEABOVE", (0, 0), (-1, 0), 1.0, colors.black),
    ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
    ("LINEBELOW", (0, -1), (-1, -1), 1.0, colors.black),
    ("TOPPADDING", (0, 0), (-1, -1), 3),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
]))
story.append(appx_table)
story.append(Spacer(1, 6))

# REFERENCES
story.append(Paragraph("REFERENCES", S_HEAD))
references = [
    "A. Gandhi, K. Adhvaryu, S. Poria, E. Cambria, and A. Hussain, \"Multimodal "
    "sentiment analysis: A systematic review of history, datasets, multimodal fusion "
    "methods, applications, challenges and future directions,\" <i>Information "
    "Fusion</i>, vol. 91, pp. 424&ndash;444, 2023.",

    "R. Das and T. D. Singh, \"Multimodal Sentiment Analysis: A Survey of Methods, "
    "Trends, and Challenges,\" <i>ACM Computing Surveys</i>, vol. 55, no. 13s, 2023, "
    "doi: 10.1145/3586075.",

    "C. Sharma, D. Bhageria, W. Scott, S. Pykl, A. Das, T. Chakraborty, "
    "V. Pulabaigari, and B. Gamb&auml;ck, \"SemEval-2020 Task 8: Memotion Analysis "
    "&mdash; the Visuo-Lingual Metaphor!,\" in <i>Proc. 14th Int. Workshop on "
    "Semantic Evaluation (SemEval-2020)</i>, 2020.",

    "D. Kiela, H. Firooz, A. Mohan, V. Goswami, A. Singh, P. Ringshia, and "
    "D. Testuggine, \"The Hateful Memes Challenge: Detecting Hate Speech in "
    "Multimodal Memes,\" in <i>Advances in Neural Information Processing Systems "
    "(NeurIPS)</i>, vol. 33, 2020.",

    "A. Singh, V. Natarajan, M. Shah, Y. Jiang, X. Chen, D. Batra, D. Parikh, and "
    "M. Rohrbach, \"Towards VQA Models That Can Read,\" in <i>Proc. IEEE/CVF Conf. "
    "on Computer Vision and Pattern Recognition (CVPR)</i>, 2019, "
    "pp. 8317&ndash;8326.",

    "V. Sanh, L. Debut, J. Chaumond, and T. Wolf, \"DistilBERT, a distilled version "
    "of BERT: smaller, faster, cheaper and lighter,\" <i>arXiv preprint "
    "arXiv:1910.01108</i>, 2019.",

    "K. He, X. Zhang, S. Ren, and J. Sun, \"Deep Residual Learning for Image "
    "Recognition,\" in <i>Proc. IEEE Conf. on Computer Vision and Pattern "
    "Recognition (CVPR)</i>, 2016, pp. 770&ndash;778.",

    "T. Niu, S. Zhu, L. Pang, and A. El Saddik, \"Sentiment Analysis on Multi-View "
    "Social Data,\" in <i>Proc. Int. Conf. on Multimedia Modeling (MMM)</i>, 2016, "
    "pp. 15&ndash;27.",
]
for i, ref in enumerate(references, start=1):
    story.append(Paragraph(f"[{i}] {ref}", S_REF))

doc.build(story)
print(f"Wrote {OUT}")
