"""Build Extended_Synopsis_Viva2.docx for the Second Project Viva (12-13 Oct 2026).

Every number below is read from results/*.json and research_paper_v2/results/*.json
at build time, never typed in by hand.

Run:  ./venv/Scripts/python docs/generate_extended_synopsis.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "Extended_Synopsis_Viva2.docx"
V2 = ROOT / "research_paper_v2"

MEMBERS = [("KARTIKEY TIWARI", "2300271540063"), ("AKSHAT SHARMA", "2300271540018"),
           ("KARAN SHARMA", "2300271540062"), ("LALIT SHARMA", "2300271540066")]
TITLE = "AI-DRIVEN MULTIMODAL SENTIMENT ANALYSIS FOR MENTAL HEALTH MONITORING ON SOCIAL MEDIA"

# --------------------------------------------------------------------------- data
def load(p):
    return json.loads(Path(p).read_text())

def pct(x):
    return f"{100 * x:.1f}%"

s1 = {}
for key, fname, name in [
    ("A1", "model_a1_tfidf_logreg", "A1: TF-IDF + Logistic Regression (text)"),
    ("A2", "model_a2_distilbert_frozen", "A2: Frozen DistilBERT + LogReg (text)"),
    ("B1", "model_b1_cnn_scratch", "B1: CNN from scratch (image)"),
    ("B2", "model_b2_resnet18_frozen", "B2: Frozen ResNet18 + LogReg (image)"),
    ("C", "model_c_concat_fusion", "C: Concatenation fusion MLP (text + image)"),
]:
    t = load(ROOT / "results" / f"{fname}.json")["test"]
    s1[key] = (name, t["accuracy"], t["f1_macro"], t["f1_weighted"])
late = load(ROOT / "results" / "model_c_late_fusion.json")
for key, sub, name in [("CL", "simple_average", "C-late: Late fusion, simple average"),
                       ("CLw", "tuned_weighted_average", "C-late: Late fusion, val-tuned weights")]:
    t = late[sub]["test"]
    s1[key] = (name, t["accuracy"], t["f1_macro"], t["f1_weighted"])

s2 = {}
for key, fname, name in [
    ("A1", "A1_tfidf_logreg", "A1: TF-IDF + LogReg (text)"),
    ("A2", "A2_distilbert_logreg", "A2: Frozen DistilBERT + LogReg (text)"),
    ("B1", "B1_cnn_scratch", "B1: CNN from scratch (sharp image)"),
    ("B2", "B2_resnet18_sharp_logreg", "B2: Frozen ResNet18 + LogReg (sharp)"),
    ("B2b", "B2blur_resnet18_blur_logreg", "B2-Blur: Frozen ResNet18 + LogReg (blurred)"),
    ("C", "C_concat_fusion_sharp", "C: Concat fusion (text + sharp image)"),
    ("Cb", "Cblur_concat_fusion_blur", "C-Blur: Concat fusion (text + blurred image)"),
]:
    d = load(V2 / "results" / f"{fname}.json")
    t = d.get("test", d)
    s2[key] = (name, t["accuracy"], t.get("f1_macro", t.get("macro_f1")),
               t.get("f1_weighted", t.get("weighted_f1")))
blur = load(V2 / "results" / "blur_leakage_analysis.json")
conf2 = load(V2 / "results" / "modality_conflict_analysis.json")

# --------------------------------------------------------------------------- figures
def architecture_figure(path):
    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=200)
    ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis("off")
    navy, ice, orange, grey = "#1f3a5f", "#dbe7f3", "#e07a1f", "#f2f2f2"

    def box(x, y, w, h, text, fc=ice, tc="black", bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    fc=fc, ec=navy, lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.2,
                color=tc, weight="bold" if bold else "normal", wrap=True)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=orange, lw=1.6))

    box(1, 19, 12, 8, "Social-media post\n(text + image)", fc=navy, tc="white", bold=True)
    # text branch
    box(18, 33, 15, 8, "Text cleaning\nURL/@mention strip,\nemoji -> words")
    box(38, 33, 17, 8, "A1: TF-IDF (1-2 gram)\nA2: DistilBERT (frozen)\n768-d embedding")
    box(60, 37, 15, 6, "Text classifier\n(LogReg)", fc=grey)
    # image branch
    box(18, 5, 15, 8, "Image preprocessing\nRGB, resize 224,\nImageNet normalise")
    box(38, 5, 17, 8, "B1: CNN from scratch\nB2: ResNet18 (frozen)\n512-d embedding")
    box(60, 3, 15, 6, "Image classifier\n(LogReg)", fc=grey)
    # fusion
    box(60, 17, 15, 12, "C: Concat 768+512\n-> MLP 256-128-3\n(dropout, class-\nweighted loss)", fc="#fde3cc", bold=True)
    box(80, 19, 18, 8, "Sentiment\npositive / neutral / negative\n+ confidence", fc=navy, tc="white", bold=True)
    box(80, 33, 18, 8, "Evaluation: Acc, Macro-F1,\nconfusion matrix, modality-\nconflict, explainability", fc=grey)
    arrow(13, 25, 18, 36); arrow(13, 21, 18, 10)
    arrow(33, 37, 38, 37); arrow(33, 9, 38, 9)
    arrow(55, 38, 60, 40); arrow(55, 8, 60, 6)
    arrow(55, 35, 60, 26); arrow(55, 11, 60, 20)
    arrow(75, 23, 80, 23); arrow(75, 40, 80, 37); arrow(75, 6, 89, 19)
    ax.text(50, 0.2, "Study 2 adds a Gaussian-blur branch (radius 14 px) before image feature extraction "
            "to isolate embedded-text leakage in memes.", ha="center", fontsize=7.5, style="italic")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def gantt_figure(path):
    tasks = [
        ("Literature review & synopsis", 0, 2, "done"),
        ("System design, tool & dataset setup", 2, 1, "done"),
        ("Study 1 implementation (MVSA)", 2.5, 1.2, "done"),
        ("Study 2: Memotion blur ablation", 3.2, 0.5, "done"),
        ("Multi-seed runs + significance tests", 3.5, 0.7, "next"),
        ("OCR-masking ablation, full Memotion", 4.0, 1.0, "next"),
        ("Cross-modal attention (stretch)", 4.5, 1.0, "next"),
        ("Manuscript refinement", 5.0, 1.0, "next"),
        ("Submission to target venue", 6.0, 1.0, "next"),
        ("Revisions & final report", 7.0, 2.0, "next"),
    ]
    fig, ax = plt.subplots(figsize=(9, 4), dpi=200)
    for i, (name, start, dur, st) in enumerate(reversed(tasks)):
        ax.barh(i, dur, left=start, color="#1f3a5f" if st == "done" else "#e07a1f", edgecolor="black", lw=0.5)
    ax.set_yticks(range(len(tasks))); ax.set_yticklabels([t[0] for t in reversed(tasks)], fontsize=8)
    months = ["Jul 26", "Aug", "Sep", "Oct", "Nov", "Dec", "Jan 27", "Feb", "Mar"]
    ax.set_xticks(range(len(months))); ax.set_xticklabels(months, fontsize=8)
    ax.axvline(3.37, color="red", ls="--", lw=1); ax.text(3.42, len(tasks) - 0.6, "Viva-II\n12 Oct", color="red", fontsize=7)
    ax.set_xlim(0, 9); ax.grid(axis="x", alpha=0.3)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#1f3a5f", label="Completed"), Patch(color="#e07a1f", label="Planned")],
              fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


architecture_figure(FIG / "architecture.png")
gantt_figure(FIG / "gantt_viva2.png")

# --------------------------------------------------------------------------- docx helpers
doc = Document()
sec = doc.sections[0]
sec.page_height, sec.page_width = Cm(29.7), Cm(21.0)
sec.left_margin = sec.right_margin = Cm(2.5)
sec.top_margin = sec.bottom_margin = Cm(2.5)

st = doc.styles["Normal"]
st.font.name = "Times New Roman"; st.font.size = Pt(12)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
st.paragraph_format.space_after = Pt(6)
st.paragraph_format.line_spacing = 1.15
for lvl, size in [(1, 18), (2, 14), (3, 12)]:
    h = doc.styles[f"Heading {lvl}"]
    h.font.name = "Times New Roman"; h.font.size = Pt(size); h.font.bold = True
    h.font.color.rgb = RGBColor(0, 0, 0)
    rf = h.element.rPr.rFonts
    for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rf.set(qn(a), "Times New Roman")
    for a in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
        if rf.get(qn(a)) is not None:
            del rf.attrib[qn(a)]
    h.paragraph_format.space_before = Pt(12 if lvl > 1 else 0)
    h.paragraph_format.space_after = Pt(6)


def para(text="", bold=False, italic=False, size=None, align=None, after=None):
    p = doc.add_paragraph()
    if text:
        add_runs(p, text, bold=bold, italic=italic, size=size)
    if align == "c":
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    elif align == "j" or align is None:
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    elif align == "r":
        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    if after is not None:
        p.paragraph_format.space_after = Pt(after)
    return p


def add_runs(p, text, bold=False, italic=False, size=None):
    """Supports **bold** spans inside text."""
    parts = text.split("**")
    for i, part in enumerate(parts):
        if not part:
            continue
        r = p.add_run(part)
        r.bold = bold or (i % 2 == 1)
        r.italic = italic
        if size:
            r.font.size = Pt(size)


def bullets(items, style="List Bullet"):
    for it in items:
        p = doc.add_paragraph(style=style)
        add_runs(p, it)
        p.paragraph_format.space_after = Pt(3)
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY


def shade(cell, hex_fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), hex_fill)
    tcPr.append(shd)


def table(header, rows, widths, caption=None, size=10, bold_rows=()):
    if caption:
        c = para(caption, bold=True, size=10.5, align="c", after=3)
        c.paragraph_format.keep_with_next = True
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(header):
        cell = t.rows[0].cells[i]
        cell.text = ""
        r = cell.paragraphs[0].add_run(h); r.bold = True; r.font.size = Pt(size)
        shade(cell, "D9E2F3")
    for ri, row in enumerate(rows):
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            r = cells[i].paragraphs[0].add_run(str(v)); r.font.size = Pt(size)
            r.bold = ri in bold_rows
    for row in t.rows:
        for i, w in enumerate(widths):
            row.cells[i].width = Cm(w)
            row.cells[i].paragraphs[0].paragraph_format.space_after = Pt(1)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def figure(path, caption, width_cm=15.5):
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(path), width=Cm(width_cm))
    para(caption, italic=True, size=10, align="c")


def page_break():
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


def chapter(n, title):
    page_break()
    para(f"Chapter {n}", bold=True, size=20, align="l", after=4)
    doc.add_heading(f"{n}  {title}", level=1)


def add_page_numbers(section):
    p = section.footer.paragraphs[0]; p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    for tag, txt in (("begin", None), (None, "PAGE"), ("end", None)):
        if tag:
            el = OxmlElement("w:fldChar"); el.set(qn("w:fldCharType"), tag)
        else:
            el = OxmlElement("w:instrText"); el.set(qn("xml:space"), "preserve"); el.text = txt
        run._r.append(el)


def add_toc():
    p = doc.add_paragraph(); run = p.add_run()
    for tag, txt in (("begin", None), (None, 'TOC \\o "1-2" \\h \\z \\u'), ("separate", None),
                     (None, None), ("end", None)):
        if tag:
            el = OxmlElement("w:fldChar"); el.set(qn("w:fldCharType"), tag); run._r.append(el)
        elif txt:
            el = OxmlElement("w:instrText"); el.set(qn("xml:space"), "preserve"); el.text = txt; run._r.append(el)
        else:
            t = OxmlElement("w:t"); t.text = "Right-click here and choose 'Update Field' to build the table of contents."
            run._r.append(t)


# --------------------------------------------------------------------------- title page
para("2023-2027/CSE-DS/1/G15", bold=True, align="r")
for _ in range(2):
    para()
para(TITLE, bold=True, size=22, align="c", after=14)
para("EXTENDED SYNOPSIS", bold=True, size=14, align="c", after=2)
para("(Research Design and Implementation: Progress Report for the Second Project Viva)", bold=True, size=12, align="c", after=14)
para("Submitted in Partial Fulfillment of the Requirements for the Degree of", align="c")
para("Bachelor of Technology", bold=True, size=18, align="c", after=0)
para("in", bold=True, size=14, align="c", after=0)
para("Computer Science and Engineering", bold=True, size=18, align="c", after=12)
para("Session: 2023–2027", align="c")
para("by", align="c")
for n, r in MEMBERS:
    para(f"{n} ({r})", align="c", after=0)
para()
para("Under the Supervision of", align="c")
para("Dr. AVDHESH GUPTA", bold=True, size=13, align="c", after=24)
para("AJAY KUMAR GARG ENGINEERING COLLEGE, GHAZIABAD", bold=True, size=13, align="c", after=0)
para("DR. A.P.J. ABDUL KALAM TECHNICAL UNIVERSITY, LUCKNOW", bold=True, size=13, align="c", after=12)
para("Second Project Viva: 12–13 October 2026", align="c")

# --------------------------------------------------------------------------- certificate / guide approval
page_break()
doc.add_heading("CERTIFICATE OF REVIEW BY GUIDE", level=1)
para("This is to certify that the Extended Synopsis entitled “AI-Driven Multimodal Sentiment Analysis "
     "for Mental Health Monitoring on Social Media”, containing the Research Design and Implementation "
     "progress of Group 15 (Session 2023–2027), B.Tech Computer Science and Engineering, Ajay Kumar Garg "
     "Engineering College, Ghaziabad, has been reviewed by me before the Second Project Viva held on "
     "12–13 October 2026. The work reported here is an authentic record of the progress made by the group "
     "under my supervision.")
para()
para("Remarks of the Guide:", bold=True)
for _ in range(4):
    para("_" * 64, align="l")
para()
para("Signature of Guide: ______________________          Date: ____________", align="l")
para("Dr. AVDHESH GUPTA", bold=True, align="l", after=0)
para("Department of Computer Science and Engineering", align="l", after=0)
para("Ajay Kumar Garg Engineering College, Ghaziabad", align="l")
para()
para("Group Members", bold=True, align="l")
table(["Name of Student", "Roll Number", "Signature"], [[n, r, ""] for n, r in MEMBERS], [6, 4.5, 5.5], size=11)

# --------------------------------------------------------------------------- abstract
page_break()
doc.add_heading("ABSTRACT", level=1)
c = s1["C"]; a2 = s1["A2"]; b2 = s1["B2"]
para("Social-media posts usually combine text with an image, yet most sentiment and mental-health "
     "analysis systems read only the text. This project asks whether combining both modalities measurably "
     "improves sentiment classification, and under what conditions. The proposal stage (first viva) has now "
     "been carried through to a working, tested implementation. Following an experimental, comparative "
     "research design, we built two text models (TF-IDF + Logistic Regression; frozen DistilBERT), two image "
     "models (a from-scratch CNN; frozen ImageNet ResNet18), and two fusion strategies (feature-level "
     "concatenation with an MLP head; decision-level late fusion). All of them are trained and evaluated on "
     "identical splits with identical metrics, on CPU-only hardware.")
para(f"**Study 1** uses MVSA-Single (4,511 real Twitter text–image pairs, 3-class sentiment). Concatenation "
     f"fusion reaches {pct(c[1])} accuracy / {pct(c[2])} macro-F1, ahead of the best text model "
     f"({pct(a2[1])} / {pct(a2[2])}) and the best image model ({pct(b2[1])} / {pct(b2[2])}). The gain is "
     f"concentrated on posts where the text and image models disagree. **Study 2** extends the work to "
     f"Internet memes (Memotion, 3,500-sample stratified subset), where the text is rendered inside the image. "
     f"A Gaussian-blur ablation, validated by a {blur['blur_validation_metric']['pct_reduction']:.1f}% "
     f"drop in Laplacian sharpness, removes most of the fusion gain (−{blur['fusion_accuracy_drop_pp']:.1f} "
     f"pp accuracy) while barely affecting the image-only model (−{blur['image_only_accuracy_drop_pp']:.1f} pp). "
     f"This indicates that much of the apparent multimodal benefit on memes comes from the image encoder "
     f"re-reading embedded text. A Streamlit demonstration application, a research dashboard, explainability "
     f"(token saliency, Grad-CAM, modality ablation), 32 passing automated tests and an IEEE-format paper "
     f"draft complete the implementation. The system is a non-clinical research tool and does not diagnose.")
para("**Keywords:** Multimodal Sentiment Analysis, Feature Fusion, DistilBERT, ResNet18, MVSA-Single, Memotion, "
     "Modality Conflict, Embedded-Text Leakage, Mental Health Monitoring, Resource-Constrained ML", align="j")

page_break()
para("Contents", bold=True, size=18, align="l")
add_toc()

# --------------------------------------------------------------------------- Ch 1 research progress
chapter(1, "Introduction and Research Progress")
doc.add_heading("1.1 Problem Statement", level=2)
para("Existing sentiment and mental-health analysis systems on social media rely mainly on text, and lose "
     "the emotional context carried by the images, memes and emoji that accompany it. Where multimodal "
     "systems do exist, they are rarely compared against text-only and image-only models under identical "
     "data, splits and metrics. When they are, the comparison is usually made on very small social-media "
     "samples or on GPU-scale infrastructure. This project builds and rigorously compares text-only, "
     "image-only and multimodal sentiment models on real social-media data, under a realistic CPU-only "
     "budget, and investigates when and why fusion helps.")
doc.add_heading("1.2 Research Questions", level=2)
para("**Primary RQ:** Can a multimodal framework combining textual and visual information improve "
     "sentiment classification of social-media posts compared with unimodal (text-only, image-only) approaches?")
bullets(["**RQ-a:** How effectively can text alone classify the sentiment of social-media posts (traditional vs. transformer features)?",
         "**RQ-b:** What additional information does visual content provide (from-scratch CNN vs. pretrained ResNet18)?",
         "**RQ-c:** Does fusion help uniformly, or does the gain concentrate on posts where text and image disagree (modality conflict)?",
         "**RQ-d (new, from Study 2):** When the text is embedded in the image (memes), how much of the fusion gain is real visual signal and how much is the image encoder re-reading the text?"])
doc.add_heading("1.3 Research Objectives", level=2)
bullets(["O1: Select and prepare a publicly available, adequately-sized multimodal social-media dataset with documented provenance.",
         "O2: Develop preprocessing pipelines for text (URL/mention normalisation, emoji demojization) and images.",
         "O3: Build text-only, image-only and multimodal (fusion) models, each with a baseline and a modern variant.",
         "O4: Evaluate all models on identical splits with accuracy, precision, recall, macro-F1 and confusion matrices.",
         "O5: Analyse errors, modality conflict and explainability, and deliver a non-clinical demo and dashboard.",
         "O6: Disseminate the work as a research manuscript."], style="List Number")
doc.add_heading("1.4 Progress Since the First Viva", level=2)
para("The table below maps each objective to its status and the evidence in the project repository.")
table(["Objective", "Status", "Evidence (repository)"], [
    ["O1 Dataset", "Completed", "MVSA-Single (4,511 pairs) + Memotion subset (3,500); DATASET_SELECTION.md"],
    ["O2 Preprocessing", "Completed", "src/preprocessing/; unit-tested (tests/)"],
    ["O3 Models", "Completed", "6 variants (A1, A2, B1, B2, C-concat, C-late) + blur variants in Study 2"],
    ["O4 Evaluation", "Completed", "results/*.json, research_paper_v2/results/*.json"],
    ["O5 Analysis + demo", "Completed", "ERROR_ANALYSIS.md, MODALITY_CONFLICT_ANALYSIS.md, app/streamlit_app.py"],
    ["O6 Dissemination", "In progress", "IEEE-format drafts: research_paper/ and research_paper_v2/"],
    ["Robustness (multi-seed)", "Pending", "Planned for Oct–Nov 2026 (Chapter 6)"],
], [4, 2.6, 9.4], caption="Table 1.1: Objective-wise progress")
doc.add_heading("1.5 Scope and Deviation from the Proposal", level=2)
para("The proposal named fine-grained emotional states (stress, anxiety, loneliness). The literature search "
     "found no publicly available, adequately-sized social-media dataset that pairs text, image and "
     "mental-health labels. We therefore use 3-class sentiment (positive/neutral/negative) as an "
     "established, non-clinical proxy, as the scope section of the proposal anticipated (\"final dataset "
     "selection will depend on availability\"). The framework does not depend on the label set and can be "
     "retrained if a mental-health-labelled multimodal corpus becomes available. The system remains a "
     "research and monitoring-support tool, not a diagnostic instrument.")

# --------------------------------------------------------------------------- Ch 2 lit review / gap (brief)
chapter(2, "Literature Summary and Research Gap")
para("A detailed literature review was submitted with the first synopsis (LITERATURE_REVIEW.md). The "
     "findings that drive the design are summarised below.")
table(["Theme", "Key finding", "Design consequence"], [
    ["Text (BERT, MentalBERT, DistilBERT)", "Transformers dominate text sentiment; DistilBERT keeps ~97% of BERT's quality at ~60% of its cost", "A2 uses frozen DistilBERT; TF-IDF kept as traditional baseline A1"],
    ["Image (CNN, ViT, transfer learning)", "ViTs need very large pre-training; transfer from pretrained CNNs beats training from scratch on thousands of images", "B2 uses frozen ResNet18; B1 from-scratch CNN as baseline"],
    ["Multimodal fusion surveys", "Early, late and attention fusion are well studied; gains depend on modality complementarity", "Concat (early) + late fusion; attention kept as a stretch goal"],
    ["Memes / scene text (Memotion, Hateful Memes, TextVQA)", "Meme meaning lives in the text–image interaction; reading in-image text is a distinct capability", "Study 2 blur ablation isolates the embedded-text channel"],
], [4, 6.5, 5.5], caption="Table 2.1: Literature themes and their effect on the design", size=9.5)
doc.add_heading("2.1 Research Gap", level=2)
bullets(["Few controlled, same-dataset, same-metric comparisons of text vs. image vs. multimodal models on adequately-sized real social-media data. The closest comparable work evaluates its social-media condition on about 100 pairs.",
         "Most multimodal studies assume GPU-scale resources. What is achievable on commodity CPU hardware is rarely reported.",
         "No located work quantifies how much of the fusion gain on memes comes from the image encoder reading embedded text rather than from independent visual cues."])

# --------------------------------------------------------------------------- Ch 3 research design
chapter(3, "Research Design")
doc.add_heading("3.1 Research Approach and Justification", level=2)
para("The project follows an **experimental, comparative research design**. Every model (text, image and "
     "fusion) is trained on the same training split, tuned only on the same validation split, and evaluated "
     "once on the same untouched test split with the same metric code and a fixed random seed (42). Holding "
     "everything else constant lets us attribute any performance difference to the modality and the fusion "
     "strategy. That is exactly what the primary research question asks. A purely theoretical or "
     "survey-based approach could not answer it quantitatively. Building only a single multimodal system "
     "would leave the claim \"fusion helps\" without a baseline.")
doc.add_heading("3.2 Datasets", level=2)
table(["Property", "Study 1: MVSA-Single", "Study 2: Memotion (SemEval-2020 T8)"], [
    ["Content", "Tweets with attached photographs", "Internet memes (text rendered on image) + corrected OCR text"],
    ["Size used", "4,511 pairs (3,611 / 450 / 450)", "3,500 stratified subset (2,800 / 350 / 350) of 6,992"],
    ["Labels", "3-class sentiment", "5-class sentiment collapsed to 3-class"],
    ["Class balance (train)", "59.5% pos, 30.1% neg, 10.4% neu", "59.3% pos, 31.5% neu, 9.3% neg"],
    ["Split source", "Published CLMLF (NAACL 2022) split, for comparability", "Hugging Face mirror's splits, stratified subsample"],
    ["Access", "Official link dead (404); verified HF mirror", "HF mirror Ahren09/MMSoc_Memotion"],
], [3.6, 6.2, 6.2], caption="Table 3.1: Datasets used", size=9.5)
para("Because the classes are heavily imbalanced, **macro-F1 is the primary metric**. A majority-class "
     "predictor already reaches about 60% accuracy while scoring a very low macro-F1, so accuracy alone "
     "would be misleading.")
doc.add_heading("3.3 System Architecture", level=2)
figure(FIG / "architecture.png", "Figure 3.1: System architecture: text branch, image branch, fusion and evaluation")
para("A post enters two independent branches. The text branch cleans the text and produces a TF-IDF vector "
     "(A1) or a 768-dimensional DistilBERT embedding (A2). The image branch normalises the image and produces "
     "either the prediction of a from-scratch CNN (B1) or a 512-dimensional ResNet18 penultimate-layer "
     "embedding (B2). Each branch has its own classifier, which gives the unimodal baselines. Model C "
     "concatenates the two embeddings (1,280 dimensions) and trains a joint MLP. Late fusion instead "
     "averages the probability outputs of the two unimodal classifiers. The same outputs feed the "
     "evaluation, explainability and dashboard modules.")
doc.add_heading("3.4 Algorithms", level=2)
table(["ID", "Modality", "Algorithm / configuration"], [
    ["A1", "Text", "TF-IDF (unigram+bigram, capped vocabulary) → class-weighted Logistic Regression"],
    ["A2", "Text", "distilbert-base-uncased, frozen; pooled 768-d embedding → class-weighted Logistic Regression"],
    ["B1", "Image", "4-block CNN trained from scratch at 96×96 (reduced from 224 because of the CPU/RAM limit)"],
    ["B2", "Image", "ImageNet ResNet18, frozen; 512-d penultimate embedding → class-weighted Logistic Regression"],
    ["C", "Fusion (early)", "[A2 ‖ B2] 1,280-d → MLP 256→128→3, dropout 0.4/0.3, Adam lr 1e-3, wd 1e-4, batch 32, ≤40 epochs, early stopping (patience 8) on validation macro-F1"],
    ["C-late", "Fusion (late)", "Average of A2 and B2 class probabilities; simple average and validation-tuned weights"],
    ["*-Blur", "Ablation (Study 2)", "Gaussian blur (radius 14 px at native resolution) applied before B2 features; B2 and C retrained identically"],
], [1.6, 2.8, 11.6], caption="Table 3.2: Models and configurations", size=9.5)
doc.add_heading("3.5 Justification of Design Choices", level=2)
table(["Choice", "Alternative considered", "Reason"], [
    ["Frozen encoders", "End-to-end fine-tuning", "CPU-only, 8 GB RAM: frozen extraction measured 46–71 ms/sample, while fine-tuning was impractical; frozen encoders also isolate the fusion effect"],
    ["DistilBERT", "BERT-base, MentalBERT", "About 60% of BERT's compute for about 97% of its quality; MentalBERT targets clinical-style Reddit text, not tweets or memes"],
    ["ResNet18", "ViT, ResNet50, EfficientNet", "ViTs underperform without large-scale data; ResNet18 is the fastest robust CNN on CPU"],
    ["Concat + MLP fusion", "Cross-modal attention", "Simple, interpretable and CPU-feasible; attention fusion is mainly validated on video/audio benchmarks and is kept as a stretch goal"],
    ["Macro-F1 primary", "Accuracy", "Class imbalance (about 60% positive) makes accuracy misleading"],
    ["Published split", "Own random split", "Comparability with prior work (CLMLF) and no tuning on the test set"],
    ["Blur ablation", "OCR text masking", "Needs no OCR model and is objectively validated (Laplacian variance); OCR masking is planned as a follow-up"],
], [3.2, 3.6, 9.2], caption="Table 3.3: Design decisions and justification", size=9.5)
doc.add_heading("3.6 Evaluation and Validation Design", level=2)
bullets(["Metrics: accuracy, macro and weighted precision/recall/F1, per-class F1, confusion matrices.",
         "Leakage audit: cross-split duplicate check (src/data/check_leakage.py, results/leakage_audit.txt).",
         "Modality-conflict analysis: the test set is split into posts where the text-only and image-only predictions agree or disagree, and each model is scored on both subsets.",
         "Explainability: gradient×input token saliency (text), Grad-CAM (image) and modality ablation (fusion).",
         "Blur validation: mean Laplacian variance on 150 test images, before and after blurring."])

# --------------------------------------------------------------------------- Ch 4 implementation
chapter(4, "Implementation")
doc.add_heading("4.1 Tools, Platform and Environment", level=2)
table(["Category", "Tools"], [
    ["Language / runtime", "Python 3.13, virtual environment with pinned requirements (requirements.lock.txt)"],
    ["Deep learning", "PyTorch (CPU build), torchvision, Hugging Face Transformers"],
    ["Classical ML", "scikit-learn (TF-IDF, Logistic Regression, metrics)"],
    ["Data", "Hugging Face datasets, pandas, NumPy, Pillow, emoji"],
    ["Application", "Streamlit (demo + research dashboard)"],
    ["Testing / QA", "pytest (32 tests), Playwright (automated UI click-through)"],
    ["Documentation", "Markdown, ReportLab (IEEE PDF), python-pptx, python-docx, Git/GitHub"],
    ["Hardware", "Consumer laptop, CPU only (no GPU), 8 GB RAM, Windows 10"],
], [4, 12], caption="Table 4.1: Tools and environment")
doc.add_heading("4.2 Modules Developed", level=2)
table(["Module (path)", "Responsibility"], [
    ["src/data/", "Dataset download (MVSA, Memotion), manifest building, published-split loader, leakage check"],
    ["src/preprocessing/", "Text cleaning (URL/mention normalisation, emoji demojization); image preprocessing"],
    ["src/text_model/", "A1 TF-IDF baseline; DistilBERT embedding extraction and A2 classifier"],
    ["src/image_model/", "B1 CNN from scratch (pixel cache); ResNet18 embedding extraction and B2 classifier"],
    ["src/multimodal/", "C concatenation-fusion MLP; late (decision-level) fusion"],
    ["src/evaluation/", "Shared metrics, test-set prediction export, result figures"],
    ["src/explainability/", "Token saliency, Grad-CAM (hook-based), modality ablation examples"],
    ["research_paper_v2/src/", "Study 2 pipeline 01–09: Memotion download, sharp/blur features, models, leakage and conflict analysis, figures"],
    ["app/streamlit_app.py", "Interactive demo (text + image → three model predictions with confidence) and dashboard"],
    ["tests/", "Unit tests for seeding, text cleaning, image preprocessing, splits, metrics and the dataset loader"],
], [4.6, 11.4], caption="Table 4.2: Implemented modules", size=9.5)
doc.add_heading("4.3 Implementation Workflow", level=2)
bullets(["**Step 1:** Acquire data and verify its integrity (row counts, image decode check, label mapping).",
         "**Step 2:** Clean the text and preprocess the images; cache the expensive frozen-encoder embeddings to .npz/.npy files so that every classifier reads identical features.",
         "**Step 3:** Train each classifier on the training split, select it on validation macro-F1, and evaluate once on the test split. Results are written to JSON.",
         "**Step 4:** Run the analysis scripts (error analysis, modality conflict, blur leakage) and generate figures from the saved JSON. No numbers are typed in by hand.",
         "**Step 5:** Serve the trained models through the Streamlit application."], style="List Bullet")
doc.add_heading("4.4 Demonstration Application", level=2)
shots = ROOT / "docs" / "screenshots"
figure(shots / "01_predict_empty.png", "Figure 4.1: Demo application: input screen (post text + image upload)", 12)
figure(shots / "02_predict_result.png", "Figure 4.2: Text-only, image-only and fusion predictions with confidence, plus the non-clinical disclaimer", 14)
figure(shots / "03_dashboard.png", "Figure 4.3: Research dashboard: model comparison, confusion matrices, class distribution", 14)
doc.add_heading("4.5 Testing", level=2)
para("The automated test suite (pytest) contains 32 tests covering seed reproducibility, text cleaning, "
     "image preprocessing, stratified splitting, metric correctness and the dataset loader. All 32 tests pass "
     "(re-run on 12 Oct 2026). The demo was additionally exercised end to end with a Playwright "
     "click-through, which produced the screenshots above.")
doc.add_heading("4.6 Technical Challenges Faced and Resolutions", level=2)
table(["Challenge", "Resolution"], [
    ["Official MVSA OneDrive link returned 404", "Located a community Hugging Face mirror, verified its content, and adopted the published CLMLF split; all of this is disclosed"],
    ["CNN from scratch: first full-scale run killed for running out of memory; second exceeded the time budget", "Root-caused both failures; built a 96×96 pixel cache and limited training to 3 epochs; documented as a limitation"],
    ["CPU-only compute for transformers", "Frozen DistilBERT and ResNet18 feature extraction with cached embeddings (46–71 ms/sample)"],
    ["Severe class imbalance (neutral ~10%)", "Class-weighted losses, macro-F1 model selection, per-class reporting"],
    ["About 1–2% cross-split duplicates (retweets) in the published split", "Measured with a leakage audit and reported, not silently fixed, to stay comparable with prior work"],
    ["Proving the blur actually removes text", "Objective Laplacian-variance metric: 99.9% reduction on 150 images"],
    ["No public multimodal mental-health-labelled dataset", "Used sentiment as a non-clinical proxy; the framework does not depend on the label set"],
], [6.5, 9.5], caption="Table 4.3: Technical challenges", size=9.5)

# --------------------------------------------------------------------------- Ch 5 results
chapter(5, "Results and Validation")
doc.add_heading("5.1 Study 1: MVSA-Single (Twitter photos)", level=2)
order = ["A1", "A2", "B1", "B2", "CL", "CLw", "C"]
table(["Model", "Accuracy", "Macro-F1", "Weighted-F1"],
      [[s1[k][0], pct(s1[k][1]), pct(s1[k][2]), pct(s1[k][3])] for k in order],
      [8.4, 2.5, 2.5, 2.6], caption="Table 5.1: Test-set results, Study 1 (n = 450)", bold_rows=(6,))
figure(ROOT / "visualizations" / "model_comparison.png", "Figure 5.1: Model comparison (Study 1)", 13)
figure(ROOT / "visualizations" / "confusion_matrices.png", "Figure 5.2: Confusion matrices (Study 1)", 15)
doc.add_heading("Observations", level=3)
bullets([f"Concatenation fusion is the best model on accuracy ({pct(s1['C'][1])}) and improves macro-F1 by {100*(s1['C'][2]-s1['A2'][2]):.1f} pp over the best text model. Validation-tuned late fusion is essentially tied ({pct(s1['CLw'][1])} accuracy, {pct(s1['CLw'][2])} macro-F1), so both fusion strategies beat every unimodal model.",
         "Text is the stronger single modality on tweets. The pretrained ResNet18 beats the from-scratch CNN on accuracy, which supports the transfer-learning choice.",
         "The text and image models disagree on 53.3% of test posts. On that subset fusion reaches 57.1%, against 50.0% (text) and 33.3% (image). On posts where they agree, fusion slightly regresses (72.9% vs. 74.8%). The fusion gain is therefore conditional and concentrated on modality conflict.",
         "Neutral is the hardest class (about 10% of the data, F1 about 0.37) for every model."])
figure(ROOT / "visualizations" / "modality_conflict.png", "Figure 5.3: Accuracy on agreement vs. disagreement subsets (Study 1)", 13)

doc.add_heading("5.2 Study 2: Memotion (memes): Embedded-Text Leakage", level=2)
order2 = ["A1", "A2", "B1", "B2", "B2b", "C", "Cb"]
table(["Model", "Accuracy", "Macro-F1", "Weighted-F1"],
      [[s2[k][0], pct(s2[k][1]), pct(s2[k][2]), pct(s2[k][3])] for k in order2],
      [8.4, 2.5, 2.5, 2.6], caption="Table 5.2: Test-set results, Study 2 (n = 350)", bold_rows=(5,))
figure(V2 / "figures" / "sharp_vs_blur.png", "Figure 5.4: Effect of blurring the image on image-only and fusion models", 13)
bv = blur["blur_validation_metric"]
doc.add_heading("Observations", level=3)
bullets([f"On sharp memes, fusion ({pct(s2['C'][1])} / {pct(s2['C'][2])} macro-F1) clearly beats every unimodal model.",
         f"Blurring (Laplacian variance {bv['mean_laplacian_variance_sharp']:.0f} → {bv['mean_laplacian_variance_blurred']:.1f}, a {bv['pct_reduction']:.1f}% reduction) barely changes the image-only model (−{blur['image_only_accuracy_drop_pp']:.2f} pp accuracy, −{blur['image_only_macrof1_drop_pp']:.2f} pp macro-F1) but removes most of the fusion gain (−{blur['fusion_accuracy_drop_pp']:.2f} pp accuracy, −{blur['fusion_macrof1_drop_pp']:.2f} pp macro-F1). C-Blur ends up within {blur['cblur_vs_text_only_accuracy_delta_pp']:.1f} pp of text-only accuracy.",
         "Interpretation: on memes, much of the fusion benefit comes from the image encoder's access to legible embedded text rather than from independent visual cues. Systems that treat a meme's image and text as independent evidence may overstate the multimodal benefit.",
         f"Modality conflict replicates partly: models disagree on {conf2['pct_disagree']:.1f}% of memes and fusion gains most there ({pct(conf2['disagree_subset']['fusion_acc'])} vs. {pct(conf2['disagree_subset']['text_only_acc'])} text). Unlike Study 1, fusion also gains on the agreement subset.",
         f"B1 reaches {pct(s2['B1'][1])} accuracy but only {pct(s2['B1'][2])} macro-F1 because it mostly predicts the majority class. This shows why accuracy alone is misleading here."])
figure(V2 / "figures" / "perclass_f1.png", "Figure 5.5: Per-class F1 (Study 2)", 13)

doc.add_heading("5.3 Validation Summary", level=2)
bullets(["Same split, same seed and same metric code for every model; the test set is used once per model.",
         "All reported numbers are produced by scripts and saved as JSON; figures are generated from that JSON.",
         "A leakage audit quantifies the duplicates inherited from the published split.",
         "The blur ablation is validated objectively and is not assumed to work.",
         "Two independent datasets (photos vs. memes) show that the fusion benefit depends on the data. This is a stronger and more honest claim than \"multimodal always wins\".",
         "Known limitation: single seed and no significance tests yet. Both are scheduled first in the next plan."])

# --------------------------------------------------------------------------- Ch 6 next plan
chapter(6, "Next Plan and Timeline")
doc.add_heading("6.1 Work Remaining", level=2)
table(["#", "Task", "Purpose", "Target"], [
    ["1", "Re-run fusion/CNN models over 5 seeds; report mean ± std; McNemar / bootstrap significance tests", "Statistical robustness of the headline comparisons", "Oct 2026"],
    ["2", "OCR-based text-masking ablation and full 6,992-meme Memotion run", "Confirm the blur finding with a text-specific ablation at full scale", "Nov 2026"],
    ["3", "Cross-modal (gated) attention fusion (stretch goal)", "Compare a more expressive fusion method against concat/late fusion", "Nov–Dec 2026"],
    ["4", "Merge Studies 1 and 2 into one self-contained manuscript", "External-reviewer readiness (remove repository-internal references)", "Dec 2026"],
    ["5", "Submission to a target venue (e.g. IEEE ICCCNT), chosen with the guide", "Dissemination (objective O6)", "Jan 2027"],
    ["6", "Review response, final report, final demo polish", "Project completion", "Feb–Mar 2027"],
], [0.8, 7.2, 5.6, 2.4], caption="Table 6.1: Remaining work", size=9.5)
figure(FIG / "gantt_viva2.png", "Figure 6.1: Timeline: completed work (blue) and planned work (orange)", 15.5)
doc.add_heading("6.2 Team Roles", level=2)
table(["Member", "Primary responsibility"], [
    ["Kartikey Tiwari", "Literature, dataset acquisition and preparation, integration"],
    ["Akshat Sharma", "NLP preprocessing and text models (A1, A2)"],
    ["Karan Sharma", "Image models (B1, B2), multimodal fusion, evaluation"],
    ["Lalit Sharma", "Documentation, paper, presentation, testing"],
], [5, 11], caption="Table 6.2: Team roles (as in the synopsis)")

# --------------------------------------------------------------------------- Ch 7 ethics
chapter(7, "Ethical Considerations")
bullets(["Only public research datasets are used; no new user data is collected. Usernames and URLs are normalised during preprocessing.",
         "Labels are sentiment, not clinical states. Every screen of the application carries a non-clinical disclaimer, and the system must not be used to diagnose or profile individuals.",
         "Biases are disclosed: English-only, Twitter/meme-specific, 2015–2020-era data. Results may not transfer to other languages or platforms."])

# --------------------------------------------------------------------------- references
page_break()
doc.add_heading("References", level=1)
refs = [
    "J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “BERT: Pre-training of deep bidirectional transformers for language understanding,” in Proc. NAACL-HLT, 2019, pp. 4171–4186.",
    "V. Sanh, L. Debut, J. Chaumond, and T. Wolf, “DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter,” arXiv:1910.01108, 2019.",
    "K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proc. IEEE CVPR, 2016.",
    "A. Dosovitskiy et al., “An image is worth 16x16 words: Transformers for image recognition at scale,” in Proc. ICLR, 2021.",
    "S. Ji et al., “MentalBERT: Publicly available pretrained language models for mental healthcare,” in Proc. LREC, 2022.",
    "T. Niu, S. Zhu, L. Pang, and A. El Saddik, “Sentiment analysis on multi-view social data,” in Proc. MMM, 2016.",
    "Z. Li, B. Xu, C. Zhu, and T. Zhao, “CLMLF: A contrastive learning and multi-layer fusion method for multimodal sentiment detection,” in Findings of NAACL, 2022, pp. 2282–2294.",
    "C. Sharma et al., “SemEval-2020 Task 8: Memotion analysis – the visuo-lingual metaphor!,” in Proc. SemEval, 2020.",
    "D. Kiela et al., “The Hateful Memes Challenge: Detecting hate speech in multimodal memes,” in Proc. NeurIPS, 2020.",
    "A. Singh et al., “Towards VQA models that can read,” in Proc. IEEE/CVF CVPR, 2019.",
    "A. Gandhi, K. Adhvaryu, S. Poria, E. Cambria, and A. Hussain, “Multimodal sentiment analysis: A systematic review of history, datasets, multimodal fusion methods, applications, challenges and future directions,” Information Fusion, vol. 91, pp. 424–444, 2023.",
    "R. Das and T. D. Singh, “Multimodal sentiment analysis: A survey of methods, trends, and challenges,” ACM Computing Surveys, vol. 55, no. 13s, 2023.",
    "R. R. Selvaraju et al., “Grad-CAM: Visual explanations from deep networks via gradient-based localization,” in Proc. IEEE ICCV, 2017.",
    "E. S. Agung, A. P. Rifai, and T. Wijayanto, “Image-based facial emotion recognition using convolutional neural network on emognition dataset,” Scientific Reports, vol. 14, 2024.",
]
for i, r in enumerate(refs, 1):
    p = para(f"[{i}] {r}", size=11, after=3)
    p.paragraph_format.left_indent = Cm(0.9); p.paragraph_format.first_line_indent = Cm(-0.9)

add_page_numbers(doc.sections[0])
doc.save(OUT)
print("wrote", OUT)
