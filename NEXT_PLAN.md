# Next Plan (prepared for the Second Project Viva, 12–13 Oct 2026)

Honest status check against the synopsis's own timeline (`PROJECT_REQUIREMENTS_ANALYSIS.md`
"Timeline" table) and the mandatory-requirements checklist: as of this viva, **every
mandatory requirement is implemented, tested, and documented** (`FINAL_PROJECT_AUDIT.md`)
— which is ahead of the synopsis's own Oct/Nov 2026 "implementation + experimentation"
milestones. This document states what is genuinely still open, not invented busywork to
fill a slide.

## What is already done (progress-based evidence for this viva)

- All 6 model variants (A1, A2, B1, B2, C-concat, C-late) trained and evaluated on
  identical splits — `results/*.json`.
- Full error analysis, modality-conflict analysis, and explainability — `ERROR_ANALYSIS.md`,
  `MODALITY_CONFLICT_ANALYSIS.md`, `results/explainability_examples/`.
- Working, interactively-tested demo app with a real click-through (text + image upload,
  Analyze, dashboard) — `docs/screenshots/`.
- Real result figures generated from the actual run data — `visualizations/`.
- Research paper drafted and typeset in IEEE two-column format — `research_paper/paper_IEEE.pdf`.
- Presentation content and an actual slide deck — `PRESENTATION_CONTENT.md`, `PRESENTATION.pptx`.
- 32/32 automated tests passing — `TESTING.md`.
- Full viva Q&A prep — `VIVA_PREPARATION.md`.
- **Study 2 (extension): embedded-text leakage in memes** — full pipeline on a 3,500-meme
  stratified Memotion subset with a sharp-vs-blurred image ablation (`research_paper_v2/`).
  Fusion 52.6% acc / 36.6% macro-F1 on sharp memes; blurring costs image-only only 1.4 pp
  accuracy but fusion 10.9 pp, i.e. most of the fusion gain on memes comes from the image
  encoder re-reading printed text. Second IEEE-format draft: `research_paper_v2/paper_IEEE.pdf`.
- **Viva-II documents** — `Extended_Synopsis_Viva2.docx/.pdf` (Research Design +
  Implementation chapters; built by `docs/generate_extended_synopsis.py`) and
  `Viva2_Self_Evaluation_Form.docx/.pdf` (built by `docs/generate_self_evaluation.py`).
  Both need printing and the guide's signature. The self-evaluation content follows the six
  criteria in the viva notice; transfer it to the official form on the CSE Project Central
  site if that template's layout differs.

## What is genuinely still open

1. **Statistical robustness.** Only a single random seed was used for the neural models
   (fusion MLP, CNN). Run-to-run variance was not characterized. This is flagged as a
   known limitation, not hidden — see `FINAL_PROJECT_AUDIT.md` "Known limitations." If
   time permits before the viva, re-running the fusion model and CNN across 3–5 seeds and
   reporting mean ± std would strengthen the headline comparison; this is optional per
   the synopsis's own "statistical validity extras" scoping (`PROJECT_REQUIREMENTS_ANALYSIS.md`
   §Optional/Extensible Features), not mandatory.
2. **Cross-modal attention fusion (stretch goal).** Explicitly deferred in
   `PROJECT_ARCHITECTURE.md` as "attempted only as a stretch experiment after the required
   A/B/C comparison... time/compute permitting." Not attempted yet. Decision for after
   this viva: attempt it only if compute/time allow without risking the already-complete
   required comparison; otherwise keep it as documented future work (already stated in
   `paper.md` §9).
3. **Manuscript readiness for external submission.** `research_paper/paper_IEEE.pdf` is
   complete and internally consistent, but it still points to internal project filenames
   (e.g., "see REPRODUCIBILITY.md") as a stand-in for a technical appendix. Before any
   real conference/journal submission (synopsis mentions IEEE ICCCNT as an example target,
   per the Jan 2027 "manuscript/patent draft" milestone), those references need to become
   self-contained inline detail or footnotes, since an external reviewer won't have the
   repository.
4. **Live rehearsal.** The demo has been click-tested automatically (Playwright), but the
   team should each personally run `./venv/Scripts/streamlit run app/streamlit_app.py`,
   click through it live, and rehearse narrating the modality-conflict finding and the
   resource-constraint story before presenting — an automated run is not a substitute for
   being comfortable live under Q&A.
5. **Supervisor review pass.** Per the viva notice: "All groups must coordinate with their
   respective mentors and ensure that the work is reviewed before appearing for the viva."
   Not something this repository can do on its own — needs an actual meeting with
   Dr. Avdhesh Gupta before 12 Oct.

## Timeline

| Date | Task |
|---|---|
| 12–13 Oct 2026 | **Second Project Viva**: progress, design, implementation, results (both studies), this plan |
| Oct 2026 | Multi-seed runs (5 seeds, mean ± std) for both studies + McNemar / bootstrap significance tests (item 1) |
| Nov 2026 | Study 2 follow-up: OCR-based text-masking ablation (a text-specific check of the blur finding) and full 6,992-meme Memotion run |
| Nov–Dec 2026 | Cross-modal attention fusion stretch experiment (item 2), only if it does not put the completed comparison at risk |
| Dec 2026 | Merge Studies 1 and 2 into one self-contained manuscript (item 3) |
| Jan 2027 | Manuscript submission (per synopsis), venue chosen with the supervisor |
| Feb–Mar 2027 | Peer-review response, revisions, final report (per synopsis) |

This plan is intentionally conservative: it does not promise new mandatory functionality,
because none is missing. It lists real, bounded extensions and the one non-technical
action item (supervisor sign-off) that only the team can complete.
