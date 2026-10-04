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
| Now – 11 Oct 2026 | Supervisor review pass (item 5); live demo rehearsal (item 4); optional multi-seed robustness run (item 1) if time allows |
| 12–13 Oct 2026 | **Second Project Viva** — present progress, design, implementation, results, this plan |
| Oct–Nov 2026 | Per synopsis's own schedule this is "implementation completion / experimentation" — already complete here, so this window is redirected to: multi-seed runs (if not done pre-viva), and, time permitting, the cross-modal-attention stretch experiment (item 2) |
| Dec 2026 | Results analysis / manuscript drafting — already drafted; this window becomes manuscript *refinement* for external readability (item 3) |
| Jan 2027 | Manuscript/patent draft submission (per synopsis) — contingent on item 3 being complete and a target venue being chosen with the supervisor |
| Feb–Mar 2027 | Peer-review response, revisions, publication, final report (per synopsis) |

This plan is intentionally conservative: it does not promise new mandatory functionality,
because none is missing. It lists real, bounded extensions and the one non-technical
action item (supervisor sign-off) that only the team can complete.
