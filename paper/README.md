# Urban Change Detection: Literature Review + GReST-CD Research Proposal

## Contents
- `urban_change_detection_proposal.tex` — complete LaTeX manuscript (literature review, original proposed methodology, planned evaluation, and carefully identified unmeasured results)
- `proposal_sections.tex` — new proposed-methodology/experiment/results sections only (for insertion into an existing paper)
- `references.bib` — BibLaTeX references
- `urban_change_detection_proposal.pdf` — compiled 16-page reading copy

## Compile
Use `pdfLaTeX` + `Biber`, with the main document `urban_change_detection_proposal.tex`:

```bash
pdflatex urban_change_detection_proposal.tex
biber urban_change_detection_proposal
pdflatex urban_change_detection_proposal.tex
pdflatex urban_change_detection_proposal.tex
```

Or run `latexmk -pdf urban_change_detection_proposal.tex` when `biber` is available.

## Important research-integrity note
GReST-CD is a proposal, **not an implemented or empirically validated method**. The results section supplies a mathematical sanity check, falsifiable hypotheses, and an empirical-results template. It **does not** report invented F1, IoU, benchmarks, or ablation gains. To publish empirical claims, implement the model and run the documented evaluation using appropriately licensed remote-sensing datasets and ground truth.

## Research goal
Assess whether joint nuisance-sensitive quality control, bounded geometry correction, direction-aware supervision when available, optional DSM fusion, and calibrated uncertainty improve urban building change detection on held-out cities.
