# CHRONO-Fair Overleaf package

Self-contained LaTeX sources for the CHRONO-Fair manuscript.

## Contents

- `main.tex` : the manuscript (ACM `acmart` journal class, `acmsmall`).
- `references.bib` : the bibliography, 116 cited Q1/A* references.
- `figures/` : all 22 figures referenced by `main.tex`.
- `main.pdf` : the compiled output (35 pages).

## Compile on Overleaf

1. Create a new Overleaf project and upload `main.tex`, `references.bib`,
   and the `figures/` folder.
2. Set the compiler to `pdfLaTeX` and the main document to `main.tex`.
3. Overleaf runs the pdfLaTeX, BibTeX, pdfLaTeX, pdfLaTeX sequence
   automatically.

## Compile locally

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Requires a TeX Live install with the `acmart` class
(`texlive-publishers` on Debian or Ubuntu).

## Notes

- The class is `\documentclass[acmsmall,nonacm,review,anonymous]{acmart}`.
  The `anonymous` option blanks the author block for double-blind review.
  Remove it for a single-blind or camera-ready version.
- Figure paths are local (`figures/...`), so the package is portable.
- The bibliography style is `ACM-Reference-Format`. Some entries lack page
  numbers or publishers; fill these before camera-ready submission.
