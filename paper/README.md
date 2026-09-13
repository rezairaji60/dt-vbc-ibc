# SCL manuscript

Open `main.tex` in Overleaf. Standard elsarticle, elsarticle-num, TikZ and BibTeX are used. The figure is specified directly from the mathematical example in the source.

```sh
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

`main.bbl` is generated convenience output; `references.bib` is authoritative. The table reports exact-replay results, not historical collocation values. See `AUTHOR_CHECKLIST.md` for required author decisions before submission.
