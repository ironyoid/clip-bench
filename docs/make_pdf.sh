#!/bin/bash
pandoc metrics.md -o metrics.pdf \
  --pdf-engine=pdflatex \
  -V geometry:margin=1.5cm \
  -V geometry:a4paper \
  -H header.tex \
  --resource-path=.
