---
name: latex-paper
description: Compile and debug LaTeX papers - venue template compliance (elsarticle, cas-sc, IEEEtran, etc.), the pdflatex/bibtex compile cycle, common compilation error patterns, and visually verifying the output PDF. Use whenever building or fixing a LaTeX manuscript.
category: writing
---

# LaTeX Paper Compilation

## Compile cycle
Standard cycle for a document with citations: `pdflatex -> bibtex -> pdflatex -> pdflatex`.
The two trailing `pdflatex` passes are not redundant - the first resolves citation keys
into the `.bbl`, the second resolves cross-references and the "Rerun to get citations
correct" warnings that result from that.

## Common failure patterns and their real cause
- **"Undefined control sequence" right after a package-specific command** - usually a
  missing `\usepackage{}` or a class (e.g. `cas-sc`, `elsarticle`) that doesn't include a
  package the draft assumes is loaded (like `amsmath` or `graphicx`).
- **Abstract environment parsing errors** - some journal classes (e.g. Elsevier's `cas-sc`)
  require the abstract in a specific structured block distinct from the plain
  `\begin{abstract}...\end{abstract}` used by generic classes; check the class's own
  sample file rather than assuming the generic syntax works.
- **Bibliography key mismatches after a citation year correction** - if a source's
  publication year was corrected (e.g. a paper that circulated first as a workshop preprint
  and later published under a different year), update both the `.bib` entry's `year` field
  and any `\cite{}` key that encodes the year, or the key and the rendered year will
  silently diverge.
- **Figures/tables floating to the wrong place or the wrong page** - often a sign of too
  many `[h]`-forced floats fighting each other; let LaTeX's float algorithm work with
  `[htbp]` rather than forcing placement, and use `\clearpage` sparingly at true section
  boundaries only.

## Venue compliance
Different venues (IEEE, Elsevier `cas-sc`/`elsarticle`, ACL, NeurIPS) have distinct margin,
font, and reference style requirements, and some enforce a hard page or word limit that
LaTeX won't warn you about - it will just compile past it. After a first successful
compile, explicitly check against the venue's stated limits (not just "did it compile").

## Visual verification (always do this, don't skip)
Never assume a successful `pdflatex` run means the output looks right. Convert the PDF to
images and inspect each page for: text/figure overlaps, orphaned headings at the bottom of
a page, caption-to-figure distance, and bibliography formatting matching the venue style.
Fix and recompile; cap iteration at ~3 passes before flagging to the user if issues persist.

## Precision under edits
When tightening a section to fit a word/page limit, re-verify every number that survived
the edit still matches the source computation - a copyedit pass is a common place for a
metric to quietly get rounded, transposed, or dropped a decimal place.
