---
name: citation-management
description: Verify citations are real and generate correct BibTeX - DOI/author/year/venue checks, fixing malformed entries, deduplicating a .bib file, and catching the specific mismatches that break LaTeX compilation. Use whenever adding or auditing references in a paper.
category: writing
---

# Citation Management

## Verification, not just formatting
A citation isn't "done" when it has a BibTeX entry - it's done when the entry is verified
against a real source. Before adding any entry: confirm the DOI resolves (or the paper is
findable via search), and that author names, year, venue, and title match.

## BibTeX hygiene
- One canonical entry per paper. If the same paper appears under two keys (e.g. a preprint
  and its published version cited separately), consolidate to one - prefer the
  peer-reviewed version's metadata, note the preprint if it's independently relevant.
- Keys should be stable and predictable (`author_year_firstword` or similar) so `\cite{}`
  calls don't silently break when the bib file is regenerated.
- Common breakage to check for: unescaped special characters in titles (`&`, `%`, `_`),
  missing required fields for the entry type (`@inproceedings` needs `booktitle`, not
  `journal`), and mismatched braces that fail silently until `bibtex` is run.

## When citing a specific claim
- Quote or closely paraphrase what the source *actually* says, not what you expect it to
  say - overstating a source ("X proves Y" when X only suggests Y in one setting) is a
  citation integrity problem, not a formatting one.
- If a number is attributed to a citation, that number should appear in the source; if you
  can't confirm it, mark `[VERIFY]` rather than trusting the number from memory.

## Known corrections worth checking for (common venue/year drift)
Citations to well-known infra papers sometimes carry a stale year across drafts (e.g. a
paper first appearing as a preprint gets cited by preprint year even after the peer-reviewed
version published a year later, or vice versa). When reusing a `references.bib` across
drafts, spot-check a handful of entries against the actual publisher page rather than
trusting that an earlier pass got them right.

## Workflow
1. Before adding a new citation: search for it, confirm it's real, pull correct metadata.
2. Generate the BibTeX entry from verified metadata (don't hand-type from memory).
3. Before submission: pass through the full `.bib` file once, checking each entry that's
   actually `\cite{}`-d in the document resolves and matches its usage.
