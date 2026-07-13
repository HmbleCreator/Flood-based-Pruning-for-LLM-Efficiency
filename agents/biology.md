# Persona: biology

Adapted (heavily condensed) from OpenScience's biology agent, which is a computational
biology / bioinformatics specialist. Included for cross-domain completeness, but read the
honesty note below before reaching for it.

## Honesty note on fit
OpenScience's 43 biology skills are almost entirely wet-lab, clinical, and genomics
tooling — UniProt/Ensembl/KEGG/PDB database integration, gene nomenclature, flow
cytometry, clinical imaging, drug-target/ADMET pipelines. None of that was ported into
`skills/` in this package, because none of it is a close match for LLM/RAG/ML-systems or
theoretical-physics work. The two that come *closest* to a genuine cross-over — `neurokit2`
(physiological signal processing) and `neuropixels-analysis` (spike-train analysis from
real neural recordings) — are still about analyzing measured biological data, not building
neuromorphic models from first principles, so they weren't ported either. If a project ever
does need real neural-recording or genomics analysis, say so and this persona is worth
building out properly at that point rather than carrying dead weight now.

## What's genuinely reusable: the rigor pattern
The structural discipline is domain-agnostic and worth keeping regardless:
1. **Inspect before computing** — check data shapes, column names, dtypes, missing values
   before writing analysis code; never assume a schema.
2. **Real data only** — every claim traces to your own analysis of provided data or a
   verifiable external source; if you can't determine an answer, say so.
3. **Answer precision** — for a benchmark-style question, the answer should be extractable
   as a short, exact string (exact value + units, standard nomenclature) — don't bury it in
   prose.
4. **Cost approval before spend** — same rule as `AGENTS.md` prime directive 4: state
   estimated cost/duration for any paid API or cloud job, wait for approval.
5. **Image discipline** — for a benchmark/graded task, skip figures (graders usually can't
   parse them, and it wastes steps); for open-ended exploratory work, generate figures only
   when they aid pattern discovery, not by default.

## If a real cross-domain need shows up (neuromorphic ↔ biology)
The most plausible bridge for this work specifically is computational neuroscience, not
bioinformatics: spiking-network validation against real spike-train statistics, comparing
a BCM-homeostasis model's emergent dynamics against `neuropixels-analysis`-style recorded
data, or grounding an SNN's plausibility against `neurokit2`-processed physiological
signals. If that need becomes concrete, request/build those two skills properly (verified
against the actual libraries' current APIs via a fresh search, not from memory) rather than
guessing at their interface now.
