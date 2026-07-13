# /write-paper

Delegate to the `agents/write.md` persona to produce a publication-ready document from
existing research artifacts.

Target document / venue: $ARGUMENTS

Preconditions (check before starting — if missing, do the missing piece first or tell the
user what's missing rather than fabricating content):
- `literature-review.md`, `reasoning.md`, and `methodology.md` exist and are current.
- Results and figures from Stage 6 (ANALYZE) exist and are real (not placeholders).

Then run the write.md workflow: skeleton -> section-by-section draft (only verified
citations) -> figures -> compile -> visual PDF check -> self-review via
`agents/reviewer.md` -> fix flagged issues -> present final.
