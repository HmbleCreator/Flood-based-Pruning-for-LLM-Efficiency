# /research

Run the full Research & Build Loop defined in `AGENTS.md` on the topic below, start to
finish, without pausing to ask "should I continue" between stages (you may still ask
clarifying questions in Stage 1, and you must stop at both critique gates for approval if
they return BLOCKING).

Topic / task: $ARGUMENTS

Steps:
1. Read `AGENTS.md` and `templates/*.md` if you haven't already this session.
2. Stage 1 (SCOPE): restate the question and success criteria back to the user in 2-3
   sentences before proceeding; ask only if something material is ambiguous.
3. Run Stages 2-4, writing `literature-review.md`, `reasoning.md`, `methodology.md`.
4. Gate 1 critique (`agents/critique.md`) on the methodology before any code/compute.
5. Stage 5 (BUILD/COMPUTE) — use `agents/ml-engineer.md` if this is a software/ML build,
   otherwise run the methodology's pipeline directly.
6. Gate 2 critique on the outputs.
7. Stages 6-7 (ANALYZE, SYNTHESIZE), updating `research-state.md` throughout.
8. If a final document is requested, hand off to `agents/write.md` for Stage 8.

Create/update `research-state.md` at every stage transition so the session survives a
context compaction or restart.
