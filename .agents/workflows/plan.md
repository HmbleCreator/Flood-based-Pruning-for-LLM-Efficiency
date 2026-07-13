# /plan

Enter read-only plan mode (see `AGENTS.md` "Plan mode"). No file edits, no commands that
mutate state — inspection and reasoning only.

Goal to plan for: $ARGUMENTS

Produce a plan comprehensive enough to execute from, but no more verbose than it needs to
be. Ask the user clarifying questions or surface tradeoffs rather than assuming intent on
anything that would be expensive to redo. End by presenting the plan and waiting for
explicit approval before any Stage 4/5 work begins.
