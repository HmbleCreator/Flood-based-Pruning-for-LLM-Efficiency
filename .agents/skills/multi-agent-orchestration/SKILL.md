---
name: multi-agent-orchestration
description: Design multi-agent systems - when to add a sub-agent role vs. keep a single agent, actor-critic/critique patterns, avoiding silent handoff failures, and routing between specialized agents. Use when building or debugging an agentic architecture with more than one agent role.
category: llm-tools
---

# Multi-Agent Orchestration

## Justify every added role
Each additional agent role should have a stated failure mode it exists to prevent - a
critique agent catches methodological errors before they're expensive to undo, a reviewer
agent catches fabricated numbers before publication, a router agent prevents a generalist
model from doing badly at a task a specialist handles better. An agent role added "for
robustness" with no specific failure mode named is complexity for its own sake - see the
simplicity prime directive in the top-level `AGENTS.md`.

## Actor-critic separation
When one agent produces an artifact and another reviews it, keep them genuinely
independent:
- The critic should not see the producer's justification/reasoning for why the result is
  correct - only the artifact and the evidence. Seeing the reasoning anchors the critic
  into accepting flawed logic (this is why `agents/reviewer.md` in this package is
  explicitly a *blind* review).
- The critic should be read-only. If it can silently "fix" what it finds, you lose the
  audit trail and the producer never learns what it got wrong.
- Don't let the same context window that produced the artifact also grade it, if you can
  avoid it - a genuinely fresh pass catches more than the producer re-reading its own
  output, which tends to confirm what it already believes it did right.

## Handoffs are where these systems actually break
The most common bug class in multi-agent systems isn't a bad individual agent - it's a
silent failure at the handoff between agents (an agent receiving malformed/incomplete
input from the previous stage and proceeding anyway instead of flagging it). Make handoffs
inspectable: log what each agent received and produced, and have the receiving agent
validate its input has what it needs before proceeding, rather than assuming.

## Routing between specialists
If routing to a specialist agent based on task type, make the routing decision explicit
and loggable (which agent, why) rather than implicit - this is the single easiest thing to
debug wrong and the hardest to debug silently. A router that's wrong 10% of the time
without ever surfacing that it routed is worse than a single generalist agent.

## Parallel vs. sequential
Genuinely independent sub-tasks (e.g. N literature-review facets, as in this package's
Stage 2) parallelize well - each has no dependency on another's output. Tasks with real
dependencies (methodology depends on literature findings) should stay sequential; forcing
parallelism onto a dependent pipeline just means agents working from stale or missing
context.
