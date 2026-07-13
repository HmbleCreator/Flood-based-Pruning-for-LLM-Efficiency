# /learn

Review this entire conversation from the start and distill it into a reusable skill.

Analyze:
1. Workflow pattern - what tools were used, in what order, what was the overall approach.
2. Failure modes - what broke, what errors occurred, how they were recovered from.
3. User corrections - where the user steered the approach differently than your first
   attempt; this captures judgment that isn't obvious from the task description alone.
4. Key parameters - specific configs, thresholds, paths, or settings that mattered.
5. Reproducibility - what someone would need to know to repeat this on a similar problem.

Then write a new file at `skills/<kebab-case-name>/SKILL.md`:

```
---
name: <kebab-case-name>
description: <one line: what it does AND when to use it>
source: learned-from-session
---

# <name>

## Overview
## Workflow pattern
## Failure modes & recovery
## User corrections
## Key parameters
## When to use
```

The "Failure modes & recovery" section is usually the most valuable part - it's the thing
a fresh session would otherwise rediscover the hard way.
