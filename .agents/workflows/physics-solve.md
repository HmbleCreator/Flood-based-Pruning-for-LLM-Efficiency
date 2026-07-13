# /physics-solve

Run the standalone 7-stage computational physics pipeline from `agents/physics.md` on the
problem below - use this instead of the full `/research` loop when the task is a
self-contained physics/engineering/quantum computation rather than a full literature-to-
paper research project (e.g. "simulate this N-body system," "find the governing equation
from this trajectory data," "check whether this proposed PDE is dimensionally consistent").

Problem: $ARGUMENTS

Steps:
1. Adopt the `agents/physics.md` persona.
2. Stage 1 (Inspect & parse): identify the domain, validate units, note known limiting
   cases to check against later.
3. Stages 2-4 (Literature & context, Methodology selection, Compute): follow the
   methodology and skill routing tables in `agents/physics.md`.
4. Stage 5 (Validate) - mandatory, all applicable checks from the checklist.
5. Stage 6 (Visualize & interpret).
6. Before Stage 7 (Report): spawn `agents/physics-critique.md` with only the artifacts
   (numerical outputs, plots, fitted parameters, script manifest) - not your reasoning.
   Iterate on MINOR_FIXES, restart on CRITICALLY_FLAWED, max 2 cycles.
7. Stage 7 (Report): Problem -> Method -> Results -> Validation -> Interpretation ->
   Limitations, every number traceable to a logged run.

If this problem is actually cross-domain (a physics component feeding into a learned
model), do the physics-specific stages here, then hand off the validated physics
artifacts to `/research` or directly to `agents/ml-engineer.md` for the ML side - don't
try to do both under one undifferentiated pass.
