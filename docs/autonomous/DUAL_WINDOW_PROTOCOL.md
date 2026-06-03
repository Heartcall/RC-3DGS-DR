# Dual-Window Long-Horizon Protocol

## Protocol Goal

This protocol lets Codex continue the long-horizon RC-RefGS -> 3DGS-DR migration in bounded, auditable windows. Each window must recover state from repository artifacts, choose exactly one safe task, claim it, execute it, verify it, log it, and end with a GO / CONDITIONAL GO / SWITCH MODEL / NO-GO decision.

This is not a background automation system. Work happens only when the user explicitly asks Codex to continue the protocol.

Long-term migration source documents:

- `docs/rc_refgs_to_3dgs_dr_modification_prompt.md`
- `docs/rc_refgs_optimization_modules_reuse_guide.md`

## Single-Window Execution Flow

Every window must follow this sequence:

1. Recover repository state.
2. Select one highest-value safe task.
3. Claim exactly one task.
4. Execute only the claimed task.
5. Verify with the task's minimum checks.
6. Log the window and update state.
7. Decide the next action and model.

The window must not rely on chat context as the source of truth. Git status, the plan, state, autonomous log, and coordination board are authoritative.

## Model Roles

Only these two model windows are allowed in this protocol:

- `gpt-5.5 high`
- `gpt-5.4 medium`

### `gpt-5.5 high`

Use this model for:

- architecture design;
- CUDA rasterizer alpha/depth exposure design;
- RC loss mathematics and gradient-boundary review;
- `train.py` schedule design;
- complex bug diagnosis;
- cross-file refactors;
- risk decisions;
- GO / NO-GO decisions for phase transitions;
- review of `gpt-5.4 medium` implementation.

### `gpt-5.4 medium`

Use this model for:

- documentation maintenance;
- small Python edits;
- argument registration;
- logging / CLI / help text;
- `py_compile` / `git diff --check`;
- Markdown table updates;
- ablation runner dry-run command generation;
- simple metrics skeletons;
- state synchronization, autonomous log cleanup, and coordination-board updates.

## Decision Definitions

- `GO`: the current model can continue with the next safe task.
- `CONDITIONAL GO`: the current model can continue only after a stated condition is satisfied, such as user confirmation to modify a CUDA extension.
- `SWITCH MODEL`: the next task should be handled by the other allowed model. The decision must include one of:
  - `SWITCH MODEL -> gpt-5.5 high`
  - `SWITCH MODEL -> gpt-5.4 medium`
- `NO-GO`: a high-risk or blocking condition requires stopping and waiting for user direction.

Every window must state `next recommended model`.

## SWITCH MODEL Rules

### Must Switch To `gpt-5.5 high`

Trigger `SWITCH MODEL -> gpt-5.5 high` when:

- CUDA rasterizer / C++ extension / autograd binding changes are needed;
- depth or alpha convention is unclear;
- training behavior or loss gradient paths are changed;
- multiple modules may need to be merged or restructured;
- verification fails and the cause is not obvious;
- the plan or long-term route needs to be rewritten.

### Must Switch To `gpt-5.4 medium`

Trigger `SWITCH MODEL -> gpt-5.4 medium` when:

- only docs, logs, CLI help, `py_compile`, dry-run runner, Markdown checklist, or other mechanical tasks remain;
- `gpt-5.5 high` has already made the architecture decision and a low-risk implementation remains;
- autonomous logs, coordination board, or verification matrix need bulk cleanup.

## Safety Boundaries

These boundaries are always active:

1. Default 3DGS-DR behavior must remain unchanged.
2. New training features must be default-off.
3. Full RC loss must not be enabled before reliable alpha/depth exposure exists.
4. Fake depth, random depth, all-ones depth, or other misleading depth proxies must not create training signal.
5. Long training runs are forbidden unless explicitly approved by the user.
6. Large dataset downloads are forbidden unless explicitly approved by the user.
7. Large CUDA dependency installs or rebuilds are forbidden unless the claimed task requires them and the plan states that requirement.
8. PSNR, SSIM, LPIPS, or geometry improvements must not be claimed without ablation evidence.
9. Each window must recover from git, `PLAN.md`, `STATE.md`, `AUTONOMOUS_LOG.md`, and `COORDINATION_BOARD.md`; chat memory is not authoritative.

## Prohibited Actions

- Do not execute more than one claimed task in a window.
- Do not make opportunistic edits outside the claim.
- Do not skip coordination-board claim.
- Do not skip verification.
- Do not keep a task `CLAIMED` after the window ends.
- Do not start training, evaluation sweeps, CUDA rebuilds, or dependency installs unless the claimed task and user approval allow them.
- Do not state that autonomous execution will continue in the background.

## Required Interpretation Of The Continue Command

User command:

```text
Continue RC-RefGS autonomous execution under the Dual-Window Long-Horizon Protocol. Recover state from git, the plan, the autonomous log, and the coordination board. Choose the next highest-value safe task for this model window, claim it in the coordination board, execute it, verify it, log it, and make a GO / CONDITIONAL GO / SWITCH MODEL / NO-GO decision.
```

Protocol interpretation:

- This command authorizes one finite model window only.
- Codex must recover state before selecting work.
- Codex must not skip claim.
- Codex must not skip verification.
- Codex must update `AUTONOMOUS_LOG.md`, `STATE.md`, and `COORDINATION_BOARD.md`.
- Codex must explicitly recommend the next model.
- If the highest-value task is not suitable for the current model, Codex must output `SWITCH MODEL` rather than forcing the task.

## Fixed Execution Template

### 1. Recover

- Read `git status`.
- Read `docs/autonomous/PLAN.md`.
- Read `docs/autonomous/STATE.md`.
- Read `docs/autonomous/AUTONOMOUS_LOG.md`.
- Read `docs/autonomous/COORDINATION_BOARD.md`.
- Read `docs/rc_refgs_to_3dgs_dr_modification_prompt.md`.

### 2. Select

- Choose the highest-value, low-risk task from `PLAN.md` that matches the current model.
- Respect dependencies and blockers.
- If the current model is not appropriate, output `SWITCH MODEL`.

### 3. Claim

Before editing, update `COORDINATION_BOARD.md` with:

- `task_id`;
- `owner_model`;
- `timestamp`;
- `files_to_touch`;
- `risk_level`.

Only one task may be claimed at a time.

### 4. Execute

- Execute only the claimed task.
- Do not make unclaimed drive-by changes.
- Preserve original 3DGS-DR default behavior.

### 5. Verify

- Run the task's minimum verification.
- At minimum, run `git diff --check`.
- Python changes require `python -m py_compile` on affected files.
- Record skipped verification with a reason.

### 6. Log

- Append a new entry to `AUTONOMOUS_LOG.md`.
- Update `STATE.md`.
- Update `COORDINATION_BOARD.md` to `DONE`, `BLOCKED`, or `DEFERRED`.

### 7. Decide

- Output one decision: `GO`, `CONDITIONAL GO`, `SWITCH MODEL`, or `NO-GO`.
- State `next recommended model`.
- State the rationale.

## Fixed Output Format

Each execution window should end with:

```markdown
## Autonomous window complete

- Task:
  - task_id:
  - status:

- Verification:
  - command: PASS/FAIL/SKIPPED

- State:
  - current_phase:
  - active_task:
  - next_candidate_tasks:
  - recommended next model:

- Decision:
  - decision:
  - reason:
```

If the user explicitly requests a different completion format for the current window, follow the user format while preserving the same facts.
