# RC-RefGS -> 3DGS-DR Autonomous Migration Plan

This plan is the source of truth for Dual-Window Long-Horizon Protocol task selection. Tasks must be claimed in `docs/autonomous/COORDINATION_BOARD.md` before execution and logged in `docs/autonomous/AUTONOMOUS_LOG.md` after verification.

Long-term source documents:

- `docs/rc_refgs_to_3dgs_dr_modification_prompt.md`
- `docs/rc_refgs_optimization_modules_reuse_guide.md`

Global constraints:

- Preserve original 3DGS-DR behavior by default.
- Keep all new training behavior default-off.
- Do not enable full RC loss before reliable alpha/depth buffers exist.
- Do not use fake depth, random depth, or all-ones depth as training signal.
- Do not run long training, download large datasets, or rebuild CUDA extensions without an explicit task and user approval.

## Phase 0: Protocol Bootstrap

### RC-AUTO-000

- task_id: RC-AUTO-000
- title: Bootstrap autonomous protocol files
- phase: Phase 0
- status: DONE
- recommended_model: gpt-5.4 medium
- risk_level: low
- dependencies: []
- files_expected:
  - `docs/autonomous/DUAL_WINDOW_PROTOCOL.md`
  - `docs/autonomous/PLAN.md`
  - `docs/autonomous/STATE.md`
  - `docs/autonomous/AUTONOMOUS_LOG.md`
  - `docs/autonomous/COORDINATION_BOARD.md`
  - `docs/autonomous/VERIFICATION_MATRIX.md`
  - `scripts/autonomous_check.py`
- verification:
  - `python scripts/autonomous_check.py`
  - `git diff --check`
- completion_criteria:
  - all protocol files exist
  - autonomous_check passes
  - initial log entry exists
  - coordination board has RC-AUTO-000 marked DONE
- rollback_notes:
  - remove `docs/autonomous` and `scripts/autonomous_check.py`

## Phase 1: Renderer Buffer Contract

### RC-AUTO-010

- task_id: RC-AUTO-010
- title: Review 3DGS-DR renderer buffer contract for RC fields
- phase: Phase 1
- status: DONE
- recommended_model: gpt-5.5 high
- risk_level: medium
- dependencies:
  - RC-AUTO-000
- files_expected:
  - `docs/autonomous/renderer_buffer_review.md`
- verification:
  - `git diff --check`
- completion_criteria:
  - documents current render outputs
  - identifies missing alpha/depth/normal_depth
  - recommends safe minimal implementation path
- rollback_notes:
  - remove `docs/autonomous/renderer_buffer_review.md`

### RC-AUTO-020

- task_id: RC-AUTO-020
- title: Add non-invasive RC buffer aliases without changing default behavior
- phase: Phase 1
- status: TODO
- recommended_model: gpt-5.4 medium
- risk_level: medium
- dependencies:
  - RC-AUTO-010
- files_expected:
  - `gaussian_renderer/__init__.py`
- verification:
  - `python -m py_compile gaussian_renderer/__init__.py`
  - `git diff --check`
- completion_criteria:
  - render can optionally expose specular_rgb/specular_confidence/normal_render
  - default call behavior unchanged
  - initial_stage safe behavior documented
- rollback_notes:
  - revert changes to `gaussian_renderer/__init__.py`

## Phase 2: Alpha/Depth Exposure Design

### RC-AUTO-030

- task_id: RC-AUTO-030
- title: Decide alpha/depth exposure strategy for c7 rasterizer
- phase: Phase 2
- status: TODO
- recommended_model: gpt-5.5 high
- risk_level: high
- dependencies:
  - RC-AUTO-010
- files_expected:
  - `docs/autonomous/alpha_depth_strategy.md`
- verification:
  - `git diff --check`
- completion_criteria:
  - identifies exact CUDA binding files to modify
  - defines alpha/depth convention
  - gives GO/NO-GO for CUDA implementation
- rollback_notes:
  - remove `docs/autonomous/alpha_depth_strategy.md`

## Phase 3: Geometry Helpers

### RC-AUTO-040

- task_id: RC-AUTO-040
- title: Implement geometry helper skeleton with documented conventions
- phase: Phase 3
- status: TODO
- recommended_model: gpt-5.5 high
- risk_level: medium
- dependencies:
  - RC-AUTO-030
- files_expected:
  - `utils/graphics_utils.py`
- verification:
  - `python -m py_compile utils/graphics_utils.py`
  - `git diff --check`
- completion_criteria:
  - helper signatures exist
  - coordinate conventions documented
  - no training behavior changed
- rollback_notes:
  - revert changes to `utils/graphics_utils.py`

## Phase 4: RC Loss Skeleton

### RC-AUTO-050

- task_id: RC-AUTO-050
- title: Add RC loss skeleton with blocked-safe behavior
- phase: Phase 4
- status: TODO
- recommended_model: gpt-5.5 high
- risk_level: medium
- dependencies:
  - RC-AUTO-020
  - RC-AUTO-040
- files_expected:
  - `utils/rc_reflection_consistency.py`
- verification:
  - `python -m py_compile utils/rc_reflection_consistency.py`
  - `git diff --check`
- completion_criteria:
  - missing depth/alpha returns zero loss with stats
  - no fake depth or fake alpha
  - source spec detach behavior documented
- rollback_notes:
  - remove `utils/rc_reflection_consistency.py`

## Phase 5: Training Integration

### RC-AUTO-060

- task_id: RC-AUTO-060
- title: Register RC training arguments default-off
- phase: Phase 5
- status: TODO
- recommended_model: gpt-5.4 medium
- risk_level: low
- dependencies:
  - RC-AUTO-050
- files_expected:
  - `arguments/__init__.py`
- verification:
  - `python -m py_compile arguments/__init__.py`
  - `python train.py --help`
  - `git diff --check`
- completion_criteria:
  - new RC args exist
  - defaults preserve original behavior
- rollback_notes:
  - revert changes to `arguments/__init__.py`

### RC-AUTO-070

- task_id: RC-AUTO-070
- title: Integrate scheduled RC loss default-off
- phase: Phase 5
- status: TODO
- recommended_model: gpt-5.5 high
- risk_level: high
- dependencies:
  - RC-AUTO-050
  - RC-AUTO-060
- files_expected:
  - `train.py`
- verification:
  - `python -m py_compile train.py`
  - `git diff --check`
- completion_criteria:
  - initial_stage never triggers RC
  - missing alpha/depth skips RC safely
  - logs valid_count and loss when enabled
- rollback_notes:
  - revert changes to `train.py`

## Phase 6: Metrics

### RC-AUTO-080

- task_id: RC-AUTO-080
- title: Add reflection consistency metric skeleton
- phase: Phase 6
- status: TODO
- recommended_model: gpt-5.4 medium
- risk_level: medium
- dependencies:
  - RC-AUTO-050
- files_expected:
  - `metrics/reflection_consistency_eval.py`
- verification:
  - `python -m py_compile metrics/reflection_consistency_eval.py`
  - `git diff --check`
- completion_criteria:
  - metric script imports
  - reports blocked/missing buffers clearly
- rollback_notes:
  - remove `metrics/reflection_consistency_eval.py`

## Phase 7: Ablation Runner

### RC-AUTO-090

- task_id: RC-AUTO-090
- title: Add dry-run ablation runner
- phase: Phase 7
- status: TODO
- recommended_model: gpt-5.4 medium
- risk_level: medium
- dependencies:
  - RC-AUTO-060
- files_expected:
  - `scripts/run_rc_3dgs_dr_ablation.py`
- verification:
  - `python -m py_compile scripts/run_rc_3dgs_dr_ablation.py`
  - `python scripts/run_rc_3dgs_dr_ablation.py --help`
  - `git diff --check`
- completion_criteria:
  - dry_run default
  - supports base/rc/wo_ref/wo_conf/refl_smooth_only
- rollback_notes:
  - remove `scripts/run_rc_3dgs_dr_ablation.py`

## Phase 8: Documentation And Readiness Review

### RC-AUTO-100

- task_id: RC-AUTO-100
- title: Final readiness review
- phase: Phase 8
- status: TODO
- recommended_model: gpt-5.5 high
- risk_level: medium
- dependencies:
  - RC-AUTO-020
  - RC-AUTO-030
  - RC-AUTO-050
  - RC-AUTO-060
  - RC-AUTO-070
  - RC-AUTO-080
  - RC-AUTO-090
- files_expected:
  - `docs/autonomous/readiness_review.md`
- verification:
  - `python scripts/autonomous_check.py`
  - `git diff --check`
- completion_criteria:
  - documents what is implemented
  - documents blockers
  - gives final GO/CONDITIONAL GO/NO-GO
- rollback_notes:
  - remove `docs/autonomous/readiness_review.md`
