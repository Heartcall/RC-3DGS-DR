# Coordination Board

Only one task may be claimed at a time. If a future window finds a `CLAIMED` or `IN_PROGRESS` task, it must recover or release that task before claiming another one.

| task_id | status | owner_model | claimed_at_jst | files_to_touch | risk_level | last_update | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RC-AUTO-000 | DONE | gpt-5.4 medium | 2026-06-04 04:11 JST | `docs/autonomous/DUAL_WINDOW_PROTOCOL.md`; `docs/autonomous/PLAN.md`; `docs/autonomous/STATE.md`; `docs/autonomous/AUTONOMOUS_LOG.md`; `docs/autonomous/COORDINATION_BOARD.md`; `docs/autonomous/VERIFICATION_MATRIX.md`; `scripts/autonomous_check.py` | low | 2026-06-04 04:11 JST | Bootstrap completed and verified. |
| RC-AUTO-010 | DONE | gpt-5.5 high | 2026-06-04 04:54 JST | `docs/autonomous/renderer_buffer_review.md` | medium | 2026-06-04 04:54 JST | Renderer buffer review completed and verified. |
| RC-AUTO-020 | TODO | unclaimed | - | `gaussian_renderer/__init__.py` | medium | 2026-06-04 04:11 JST | Depends on RC-AUTO-010. |
| RC-AUTO-030 | TODO | unclaimed | - | `docs/autonomous/alpha_depth_strategy.md` | high | 2026-06-04 04:11 JST | Requires gpt-5.5 high. |
| RC-AUTO-040 | TODO | unclaimed | - | `utils/graphics_utils.py` | medium | 2026-06-04 04:11 JST | Depends on RC-AUTO-030. |
| RC-AUTO-050 | TODO | unclaimed | - | `utils/rc_reflection_consistency.py` | medium | 2026-06-04 04:11 JST | Depends on RC-AUTO-020 and RC-AUTO-040. |
| RC-AUTO-060 | TODO | unclaimed | - | `arguments/__init__.py` | low | 2026-06-04 04:11 JST | Default-off args only. |
| RC-AUTO-070 | TODO | unclaimed | - | `train.py` | high | 2026-06-04 04:11 JST | Requires gpt-5.5 high because it changes training schedule. |
| RC-AUTO-080 | TODO | unclaimed | - | `metrics/reflection_consistency_eval.py` | medium | 2026-06-04 04:11 JST | Metrics skeleton only. |
| RC-AUTO-090 | TODO | unclaimed | - | `scripts/run_rc_3dgs_dr_ablation.py` | medium | 2026-06-04 04:11 JST | Dry-run default required. |
| RC-AUTO-100 | TODO | unclaimed | - | `docs/autonomous/readiness_review.md` | medium | 2026-06-04 04:11 JST | Final readiness review after implementation tasks. |
