# Verification Matrix

Use the smallest verification set that proves the claimed task. Always run `git diff --check` before marking a task DONE.

| Change type | Required commands | Notes |
| --- | --- | --- |
| docs-only | `git diff --check`; Markdown trailing whitespace scan | No training, imports, or CUDA work required. |
| autonomous protocol docs | `python scripts/autonomous_check.py`; `git diff --check` | Required for `docs/autonomous/` changes. |
| Python syntax | `python -m py_compile affected_files`; `git diff --check` | Use exact affected files, not broad unrelated modules. |
| CLI params | `python train.py --help`; `python eval.py --help`; `git diff --check` | Run only after argument/parser edits. |
| runner dry-run | `python scripts/run_rc_3dgs_dr_ablation.py --help`; `python scripts/run_rc_3dgs_dr_ablation.py ... --dry_run`; `git diff --check` | Runner must default to dry-run and skip long training. |
| CUDA extension changes | `pip install -e submodules/diff-gaussian-rasterization_c7`; import smoke test; `git diff --check` | Only allowed after explicit high-model GO and claimed CUDA task. |
| training smoke | short `python train.py ...` command with existing dataset; `git diff --check` | Only if user explicitly allows and dataset exists. |
| metrics skeleton | `python -m py_compile metrics/reflection_consistency_eval.py`; `python metrics/reflection_consistency_eval.py --help`; `git diff --check` | Metrics must report missing buffers clearly. |

## Markdown Trailing Whitespace Scan

Use:

```bash
if grep -RIn '[[:blank:]]$' docs/autonomous; then exit 1; else echo 'no trailing whitespace'; fi
```

## CUDA Gate

CUDA extension verification is not permitted in routine medium-model windows. If a task requires edits under `submodules/diff-gaussian-rasterization_c7`, the window must end with `SWITCH MODEL -> gpt-5.5 high` unless it is already running under `gpt-5.5 high` with explicit task scope.
