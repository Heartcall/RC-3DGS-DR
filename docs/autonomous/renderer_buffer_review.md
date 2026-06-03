# RC-AUTO-010 Renderer Buffer Contract Review

## Scope

This review documents the current 3DGS-DR renderer buffer contract for RC-RefGS migration. It is a docs-only task. No renderer, training, CUDA, or evaluation code is changed in this window.

Primary sources:

- `docs/rc_refgs_to_3dgs_dr_modification_prompt.md`
- `docs/rc_refgs_optimization_modules_reuse_guide.md`
- `gaussian_renderer/__init__.py`
- `train.py`
- `arguments/__init__.py`
- `scene/cameras.py`
- `utils/general_utils.py`
- `submodules/diff-gaussian-rasterization_c7`

## Current Python Renderer Contract

`gaussian_renderer.render()` currently has this signature:

```python
def render(
    viewpoint_camera,
    pc,
    pipe,
    bg_color,
    scaling_modifier=1.0,
    initial_stage=False,
    more_debug_infos=False,
):
```

### `initial_stage=True`

The initial stage uses `diff_gaussian_rasterization_c3` and returns:

```python
{
    "render": base_color,
    "viewspace_points": screenspace_points,
    "visibility_filter": _radii > 0,
    "radii": _radii,
}
```

Observed behavior:

- This stage has no deferred reflection buffers.
- This stage is active until `iteration > opt.init_until_iter` in `train.py`.
- RC loss must never run in this stage.

### `initial_stage=False`

The deferred reflection stage uses `diff_gaussian_rasterization_c7`.

Inputs to c7:

```python
normals = pc.get_min_axis(viewpoint_camera.camera_center)
refl_ratio = pc.get_refl
input_ts = torch.cat([torch.zeros_like(normals), normals, refl_ratio], dim=-1)
bg_map = torch.cat([bg_map_const, torch.zeros(4, imH, imW, device="cuda")], dim=0)
out_ts, _radii = rasterizer_c7(..., colors_precomp=input_ts, bg_map=bg_map)
```

The 7 output channels are parsed as:

```python
base_color = out_ts[:3]
normal_map = normalize(out_ts[3:6])
refl_strength = out_ts[6:7]
refl_color = get_refl_color(pc.get_envmap, viewpoint_camera.HWK, viewpoint_camera.R, viewpoint_camera.T, normal_map)
final_image = (1 - refl_strength) * base_color + refl_strength * refl_color
```

Returned buffers:

```python
{
    "render": final_image,
    "refl_strength_map": refl_strength,
    "normal_map": normal_map.permute(2, 0, 1),
    "refl_color_map": refl_color,
    "base_color_map": base_color,
    "viewspace_points": screenspace_points,
    "visibility_filter": _radii > 0,
    "radii": _radii,
}
```

## RC Buffer Mapping

The safest RC buffer aliases for 3DGS-DR are:

| RC field | Current 3DGS-DR source | Status | Notes |
| --- | --- | --- | --- |
| `specular_rgb` | `refl_color_map` | available as alias | Preferred first alias because it is the deferred environment reflection color. |
| `specular_rgb_weighted` | `refl_color_map * refl_strength_map` | computable alias | Useful ablation; do not make it the only mode before testing. |
| `specular_confidence` | `refl_strength_map` | available as alias | Confidence direction is high strength -> high confidence. Do not use `(1 - refl_strength)`. |
| `normal_render` | `normal_map` | available as alias | Current output is CHW after world-space reflection normal normalization. |
| `base_rgb` | `base_color_map` | available | Existing deferred base branch output. |
| `final_rgb` | `render` | available | Current training/eval image. |
| `alpha_map` | c7 final transmittance `T` -> `1 - T` | missing from Python | Requires c7 binding/output change. |
| `depth_map` | c7 expected depth or depth channel normalized by alpha | missing from Python | Requires design and likely c7 binding/output change. |
| `normal_depth` | `depth_to_normal(depth_map)` | blocked | Requires reliable depth first. |

## Missing Buffers And Risk

### `alpha_map`

Current c7 CUDA path computes final transmittance:

- `ImageState::fromChunk()` allocates `img.accum_alpha`.
- `FORWARD::render()` receives `final_T`.
- The CUDA kernel writes `final_T[pix_id] = T`.

Current Python binding discards it:

- `_C.rasterize_gaussians()` returns `num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer`.
- `_RasterizeGaussians.forward()` returns only `color, radii`.

Conclusion: `alpha_map = 1 - final_T` is a plausible convention, but exposing it requires a CUDA/C++/autograd binding change. That is a `SWITCH MODEL -> gpt-5.5 high` task under the protocol.

### `depth_map`

Current c7 path has per-Gaussian depths:

- preprocessing writes `geomState.depths`;
- sorting uses depth in `duplicateWithKeys`;
- the render kernel does not output per-pixel expected depth.

Conclusion: per-pixel `depth_map` is not currently available. It must not be faked with all-ones, random, or nearest-center depth for training signal.

### `normal_depth`

`normal_depth` depends on a reliable `depth_map`. Until depth is exposed and convention-checked, any full RC loss using normal agreement is blocked.

## Current Training Interaction

Current `train.py` behavior:

- Calls `render(..., initial_stage=initial_stage)` once per iteration.
- Computes L1 + DSSIM reconstruction loss.
- Optionally adds `use_env_scope` reflection mask loss when `refl_strength_map` exists.
- Calls `loss.backward()` immediately after those losses.

Current `arguments/__init__.py` contains `lambda_refl_smooth = 0.4`, but the main training loop does not use it. Do not enable this value by default as part of RC migration.

## Safe Minimal Implementation Path

### Step 1: Non-invasive aliases only

Add optional renderer parameters:

```python
return_rc_buffers=False
rc_specular_mode="color"
```

When `return_rc_buffers=True` and `initial_stage=False`, add aliases only:

```python
results["specular_rgb"] = refl_color
results["specular_rgb_weighted"] = refl_color * refl_strength
results["specular_confidence"] = refl_strength
results["normal_render"] = results["normal_map"]
```

When `initial_stage=True`, do not compute RC aliases from fake values. Either omit them or return them as `None` only under `return_rc_buffers=True`.

Default behavior must be unchanged:

- Existing calls without `return_rc_buffers` must return the same keys and values.
- `eval.py`, `train.py`, and `net_viewer.py` must keep working without edits.

Recommended task: RC-AUTO-020, `gpt-5.4 medium`, because it is a small Python aliasing change if no training integration is included.

### Step 2: Alpha/depth strategy before implementation

Do not implement RC loss before alpha/depth strategy is written. The strategy must decide:

- whether c7 should return `alpha_map = 1 - final_T`;
- whether depth should be a true expected-depth output in CUDA render;
- whether a temporary depth channel is acceptable only after alpha exposure;
- how autograd binding signatures should handle extra outputs;
- what smoke test proves shapes and conventions.

Recommended task: RC-AUTO-030, `gpt-5.5 high`, because it involves CUDA binding and convention risk.

### Step 3: Full RC loss remains blocked

Full RC loss needs:

- `specular_rgb`;
- `specular_confidence`;
- `alpha_map`;
- `depth_map`;
- `normal_render`;
- optional `normal_depth`.

Until alpha/depth are reliable, RC loss skeletons must return zero loss with stats such as `valid_count=0` instead of producing misleading training signal.

## GO / NO-GO Assessment

- RC-AUTO-020: GO for a default-off, non-invasive alias implementation after this review.
- RC-AUTO-030: SWITCH MODEL -> gpt-5.5 high is required before alpha/depth implementation because CUDA binding and depth convention decisions are high risk.
- Full RC loss training integration: NO-GO until alpha/depth exposure is designed and implemented.

## Completion Criteria Check

- documents current render outputs: yes.
- identifies missing alpha/depth/normal_depth: yes.
- recommends safe minimal implementation path: yes.
