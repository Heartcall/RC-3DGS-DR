# Prompt: 将 RC-RefGS 的跨视角反射一致性优化迁移到 3DGS-DR

> 这是一份可交给另一个 Codex/开发者执行的二级 prompt。执行者需要在 3DGS-DR 仓库中完成改造，但不要机械复制 RC-RefGS 的 roughness/PBR 结构；应优先复用 3DGS-DR 已有的 deferred reflection buffers。

## 0. 先读代码事实，再做改造

你要先阅读并交叉核对这些文件：

- `README.md`
- `train.py`
- `arguments/__init__.py`
- `gaussian_renderer/__init__.py`
- `scene/gaussian_model.py`
- `utils/loss_utils.py`
- `utils/general_utils.py`
- `utils/graphics_utils.py`
- `eval.py`
- `train.sh`
- `eval.sh`
- `scene/cameras.py`
- `scene/dataset_readers.py`
- `utils/image_utils.py`
- `net_viewer.py`
- `submodules/diff-gaussian-rasterization_c7`
- `submodules/diff-gaussian-rasterization_c3`
- `docs/rc_refgs_optimization_modules_reuse_guide.md`

不要先写代码。先确认下面的 3DGS-DR 事实仍成立；如果代码已变化，以当前代码为准并更新改造方案。

## 1. 3DGS-DR 当前结构摘要

### 1.1 Renderer 当前返回的 buffers

当前 `gaussian_renderer/__init__.py` 中 `render()` 有两个阶段：

- `initial_stage=True`：调用 `diff_gaussian_rasterization_c3`，只返回普通 3DGS RGB 和训练需要的可见性信息：
  - `render`
  - `viewspace_points`
  - `visibility_filter`
  - `radii`
- `initial_stage=False`：调用 `diff_gaussian_rasterization_c7`，把 `[base_color, normal, refl_strength]` 作为 7 通道 feature rasterize，再做 deferred reflection 合成，返回：
  - `render`
  - `refl_strength_map`
  - `normal_map`
  - `refl_color_map`
  - `base_color_map`
  - `viewspace_points`
  - `visibility_filter`
  - `radii`

当前 renderer 没有返回 `alpha_map`、`depth_map`、`normal_depth` 或 `surf_depth`。这些是 RC loss 的阻塞项，必须先补齐或提供可靠 fallback。

### 1.2 Deferred reflection 合成逻辑

当前 deferred reflection 路径是：

```python
normals = pc.get_min_axis(viewpoint_camera.camera_center)
refl_ratio = pc.get_refl
input_ts = torch.cat([torch.zeros_like(normals), normals, refl_ratio], dim=-1)
out_ts, radii = rasterizer_c7(..., colors_precomp=input_ts, bg_map=bg_map)

base_color = out_ts[:3]
normal_map = normalize(out_ts[3:6])
refl_strength = out_ts[6:7]
refl_color = envmap(reflect(view_ray, normal_map))
final_image = (1 - refl_strength) * base_color + refl_strength * refl_color
```

语义边界：

- `base_color_map` 是 SH/base branch rasterized color。
- `normal_map` 来自每个 Gaussian 的最小尺度轴，经过 c7 rasterize 后归一化。
- `refl_color_map` 是环境贴图对反射方向的查询结果，本身不是最终反射贡献。
- `refl_strength_map` 是 learned reflection strength，越高表示越强反射。
- 最终反射贡献可写为 `refl_strength_map * refl_color_map`。

迁移 RC-RefGS 时，`refl_color_map` 可作为 RC-RefGS 中 `spec_light/specular_rgb` 的首选替代；`refl_strength_map` 可作为 `roughness_map/specular_confidence` 的替代，但语义方向不同：

- RC-RefGS：`roughness` 越低，specular confidence 越高。
- 3DGS-DR：`refl_strength` 越高，specular confidence 越高。

因此不要把 RC-RefGS 的 `roughness_map` 机械搬到 3DGS-DR。3DGS-DR 已经有更合适的 `refl_strength_map`、`refl_color_map`、`base_color_map`、`normal_map` 和 environment map reflection 结构。

### 1.3 训练阶段

当前 `train.py` 的关键阶段和参数：

- `init_until_iter`，默认 `3000`：初始阶段，`initial_stage=True`，只训练普通 RGB 3DGS 渲染，不走 deferred reflection buffers。
- `feature_rest_from_iter`，默认 `10000`：超过该迭代后每 1000 步提升一次 SH degree。
- `normal_prop_until_iter + longer_prop_iter`，默认 `24000 + longer_prop_iter`：normal propagation 窗口。该窗口内通过 opacity reset、reflection reset、color disturbance、scale reset 等工程策略推动 reflective normal propagation。
- `longer_prop_iter`：延长 normal propagation 和总训练长度；`TOT_ITER = iterations + longer_prop_iter + 1`。
- `opac_lr0_interval`：normal propagation 阶段周期性打开/关闭 opacity learning rate；默认 `200`，设为 `0` 时不执行这一路径。
- `densification_interval_when_prop`：normal propagation 阶段 densification interval，默认 `500`。
- `use_env_scope`、`env_scope_center`、`env_scope_radius`：真实场景中限制环境光作用范围，并对 scope 外 Gaussian 的 `refl_strength` 加惩罚。
- env map optimization：`GaussianModel.training_setup()` 把 `self.env_map.parameters()` 加入 optimizer，学习率来自 `envmap_cubemap_lr`。
- `lambda_refl_smooth`：当前在 `arguments/__init__.py` 中定义，默认 `0.4`，但主训练循环没有使用它；`utils/loss_utils.py` 中存在 `smooth_img_loss()` 和 `bilateral_smooth_img_loss()`，可作为 reflection strength smoothness 的候选实现或对照基础。

当前主训练 loss 是 RGB L1 + DSSIM，加上可选的 `use_env_scope` reflection mask loss。没有 RC loss、没有显式 normal regularization loss、没有正在使用的 reflection smoothness loss。

### 1.4 Depth/alpha/normal/confidence 现状

当前已有：

- Gaussian opacity 参数：`GaussianModel.get_opacity`。
- c7 CUDA rasterizer 内部 blending transmittance：`final_T`，存入 `ImageState::accum_alpha`，但 Python binding 没有返回。
- per-Gaussian preprocessing depth：`geomState.depths`，用于 tile sorting，但 Python binding 没有返回 per-pixel depth。
- renderer normal：`normal_map`，可作为 `normal_render`。
- reflection confidence：`refl_strength_map`，可作为 `specular_confidence`。

当前缺失：

- `alpha_map`：应从 c7 rasterizer 暴露 `1 - final_T`。
- `depth_map`/`surf_depth`：应从 rasterizer 暴露 expected depth，或者临时用额外 feature channel rasterize depth numerator 并除以 alpha。
- `normal_depth`：应由 `depth_map` 反投影后的 finite difference/cross product 计算；如果没有 depth，不能声称已实现完整 RC normal agreement。

推荐优先级：

1. 首选：扩展 c7 rasterizer 返回 `alpha_map` 和 expected `depth_map`。
2. 可接受短期 fallback：把 camera-space depth 作为第 8 个 feature channel 进入 c7，返回 `depth_accum = sum(depth_i * alpha_i * T_i)`，再用 `depth_map = depth_accum / alpha_map.clamp_min(eps)`；同时从 c7 暴露 `alpha_map = 1 - final_T`。
3. 不推荐：用 Gaussian 中心最近点或 `radii > 0` 近似 depth/alpha 直接训练 RC。它只能用于 dry-run/static smoke，不应进入正式 ablation 结论。

## 2. 改造目标

目标不是替换 3DGS-DR 的 deferred reflection，而是在现有 deferred reflection 基础上增加：

- renderer intermediate buffers 标准化；
- reflection consistency loss；
- scheduled training integration；
- optional reflection strength smoothness，或使用现有 `lambda_refl_smooth` 做对照；
- reflection consistency / reflective-region evaluation metrics；
- base/rc/wo_ref/wo_conf/refl_smooth_only ablation runner。

核心设计：

- `specular_rgb` 默认使用 `refl_color_map`。增加 CLI `--rc_specular_mode {color,weighted}`，其中 `weighted` 使用 `refl_color_map * refl_strength_map`。
- `specular_confidence` 使用 `refl_strength_map`，权重方向为 `refl_strength ** gamma`。
- `normal_render` 使用 `normal_map`。
- `normal_depth` 使用 `depth_to_normal(depth_map)`；若 depth 不可用，则 RC loss 必须跳过 normal agreement 或返回 zero，并记录 `valid_count=0`。
- RC loss 只在 `initial_stage=False` 且 geometry/alpha 相对稳定后启用，默认关闭。

## 3. 文件级改造计划

### 3.1 `gaussian_renderer/__init__.py`

修改目标：让 `render()` 可选返回 RC 所需 buffers，并保持现有调用兼容。

新增参数建议：

```python
def render(
    viewpoint_camera,
    pc,
    pipe,
    bg_color,
    scaling_modifier=1.0,
    initial_stage=False,
    more_debug_infos=False,
    return_rc_buffers=False,
    rc_specular_mode="color",
):
    ...
```

非 initial stage 返回 dict 应扩展为：

```python
{
    "render": final_image,
    "base_color_map": base_color,
    "refl_color_map": refl_color,
    "refl_strength_map": refl_strength,
    "normal_map": normal_map_chw,
    "alpha_map": alpha_or_opacity,        # 新增，若可获得
    "depth_map": depth_or_expected_depth, # 新增，若可获得
    "specular_rgb": refl_color if rc_specular_mode == "color" else refl_color * refl_strength,
    "specular_confidence": refl_strength,
    "normal_render": normal_map_chw,
    "normal_depth": depth_to_normal(viewpoint_camera, depth_map) if depth_map is not None else None,
    "viewspace_points": screenspace_points,
    "visibility_filter": _radii > 0,
    "radii": _radii,
}
```

注意：

- initial stage 没有 deferred reflection buffers；`return_rc_buffers=True` 时可以返回同名 key 但值为 `None`，或者在训练集成中直接禁止 initial stage 计算 RC。
- `normal_map` 当前先转为 HWC normalize，再转回 CHW；继续保持 CHW 输出，避免破坏 `eval.py` 和 `net_viewer.py`。
- `specular_rgb` 的颜色空间要和 `render`/GT 一致。当前 `refl_color_map` 来自 `torch.sigmoid(env_map(...))`，与 final image 同在 `[0,1]` 范围；不要混入 RC-RefGS 的 linear/sRGB 假设。

### 3.2 `submodules/diff-gaussian-rasterization_c7`

修改目标：暴露 alpha/depth。

必须检查这些文件：

- `submodules/diff-gaussian-rasterization_c7/rasterize_points.cu`
- `submodules/diff-gaussian-rasterization_c7/cuda_rasterizer/config.h`
- `submodules/diff-gaussian-rasterization_c7/cuda_rasterizer/forward.h`
- `submodules/diff-gaussian-rasterization_c7/cuda_rasterizer/forward.cu`
- `submodules/diff-gaussian-rasterization_c7/cuda_rasterizer/rasterizer_impl.cu`
- `submodules/diff-gaussian-rasterization_c7/cuda_rasterizer/rasterizer.h`
- `submodules/diff-gaussian-rasterization_c7/diff_gaussian_rasterization_c7/__init__.py`
- `submodules/diff-gaussian-rasterization_c7/ext.cpp`

推荐实现：

- alpha：`FORWARD::render()` 内已写 `final_T[pix_id] = T`，Python 侧需要返回 `alpha = 1 - final_T`。
- depth：在 CUDA render kernel 内新增 `depth_accum += depth[gaussian_id] * alpha * T`，再输出 `depth_accum / alpha.clamp_min(eps)`；需要把 per-Gaussian `geomState.depths` 传入 render kernel。
- binding：把 `_C.rasterize_gaussians()` 返回值从 `(num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer)` 扩展为 `(num_rendered, color, alpha, depth, radii, geomBuffer, binningBuffer, imgBuffer)`。
- autograd：如果 alpha/depth 只用于 mask/geometry correspondence，建议在 RC loss 中 detach 它们，减少 backward 改造；Python autograd Function 仍需在 backward signature 中接收新增输出的 grad placeholder。

短期可选方案：

- 把 depth 作为第 8 个 rasterized channel：`NUM_CHANNELS=8`，`input_ts = cat([zeros3, normals3, refl1, depth1])`。
- c7 output 解析 `depth_accum = out_ts[7:8]`。
- 同时仍需暴露 `alpha_map = 1 - final_T`，否则无法把 accumulated depth 归一化。
- 该方案实现快，但要验证 depth convention 和背景区域行为。

### 3.3 `utils/graphics_utils.py`

新增 camera geometry helpers：

```python
def backproject_depth(camera, depth_map):
    """Return world points [H,W,3] from camera depth [1,H,W]."""
    ...

def project_points(camera, points_world):
    """Return grid [1,H,W,2], projected_depth [1,H,W], valid [1,H,W]."""
    ...

def depth_to_normal(camera, depth_map):
    """Finite-difference normal from depth, returned as [3,H,W] in world coordinates."""
    ...
```

约束：

- 使用 `camera.HWK`、`camera.world_view_transform`、`camera.full_proj_transform` 和当前 `sample_camera_rays()` 的相机约定。
- `grid_sample` 需要 `[-1, 1]` NDC grid，后续 loss 使用 `align_corners=True`，这里不要混用 `align_corners=False` 坐标。
- normal 坐标系必须和 `normal_map` 一致；当前 `normal_map` 用 world-space camera rays 做 reflection，所以 `normal_depth` 也应输出 world-space normal。

### 3.4 `utils/rc_reflection_consistency.py`

新增文件，集中放 RC loss 和 pair selection，避免把训练逻辑塞进 `train.py`。

建议接口：

```python
from dataclasses import dataclass

@dataclass
class ReflectionConsistencyStats:
    loss: torch.Tensor
    valid_count: int
    mean_weight: float


def choose_pair_camera(cameras, src_camera, max_angle_deg=20.0, max_distance_ratio=None):
    ...


def reflection_consistency_loss(src, tgt, src_cam, tgt_cam, cfg):
    ...
    return ReflectionConsistencyStats(loss=loss, valid_count=valid_count, mean_weight=mean_weight)
```

核心伪代码：

```python
def reflection_consistency_loss(src, tgt, src_cam, tgt_cam, cfg):
    required = ["specular_rgb", "specular_confidence", "alpha_map", "depth_map", "normal_render"]
    if any(src.get(k) is None or tgt.get(k) is None for k in required):
        zero = src["render"].sum() * 0.0
        return ReflectionConsistencyStats(zero, 0, 0.0)

    src_depth = src["depth_map"].detach()
    tgt_depth = tgt["depth_map"].detach()
    points = backproject_depth(src_cam, src_depth)
    grid, projected_depth, valid = project_points(tgt_cam, points)

    sampled_spec = sample_map(tgt["specular_rgb"], grid)
    sampled_alpha = sample_map(tgt["alpha_map"].detach(), grid)
    sampled_depth = sample_map(tgt_depth, grid)

    src_spec = flatten_chw(src["specular_rgb"])
    src_alpha = flatten_chw(src["alpha_map"].detach())[:, 0]
    src_conf = flatten_chw(src["specular_confidence"].detach())[:, 0]
    tgt_alpha = flatten_chw(sampled_alpha)[:, 0]

    depth_ok = (flatten_chw(sampled_depth)[:, 0] - flatten_chw(projected_depth)[:, 0]).abs()
    depth_ok = depth_ok < cfg.ref_consistency_depth_tolerance * flatten_chw(projected_depth).abs()[:, 0].clamp_min(1e-6)

    if src.get("normal_depth") is not None:
        normal_agree = (src["normal_render"].detach() * src["normal_depth"].detach()).sum(0).clamp(0, 1).reshape(-1)
    else:
        normal_agree = torch.ones_like(src_alpha)

    refl_ok = src_conf > cfg.ref_consistency_refl_threshold
    mask = (
        flatten_valid(valid)
        & depth_ok
        & (src_alpha > cfg.ref_consistency_alpha_threshold)
        & (tgt_alpha > cfg.ref_consistency_alpha_threshold)
        & refl_ok
        & (normal_agree > cfg.ref_consistency_normal_threshold)
    )

    spec_intensity = src_spec.detach().mean(dim=-1).clamp(0, 1)
    refl_weight = src_conf.clamp(0, 1).pow(cfg.ref_consistency_gamma)
    weight = mask.float() * src_alpha * tgt_alpha * normal_agree * refl_weight * spec_intensity

    if weight.sum() <= 1e-6:
        return ReflectionConsistencyStats(src_spec.sum() * 0.0, 0, 0.0)

    residual = (src_spec.detach() - flatten_chw(sampled_spec)).abs().mean(dim=-1)
    loss = (weight * residual).sum() / weight.sum().clamp_min(1e-6)
    return ReflectionConsistencyStats(loss, int(mask.sum().item()), float(weight[mask].mean().detach().item()))
```

Stop-gradient 规则：

- source `specular_rgb` detach，target sampled spec 回传梯度，复用 RC-RefGS 当前方向。
- depth/alpha/normal/confidence 默认用于 mask/weight，建议 detach。
- 后续可 ablation symmetric loss，但不要作为首版默认。

### 3.5 `utils/loss_utils.py`

当前有 `smooth_img_loss()` 和 `bilateral_smooth_img_loss()`，但没有通用 TV loss。建议新增：

```python
def tv_loss(x):
    # x: [B,C,H,W] or [C,H,W]
    if x.dim() == 3:
        x = x[None]
    dh = (x[:, :, 1:, :] - x[:, :, :-1, :]).pow(2).mean()
    dw = (x[:, :, :, 1:] - x[:, :, :, :-1]).pow(2).mean()
    return dh + dw
```

3DGS-DR 的 smoothness 对象应是 `refl_strength_map`，不是 roughness：

```python
loss = loss + opt.lambda_refl_smooth * tv_loss(render_pkg["refl_strength_map"])
```

但因为 `lambda_refl_smooth` 当前默认 `0.4` 且过去未使用，不能直接打开。建议：

- 保持默认关闭：把 `lambda_refl_smooth` 默认改为 `0.0`，或新增 `--enable_refl_smooth_loss` gate。
- ablation variant 命名为 `refl_smooth_only`，不要叫 `rough_only`，避免语义混淆。

### 3.6 `arguments/__init__.py`

新增 RC 参数，默认全部关闭或保守：

```python
self.lambda_ref_consistency = 0.0
self.ref_consistency_start = 6000
self.ref_consistency_every = 4
self.ref_consistency_max_angle = 20.0
self.ref_consistency_gamma = 2.0
self.ref_consistency_alpha_threshold = 0.2
self.ref_consistency_refl_threshold = 0.05
self.ref_consistency_normal_threshold = 0.0
self.ref_consistency_depth_tolerance = 0.02
self.ref_consistency_min_valid = 64
self.rc_specular_mode = "color"  # "color" or "weighted"
```

建议：

- `ref_consistency_start` 不要早于 `init_until_iter`。首版可以用 `max(opt.ref_consistency_start, opt.init_until_iter + 1000)`。
- `lambda_ref_consistency` 首轮从 `1e-3`、`5e-3`、`1e-2`、`2e-2` 网格试验，不要直接上大权重。
- 对真实场景先降低 max angle 到 `10-15` 度，避免遮挡和深度错配。

### 3.7 `train.py`

修改目标：在主 RGB loss 后、`loss.backward()` 前接入 scheduled RC loss 和可选 reflection strength smoothness。

集成伪代码：

```python
from utils.loss_utils import l1_loss, ssim, tv_loss
from utils.rc_reflection_consistency import choose_pair_camera, reflection_consistency_loss

...
render_pkg = render(
    viewpoint_cam,
    gaussians,
    pipe,
    background,
    initial_stage=initial_stage,
    return_rc_buffers=(not initial_stage),
    rc_specular_mode=opt.rc_specular_mode,
)

Ll1 = l1_loss(image, gt_image)
loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

if (
    (not initial_stage)
    and opt.lambda_refl_smooth > 0
    and "refl_strength_map" in render_pkg
):
    loss = loss + opt.lambda_refl_smooth * tv_loss(render_pkg["refl_strength_map"])

use_rc = (
    (not initial_stage)
    and opt.lambda_ref_consistency > 0
    and iteration >= max(opt.ref_consistency_start, opt.init_until_iter + 1)
    and iteration % opt.ref_consistency_every == 0
)
if use_rc:
    pair_cam = choose_pair_camera(scene.getTrainCameras(), viewpoint_cam, opt.ref_consistency_max_angle)
    if pair_cam is not None:
        pair_pkg = render(
            pair_cam,
            gaussians,
            pipe,
            background,
            initial_stage=False,
            return_rc_buffers=True,
            rc_specular_mode=opt.rc_specular_mode,
        )
        rc_stats = reflection_consistency_loss(render_pkg, pair_pkg, viewpoint_cam, pair_cam, opt)
        if rc_stats.valid_count >= opt.ref_consistency_min_valid:
            loss = loss + opt.lambda_ref_consistency * rc_stats.loss
```

日志建议：

- TensorBoard scalar：
  - `train_loss_patches/rc_loss`
  - `train_loss_patches/rc_valid_count`
  - `train_loss_patches/rc_mean_weight`
  - `train_loss_patches/refl_smooth_loss`
- 终端每 100/1000 step 打印 RC valid count，避免 loss 静默为 0。

训练边界：

- RC loss 触发时需要额外 render target view，平均成本约增加 `1 / ref_consistency_every` 次 render。
- initial stage 不启用 RC，因为没有 deferred reflection buffers。
- 如果 depth/alpha 尚未实现，RC loss 必须默认跳过，不能用错误 proxy 产生看似正常的训练信号。

### 3.8 `eval.py` 和新增 metrics

当前 `eval.py` 只统计 test split 的 PSNR/SSIM/LPIPS/FPS，可保存 RGB/normal。需要新增两类评估：

1. full render quality：保留现有 PSNR/SSIM/LPIPS/FPS。
2. reflection consistency 和 reflective-region quality：
   - `mean_reflection_consistency`：越低越好。
   - `reflective_region_psnr/ssim/lpips`：用 `refl_strength_map > threshold` 且 alpha/depth valid 的区域。

建议新增文件：

- `metrics/reflection_consistency_eval.py`
- `metrics/render_quality_eval.py` 或在现有 `eval.py` 中加 `--eval_reflection_metrics`

命令建议：

```bash
python eval.py --model_path output/SCENE --save_images
python metrics/reflection_consistency_eval.py \
  --model_path output/SCENE \
  --split test \
  --max_angle 20 \
  --refl_threshold 0.05
```

如果仓库没有 `metrics/` 目录，可以创建；否则不要把大量 metric 逻辑塞进 `eval.py`。

### 3.9 Ablation runner

新增 dry-run-first runner，例如：

- `scripts/run_rc_3dgs_dr_ablation.py`

Variants：

- `base`：原始 3DGS-DR，不加 RC，不加 reflection smoothness。
- `rc`：只开启 `lambda_ref_consistency`。
- `wo_ref`：和 `rc` 相同实验配置但 `lambda_ref_consistency=0`，用于排除 runner 其他改动影响。
- `wo_conf`：保留 RC，但设置 `ref_consistency_gamma=0` 或改用 uniform confidence；注意这不是删除 alpha/depth/spec intensity 权重。
- `refl_smooth_only`：关闭 RC，只开启 `lambda_refl_smooth` 或 `enable_refl_smooth_loss`。

runner 必须支持：

- `--dry_run` 默认只打印命令，不启动训练。
- `--execute` 才启动训练。
- `--scene_root`、`--scene`、`--output_root`、`--iterations`、`--variant`。
- `--skip_complete` 默认开启，已有输出不重复跑。
- 输出 JSON/CSV summary，明确 completed/incomplete/OOM/excluded scene。

示例命令：

```bash
python scripts/run_rc_3dgs_dr_ablation.py \
  --scene_root data/ref_nerf/ref_synthetic \
  --scenes teapot toaster car \
  --variants base rc wo_conf refl_smooth_only \
  --output_root output/rc_3dgs_dr_ablation \
  --iterations 61000 \
  --dry_run
```

### 3.10 `README.md`、`train.sh`、`eval.sh` 和 docs

更新目标：让后续使用者知道 RC 迁移是可选增强，不是默认 3DGS-DR 行为。

建议改动：

- `README.md`：新增一个短节 `Reflection Consistency Extension`，说明默认关闭、需要 alpha/depth buffers、推荐先 dry-run ablation。
- `train.sh`：不要把所有示例都改成 RC；只新增 1-2 条注释形式的 RC 示例命令。
- `eval.sh`：新增 reflection metric 示例命令，但保留原始 `eval.py` 调用。
- `docs/rc_reflection_consistency_3dgs_dr.md`：记录 buffer contract、参数、ablation protocol 和已知风险。

### 3.11 `scene/gaussian_model.py`、`utils/general_utils.py`、`scene/cameras.py`

首版不建议在这些文件中引入大改：

- `scene/gaussian_model.py`：不要新增 roughness 参数；可只增加一个明确命名的 depth helper，或在 renderer 中直接计算 camera-space depth。
- `utils/general_utils.py`：当前 `sample_camera_rays()` 已服务 deferred reflection；如果新增 backprojection helper 需要复用 ray 逻辑，注意不要改变现有 reflection 渲染行为。
- `scene/cameras.py`：当前已有 `HWK`、`world_view_transform`、`full_proj_transform`、`camera_center`；RC helpers 应消费这些字段，不要改 camera 数据结构，除非发现 `HWK` 在某类数据上缺失。

## 4. RC loss 的 3DGS-DR 专用解释

### 4.1 Buffer 映射

| RC-RefGS 概念 | 3DGS-DR 替代 | 说明 |
| --- | --- | --- |
| `spec_light` / `specular_rgb` | `refl_color_map`，可 ablation 为 `refl_color_map * refl_strength_map` | 首选不改 deferred reflection，只约束环境反射颜色或最终反射贡献。 |
| `roughness_map` | `refl_strength_map` | 语义方向相反：roughness 低才高置信；refl_strength 高才高置信。 |
| `rend_alpha` | `alpha_map = 1 - final_T` | 需要 c7 rasterizer 暴露。 |
| `surf_depth` | expected `depth_map` | 需要 c7 rasterizer 暴露或 depth channel fallback。 |
| `rend_normal` | `normal_map` | 当前 renderer 已有。 |
| `surf_normal` | `depth_to_normal(depth_map)` | 新增。 |
| roughness smoothness | reflection strength smoothness | 只能作为材料/反射强度平滑对照，不是 RC loss。 |

### 4.2 Loss 公式

对 source view `s` 和 target view `t`：

1. 用 `D_s` 反投影 source pixel 到 world point。
2. 投影到 target view，得到 `grid_t` 和 target projected depth。
3. 用 `grid_sample(..., align_corners=True)` 采样 target `specular_rgb/alpha/depth`。
4. mask：
   - 投影在 target 内；
   - target sampled depth 与 projected depth 一致；
   - source/target alpha 足够高；
   - source `refl_strength` 足够高；
   - renderer normal 与 depth normal 一致。
5. weight：
   - `alpha_s * alpha_t * normal_agree * refl_strength_s ** gamma * mean(specular_rgb_s)`。
6. residual：
   - `abs(stopgrad(specular_rgb_s) - sampled_specular_rgb_t).mean(channel)`。

不要把 `refl_strength` 当 roughness 使用；不要写 `(1 - refl_strength) ** gamma`。

## 5. Verification protocol

完成代码改造后，至少运行：

```bash
python -m py_compile \
  train.py eval.py arguments/__init__.py gaussian_renderer/__init__.py \
  scene/gaussian_model.py scene/cameras.py scene/dataset_readers.py \
  utils/loss_utils.py utils/general_utils.py utils/graphics_utils.py utils/image_utils.py \
  net_viewer.py
```

如果新增 Python 文件：

```bash
python -m py_compile utils/rc_reflection_consistency.py scripts/run_rc_3dgs_dr_ablation.py
```

如果改了 CUDA extension：

```bash
pip install -e submodules/diff-gaussian-rasterization_c7
```

然后运行最小静态/功能检查：

```bash
python train.py --help | grep -E "ref_consistency|rc_specular|refl_smooth"
python scripts/run_rc_3dgs_dr_ablation.py --help
python scripts/run_rc_3dgs_dr_ablation.py --scene_root data/ref_nerf/ref_synthetic --scenes teapot --variants base rc --dry_run
git diff --check
```

如果有可用 GPU 和小数据集，再跑低迭代 smoke：

```bash
python -u train.py \
  -s data/ref_nerf/ref_synthetic/teapot \
  --eval --iterations 1200 --white_background \
  --lambda_ref_consistency 0.001 \
  --ref_consistency_start 600 \
  --ref_consistency_every 8 \
  --test_iterations 1200 \
  --save_iterations 1200
```

smoke 通过条件：

- 训练不 crash。
- initial stage 不触发 RC。
- deferred stage RC valid count 非零，或明确因缺 depth/alpha 被跳过。
- `render_pkg` 包含 `specular_rgb/specular_confidence/alpha_map/depth_map/normal_render/normal_depth`。
- `eval.py` 仍可输出原始 PSNR/SSIM/LPIPS/FPS。

## 6. Ablation protocol 和结果边界

正式 ablation 至少包含：

- `base`
- `rc`
- `wo_ref`
- `wo_conf`
- `refl_smooth_only`

每个 variant 报告：

- full PSNR/SSIM/LPIPS/FPS；
- mean reflection consistency，越低越好；
- reflective-region PSNR/SSIM/LPIPS；
- RC valid pixel count 分布；
- failed/OOM/incomplete scene 列表。

解释边界：

- RC-RefGS reuse guide 的稳定证据主要支持 reflection consistency 改善；render quality 是 mixed/tradeoff，不能预设迁移到 3DGS-DR 后 PSNR 必然提高。
- confidence-aware TSDF/mesh extraction 不是本次 3DGS-DR 迁移首要目标；除非新增 geometry/mesh 指标，否则不要写“几何质量提升”。
- Shiny Blender Real 或其他真实场景如果 OOM/incomplete，必须单独列出，不并入完整平均结论。
- `refl_smooth_only` 只能说明 reflection strength smoothness 的对照效果，不能替代跨视角 RC loss。

### 6.1 Confidence-aware TSDF 是否迁移

不作为首版迁移目标。RC-RefGS reuse guide 中的 confidence-aware TSDF 是 mesh extraction 工程增强：用 `alpha * normal_agree` 过滤低可信 depth，再做 TSDF fusion。3DGS-DR 当前仓库没有对应 mesh extraction 主流程；除非你同时新增 mesh extraction 和几何指标，否则不要把它写进 RC 训练贡献。

如果后续确实要迁移：

- 前提是 renderer 已可靠返回 `alpha_map/depth_map/normal_render/normal_depth`。
- confidence 可用 `alpha_map * clamp(dot(normal_render, normal_depth), 0, 1)`，而不是 `refl_strength`。
- 只允许声明为 mesh extraction 的工程过滤模块；必须用 Chamfer、normal consistency、mesh completeness 等几何指标单独验证。

## 7. 风险与常见坑

- c7 rasterizer 当前只返回 color/radii；alpha/depth 不暴露时，RC loss 不完整。
- `ImageState::accum_alpha` 实际存的是 final transmittance `T`，alpha 应是 `1 - T`。
- expected depth 必须与 target projection depth 使用同一坐标/尺度；否则 depth consistency mask 会错误。
- `normal_map` 和 `depth_to_normal()` 必须在同一坐标系；否则 normal agreement 和 reflection direction 都会错。
- `grid_sample align_corners=True` 与 NDC grid 必须匹配。
- `refl_color_map` 与 `render` 均在 `[0,1]`，不要混入 RC-RefGS 的 `linear2srgb` 假设。
- `lambda_refl_smooth` 当前默认存在但未使用；直接启用默认 `0.4` 风险很高，首版应默认关闭或加 gate。
- `wo_conf` 需要精确定义：`gamma=0` 只中和 `refl_strength ** gamma`，不是删除 alpha/depth/spec intensity 等所有权重。
- RC 额外 target render 会增加显存和时间；先用 `every=4/8`，再调 lambda。
- 如果 weight sum 经常为 0，训练不会报错但没有 RC 信号；必须记录 valid count。

## 8. 推荐提交顺序

1. Renderer buffer contract：补 `return_rc_buffers`、`specular_rgb/specular_confidence`，不启用 RC。
2. c7 alpha/depth exposure：补 `alpha_map/depth_map`，加最小 smoke。
3. Geometry helpers：实现 `backproject_depth/project_points/depth_to_normal`。
4. RC loss：新增 `utils/rc_reflection_consistency.py`，单元测试 shape/mask/zero-weight。
5. Training schedule：接入 `train.py` 和 `arguments/__init__.py`，默认关闭。
6. Metrics：新增 reflection consistency / reflective-region evaluation。
7. Ablation runner：dry-run first，再执行小规模 smoke。
8. Docs：更新 README 或新增 `docs/rc_reflection_consistency.md`，明确代码事实、迁移建议、实验边界。

## 9. Codex 分析过程摘要

本 prompt 生成前已做的分析：

- 使用 `using-superpowers`：先检查当前 skills 工作流。
- 使用 `writing-plans`：把迁移需求组织成文件级、函数级、验证级的可执行计划。
- 使用 `verification-before-completion`：要求完成前运行轻量静态检查。
- 参考但未执行 `subagent-driven-development` / `executing-plans` / `dispatching-parallel-agents`：当前任务是单个文档交付，且 subagent 工具只允许在用户显式要求代理/并行代理时使用。
- 检查了当前可用 skills；本环境有 `using-superpowers`、`writing-plans`、`executing-plans`、`subagent-driven-development`、`dispatching-parallel-agents`、`verification-before-completion` 等工作流技能，但没有单独命名的 repository/codebase analysis 或 documentation writing skill。

已阅读/核对：

- `docs/rc_refgs_optimization_modules_reuse_guide.md`，完整 576 行。
- 3DGS-DR 指定源码文件和可选文件/目录：`README.md`、`train.py`、`arguments/__init__.py`、`gaussian_renderer/__init__.py`、`scene/gaussian_model.py`、`utils/loss_utils.py`、`utils/general_utils.py`、`utils/graphics_utils.py`、`eval.py`、`train.sh`、`eval.sh`、`scene/cameras.py`、`scene/dataset_readers.py`、`utils/image_utils.py`、`net_viewer.py`、`submodules/diff-gaussian-rasterization_c7`、`submodules/diff-gaussian-rasterization_c3`。

未验证内容：

- 未启动训练、评估或 CUDA extension 重编译。
- 未验证 alpha/depth extension 的具体 CUDA patch 是否可编译；文中给出的是实现要求和推荐路径。
- 未验证 RC loss 在 3DGS-DR 上的指标收益；所有收益都必须通过 ablation protocol 实测。
- 未从 GitHub 网络重新拉取源码；本分析基于当前工作区的 3DGS-DR 本地源码。
