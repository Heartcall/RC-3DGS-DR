# RC-RefGS 优化模块复用说明书

## 0. 结论摘要

- RC-RefGS 相对 Ref-GS 最可复用的核心是：让 renderer 显式返回 specular、roughness、alpha、depth、normal 等中间缓冲，再用几何对应把不同视角的 specular 输出约束到一致。
- Reflection Consistency Loss 是主要方法模块；它不依赖 Gaussian 参数化本身，但依赖可投影深度、可采样 specular 分支、opacity/confidence 和反射区域判定。
- `roughness_map` 在当前实现中既参与 Ref-GS 的方向光照建模，也参与 RC loss 的置信度加权；迁移到没有 roughness 的方法时，应替换为“反射置信度”而不是强行伪造物理 roughness。
- `lambda_roughness_smoothness + tv_loss(roughness_map)` 是可复用的材料正则/ablation control，但不应和跨视角 RC loss 混为同一理论贡献。
- confidence-aware TSDF/mesh extraction 使用 `alpha * normal_agree` 过滤低置信深度，是工程增强模块；除非有独立几何指标支持，不应宣称它证明 RC-RefGS 几何必然更优。
- 已完成 FD-P2-lite/non-Shiny-Real 范围内，RC 主要稳定改善 `mean_reflection_consistency`；渲染质量指标是 mixed/tradeoff，不能宣称全局 PSNR/SSIM/LPIPS 或几何质量全面优于 Ref-GS。
- 迁移到 3DGS、2DGS、GaussianShader、Deferred Reflection GS 或 NeRF-like reflective reconstruction 时，优先抽象成 `render(view) -> buffers` 接口和 scheduled pairwise consistency loss，而不是复制 Ref-GS 的全部 PBR/MLP 参数结构。

## 1. RC-RefGS 相对 Ref-GS 的总体改动

| 原 Ref-GS 机制 | RC-RefGS 新增机制 | 解决的问题 | 代码/日志证据 |
| --- | --- | --- | --- |
| 单视角监督主要由 PBR RGB 重建项驱动。 | 额外暴露 `spec_light`、`roughness_map`、`surf_depth`、`rend_normal/surf_normal` 等 buffers，并在训练中额外 render pair view。 | 把 specular fitting 从单视角颜色拟合转成跨视角几何对应约束。 | `gaussian_renderer/__init__.py:84`、`gaussian_renderer/__init__.py:202`、`train.py:140` |
| Ref-GS 已有 albedo/roughness/feature/Sph-Mip/light MLP 的方向分解。 | `reflection_consistency_loss` 使用 specular branch 输出、粗糙度、alpha、深度和 normal agreement 构造权重与 mask。 | 抑制同一 3D 表面在相邻视角下 specular 预测不一致。 | `scene/gaussian_model.py:117`、`utils/reflection_consistency.py:119`、`utils/reflection_consistency.py:176` |
| 基础 loss 为 L1 + DSSIM，另有 alpha warmup 和 normal regularization。 | 增加 scheduled `lambda_ref_consistency` 与可选 roughness TV 正则。 | 控制额外 loss 只在训练稳定后、按间隔启用，降低成本和早期错误几何影响。 | `train.py:114`、`train.py:119`、`train.py:124`、`train.py:136`、`train.py:140` |
| Mesh extraction 使用渲染深度做 TSDF 融合。 | 增加 `conf = alpha * clamp(dot(rend_normal, surf_normal),0,1)`，可按 `conf_threshold` mask depth。 | 过滤低 alpha 或 normal 不一致区域，减少 TSDF 融合噪声。 | `utils/mesh_utils.py:120`、`utils/mesh_utils.py:124`、`utils/mesh_utils.py:154`、`utils/mesh_utils.py:183` |
| 评估可看重建质量。 | 增加 reflection consistency eval、reflective-region PSNR、full/reflective PSNR/SSIM/LPIPS 和 ablation variants。 | 区分“反射一致性改善”和“渲染质量改善”，避免过度声明。 | `metrics/reflection_consistency_eval.py:113`、`metrics/render_quality_eval.py:116`、`scripts/run_rc_refgs_ablation_direct.py:20` |
| 子集实验可能只覆盖 teapot/toaster/car 等。 | 后续计划要求 claim-bearing 范围覆盖 Shiny Blender Synthetic、Shiny Blender Real、Glossy Synthetic。 | 防止把非完整数据集证据写成全局结论。 | `docs/superpowers/plans/2026-05-19-rc-refgs-full-implementation-and-experiment-roadmap.md:14`、`docs/superpowers/logs/rc-refgs-fd-p2-lite-final-results-analysis-2026-06-01.md:12` |

## 2. 模块 A：Renderer Intermediate Buffers

### 目的

Renderer intermediate buffers 是 RC-RefGS 可复用性的入口。RC loss、roughness smoothness、reflection metrics 和 confidence-aware mesh extraction 都不直接操作 Gaussian 内部参数，而是消费 renderer 返回的图像空间 buffers。

### 涉及文件

- `gaussian_renderer/__init__.py:84`：读取 `pc.get_albedo`、`pc.get_roughness`、`pc.get_language_feature` 并拼接到 rasterizer feature 输入。
- `gaussian_renderer/__init__.py:90`：rasterizer 返回 `albedo_map`、`out_ts`、`radii`、`allmap`。
- `gaussian_renderer/__init__.py:102`：从 `allmap` 解析 `rend_alpha`、`rend_normal`、depth/distortion。
- `gaussian_renderer/__init__.py:125`：由 view ray 和 normal 计算 reflection direction `wo`。
- `gaussian_renderer/__init__.py:151`：用 `dir_encoding` 和 `light_mlp` 预测 specular light。
- `gaussian_renderer/__init__.py:202`：统一返回 `pbr_rgb`、`spec_light`、`diff_light`、`roughness_map`、`reflection_dir`、`feature_map`、`rend_alpha`、`rend_normal`、`surf_depth`、`surf_normal` 等 key。

### 关键变量

| Buffer | 近似 shape | 来源 | 语义 | 当前使用位置 |
| --- | --- | --- | --- | --- |
| `pbr_rgb` | `[3,H,W]` | `spec_light + diff_light` 后 `linear2srgb`/clamp，再 scatter。 | 最终 PBR 渲染颜色；训练和 render-quality metric 使用时再 alpha composite 到背景。 | `gaussian_renderer/__init__.py:172`、`train.py:114`、`metrics/render_quality_eval.py:69` |
| `spec_light` | `[3,H,W]` | `reflection_dir + roughness + feature_map` 经 Sph-Mip encoding 和 `light_mlp`。 | specular 分支输出；当前返回值不是最终 sRGB composite。 | `gaussian_renderer/__init__.py:151`、`utils/reflection_consistency.py:129` |
| `diff_light` | `[3,H,W]` | rasterized `albedo_map`。 | diffuse/albedo 分支输出。 | `gaussian_renderer/__init__.py:169` |
| `roughness_map` | `[1,H,W]` | Gaussian roughness 与 feature 拼接后 rasterize，取 `out_ts[..., :1]`。 | 表面粗糙度/反射置信度线索。 | `gaussian_renderer/__init__.py:132`、`train.py:136`、`utils/reflection_consistency.py:133` |
| `reflection_dir` | `[3,H,W]` | `reflect(-viewdirs, rend_normal)`。 | 反射方向，用于 directional encoding 和分析。 | `gaussian_renderer/__init__.py:125`、`gaussian_renderer/__init__.py:194` |
| `feature_map` | `[4,H,W]` | `pc.get_language_feature` rasterized 后归一化。 | specular MLP 的 per-point/per-pixel feature。 | `gaussian_renderer/__init__.py:86`、`gaussian_renderer/__init__.py:133` |
| `rend_alpha` | `[1,H,W]` | `allmap[1:2]`。 | rasterized opacity/visibility。 | `gaussian_renderer/__init__.py:102`、`train.py:114`、`utils/reflection_consistency.py:130` |
| `rend_normal` | `[3,H,W]` | `allmap[2:5]` 变换并 normalize。 | rasterizer/primitive normal。 | `gaussian_renderer/__init__.py:104`、`train.py:125`、`utils/mesh_utils.py:120` |
| `surf_depth` | `[1,H,W]` | expected depth 与 median depth 按 `pipe.depth_ratio` 混合。 | 用于 backprojection/project、mesh extraction。 | `gaussian_renderer/__init__.py:111`、`utils/reflection_consistency.py:135`、`utils/mesh_utils.py:116` |
| `surf_normal` | `[3,H,W]` | `depth_to_normal(viewpoint_camera, surf_depth)` 并乘 detached alpha。 | depth-derived normal，用于 normal agreement。 | `gaussian_renderer/__init__.py:119`、`train.py:125`、`utils/mesh_utils.py:121` |

### 输入输出与依赖

- 最小输入：camera intrinsics/extrinsics、primitive/render representation、可导 RGB/specular 渲染路径、depth/alpha/normal 输出。
- 当前 Ref-GS 依赖：Gaussian albedo/roughness/mask/feature、Sph-Mip directional encoding、light MLP、2DGS-style normal/depth allmap。
- 最小输出：`spec_light`、`rend_alpha`、`roughness_map 或 specular_confidence`、`surf_depth`、`rend_normal`、`surf_normal`。

### 迁移接口

迁移建议：其他方法不需要复制 `SphMipEncoding + light_mlp`，但必须提供语义等价的图像空间 buffers：

```python
render(view) -> {
    "rgb": Tensor[3,H,W],
    "alpha": Tensor[1,H,W],
    "depth": Tensor[1,H,W],
    "normal_render": Tensor[3,H,W],
    "normal_depth": Tensor[3,H,W],
    "specular_rgb": Tensor[3,H,W],
    "roughness_or_confidence": Tensor[1,H,W],
}
```

### 伪代码

```python
def render_with_rc_buffers(view, model):
    base = model.render(view)
    alpha = base["alpha"]
    depth = base["depth"]
    normal_render = base["normal_render"]
    normal_depth = depth_to_normal(view, depth)
    specular_rgb = base.get("specular_rgb", estimate_specular_residual(base))
    confidence = base.get("roughness_map", estimate_reflective_confidence(base))
    return {
        **base,
        "specular_rgb": specular_rgb,
        "alpha": alpha,
        "depth": depth,
        "normal_render": normal_render,
        "normal_depth": normal_depth,
        "roughness_or_confidence": confidence,
    }
```

## 3. 模块 B：Reflection Consistency Loss

### 目的

Reflection Consistency Loss 用 source view 的深度把像素反投影到 3D，再投影到 target view，在 target specular map 上采样对应点，并约束 source/target 的 specular 输出一致。这样把“每个视角各自拟合高光颜色”的问题，转成“同一几何表面在可见相邻视角下的 specular 表达应一致”的跨视角约束。

### 数学形式

对 source view `s` 和 target view `t`：

1. `X_s(u) = backproject_s(u, D_s(u))`
2. `(v, z_t_pred) = project_t(X_s(u))`
3. `S_t(v) = grid_sample(S_t, v)`
4. `M(u) = valid * depth_ok * alpha_ok * roughness_ok * normal_ok`
5. `w(u) = M(u) * alpha_s(u) * alpha_t(v) * normal_agree(u) * mean(S_s(u)) * (1 - roughness_s(u))^gamma`
6. `L_rc = sum_u w(u) * |stopgrad(S_s(u)) - S_t(v)| / (sum_u w(u) + eps)`

代码中 residual 为 L1 mean over RGB channels：`(src_spec.detach() - sampled_spec).abs().mean(dim=-1)`。

### 代码路径

- `utils/reflection_consistency.py:32`：`choose_pair_camera` 根据相机中心方向夹角/距离选择 pair。
- `utils/reflection_consistency.py:60`：`backproject_depth` 根据 FoV 和 `world_view_transform` 将 source depth 反投影到 world points。
- `utils/reflection_consistency.py:91`：`project_points` 使用 target `full_proj_transform` 输出 NDC grid、projected depth、valid mask。
- `utils/reflection_consistency.py:103`：`_sample_map` 使用 `F.grid_sample(..., align_corners=True)` 采样 target maps。
- `utils/reflection_consistency.py:119`：`reflection_consistency_loss` 主函数。
- `utils/reflection_consistency.py:170`：depth consistency mask。
- `utils/reflection_consistency.py:176`：valid/depth/alpha/roughness/normal mask。
- `utils/reflection_consistency.py:185`：权重组合。
- `utils/reflection_consistency.py:190`：source spec stop-gradient。

### 关键 mask

| Mask/权重项 | 代码证据 | 含义 | 迁移注意事项 |
| --- | --- | --- | --- |
| projection valid | `utils/reflection_consistency.py:98` | 投影在 target NDC 内且深度为正。 | 不同 renderer 的 NDC 范围和相机矩阵约定必须对齐。 |
| depth consistency | `utils/reflection_consistency.py:170` | sampled target depth 与投影深度相差小于 relative tolerance。 | depth convention 不一致会直接造成 false negative/false positive。 |
| alpha threshold | `utils/reflection_consistency.py:178` | source/target 都应为可见表面。 | NeRF-like 方法可用 opacity/accumulated transmittance 替代。 |
| roughness threshold | `utils/reflection_consistency.py:180` | 只在较镜面区域强化 specular consistency。 | 无 roughness 方法应使用 reflective confidence 或 spec branch activation 替代。 |
| normal agreement | `utils/reflection_consistency.py:165` | raster normal 与 depth normal 一致时才可信。 | normal 坐标系必须统一。 |
| spec confidence | `utils/reflection_consistency.py:172` | `mean(spec) * (1 - roughness)^gamma`。 | gamma=0 只去掉 roughness exponent，不去掉 spec intensity 权重。 |

### 超参数

| 参数 | 代码默认值 | 代码位置 | 含义 | 迁移建议 |
| --- | ---: | --- | --- | --- |
| `gamma` / `ref_consistency_gamma` | `2.0` | `utils/reflection_consistency.py:124`、`arguments/__init__.py:117` | 控制 `(1-roughness)^gamma` 对低粗糙度区域的偏好。 | 从 `1.0-2.0` 起步；如果反射 mask 过稀，可降低 gamma。 |
| `alpha_threshold` | `0.2` | `utils/reflection_consistency.py:125` | 可见性阈值。 | 对半透明/体渲染方法需要按 opacity 分布重调。 |
| `roughness_threshold` | `0.6` | `utils/reflection_consistency.py:126` | 反射区域阈值。 | 若 roughness 未校准，应替换为 learned confidence threshold。 |
| `depth_tolerance` | `0.02` | `utils/reflection_consistency.py:127` | target depth 与投影深度的相对容忍度。 | depth 噪声大或尺度不同的方法要先做单位/尺度校准。 |
| `ref_consistency_max_angle` | `20.0` | `arguments/__init__.py:116` | camera pair 最大夹角。 | 角度过大易产生错误对应；小 baseline 数据可从 `10-20` 度试起。 |

### 计算流程

1. 从训练 camera pool 中选择 source 的 target pair。
2. 对 source view 和 target view 各 render 一次，拿到 specular/alpha/depth/normal/roughness。
3. 用 source `surf_depth` 反投影到 world。
4. 把 world points 投影到 target NDC grid。
5. 用 `grid_sample` 在 target 上采样 `spec_light`、`rend_alpha`、`surf_depth`。
6. 组合 projection/depth/alpha/roughness/normal mask。
7. 对 source spec detach，对 target sampled spec 回传梯度。
8. 用置信度加权 L1 得到 scalar loss。

### 可复用伪代码

```python
def reflection_consistency_loss(src, tgt, src_cam, tgt_cam, gamma=2.0):
    points = backproject(src_cam, src["depth"])
    grid, projected_depth, valid = project(tgt_cam, points)

    tgt_spec = grid_sample(tgt["specular_rgb"], grid, align_corners=True)
    tgt_alpha = grid_sample(tgt["alpha"], grid, align_corners=True)
    tgt_depth = grid_sample(tgt["depth"], grid, align_corners=True)

    src_spec = flatten_hw(src["specular_rgb"])
    src_alpha = flatten_hw(src["alpha"])
    src_depth_conf = depth_consistency(tgt_depth, projected_depth)
    normal_agree = dot(src["normal_render"], src["normal_depth"]).clamp(0, 1)
    refl_conf = reflective_confidence(src, gamma)

    mask = valid & src_depth_conf & (src_alpha > 0.2) & (tgt_alpha > 0.2)
    weight = mask.float() * src_alpha * tgt_alpha * normal_agree * refl_conf
    residual = (src_spec.detach() - tgt_spec).abs().mean(dim=-1)
    return (weight * residual).sum() / weight.sum().clamp_min(1e-6)
```

### 迁移注意事项

- 迁移建议：如果目标方法的 specular branch 是 view-dependent residual，而不是物理 specular light，也可以替代 `spec_light`，但必须保证颜色空间一致。
- 迁移建议：如果不希望梯度只从 target 路径回传，可以做 symmetric loss：`L(s->t) + L(t->s)`；这不是当前仓库事实，需要单独 ablation。
- 迁移建议：camera pair 选择最好加入 frustum overlap 或 depth overlap 检查；当前实现主要按角度/距离选 pair。

## 4. 模块 C：Training Integration and Scheduling

### Loss 组合

当前 `train.py` 的训练 loss 由以下部分组成：

| Loss | 代码证据 | 触发条件 | 作用 |
| --- | --- | --- | --- |
| PBR RGB L1 + DSSIM | `train.py:114`、`utils/loss_utils.py:17`、`utils/loss_utils.py:43` | 每次迭代 | 主重建监督。 |
| alpha BCE warmup | `train.py:119`、`utils/loss_utils.py:81` | `iteration < 3000` | 早期稳定 foreground/opacity。 |
| normal regularization | `train.py:124` | 每次迭代，代码中硬编码 `lambda_normal=0.05` | 约束 raster normal 与 depth normal 一致。 |
| roughness smoothness | `train.py:136`、`utils/loss_utils.py:91` | `lambda_roughness_smoothness > 0` 且迭代达到 start | 材料图像空间 TV 正则。 |
| reflection consistency | `train.py:140`、`utils/reflection_consistency.py:119` | lambda>0、达到 start、按 every 触发、存在 pair camera | 跨视角 specular consistency。 |

### 新增/相关超参数

| 参数 | 代码默认值 | 代码证据 | 说明 |
| --- | ---: | --- | --- |
| `lambda_ref_consistency` | `0.0` | `arguments/__init__.py:113` | RC loss 权重；默认关闭。 |
| `ref_consistency_start` | `3000` | `arguments/__init__.py:114` | 迭代到该步后才启用 RC。 |
| `ref_consistency_every` | `4` | `arguments/__init__.py:115` | 每隔多少 iteration 计算一次 RC。 |
| `ref_consistency_max_angle` | `20.0` | `arguments/__init__.py:116` | pair camera 最大角度。 |
| `ref_consistency_gamma` | `2.0` | `arguments/__init__.py:117` | roughness confidence exponent。 |
| `lambda_roughness_smoothness` | `0.0` | `arguments/__init__.py:118` | roughness TV 权重；默认关闭。 |
| `roughness_smoothness_start` | `3000` | `arguments/__init__.py:119` | roughness TV 启动迭代。 |

### 代码默认值 vs 经验建议

- 代码默认值：RC loss 和 roughness smoothness 都是 off-by-default；直接运行默认训练不会启用这两个模块。
- 代码默认值：`lambda_normal` 参数存在于 `arguments/__init__.py:112`，但当前 `train.py:124` 主路径硬编码 `lambda_normal=0.05`；不要在迁移文档中假设它已完全由 CLI 控制。
- 已有 ablation runner 配置：`scripts/run_rc_refgs_ablation_direct.py:139` 中 `rc` variant 默认开启 `--lambda_ref_consistency`，runner 默认 lambda 为 `0.02`；这是实验 runner 配置，不是训练代码默认值。
- 迁移建议：第一次迁移可从较小 `lambda_ref_consistency` 起步，例如 `1e-3` 到 `2e-2`，延后到几何/alpha 相对稳定后启用，并用 `every=4` 或更稀疏频率控制成本。
- 迁移建议：若 target projection 错配较多，先降低 pair angle、提高 alpha/depth/normal mask 严格性，再调大 lambda。

### 额外 render 成本

- RC loss 触发时需要额外 render 一个 pair view，训练成本近似增加 `1 / ref_consistency_every` 个 render。
- 当前默认 `ref_consistency_every=4`，意味着平均每 4 次迭代增加一次 target render。
- 对显存敏感方法，迁移建议先使用低频 RC 或 detach 更多 target-independent buffers，再逐步放开。

### 可开关配置

```bash
# base: 不加 RC 和 roughness smoothness
python train.py ...

# rc: 开启跨视角 reflection consistency
python train.py ... \
  --lambda_ref_consistency 0.02 \
  --ref_consistency_start 3000 \
  --ref_consistency_every 4 \
  --ref_consistency_max_angle 20 \
  --ref_consistency_gamma 2

# rough_only: 只开 roughness TV
python train.py ... \
  --lambda_ref_consistency 0.0 \
  --lambda_roughness_smoothness 0.01 \
  --roughness_smoothness_start 3000
```

## 5. 模块 D：Roughness Smoothness

### 作用

Roughness Smoothness 在图像空间对 `roughness_map` 加 TV loss：

- 代码证据：`train.py:136` 使用 `tv_loss(render_pkg["roughness_map"][None])`。
- TV 实现：`utils/loss_utils.py:91` 对水平/垂直相邻差分平方求平均。
- 触发时机：`lambda_roughness_smoothness > 0` 且 `iteration >= roughness_smoothness_start`。

### 风险

- 过强 TV 会把真实材料变化、边缘高光或局部反射区域抹平。
- 如果 roughness 只是网络内部 latent，而非可解释物理量，TV 可能约束错误对象。
- 在没有可靠 reflection mask 的数据上，roughness smoothness 可能改善视觉平滑但削弱反射区域差异。

### 适用条件

- 材料区域连续、roughness 语义较稳定。
- 需要控制 `roughness_map` 噪声，以便 reflection confidence 更稳定。
- 希望做 `rough_only` control，区分“材料平滑”与“跨视角反射一致性”的贡献。

### 与 RC loss 的关系

Roughness Smoothness 不是 RC loss 的替代项：

- RC loss 使用几何对应和 target view specular 采样，目标是跨视角一致性。
- Roughness TV 只约束单视角图像空间粗糙度平滑。
- ablation runner 明确把 `rough_only` 设计为关闭 RC、开启 roughness TV 的 control。代码证据：`scripts/run_rc_refgs_ablation_direct.py:188`。

### Ablation 方式

- `base`：不启用 RC，不启用 roughness smoothness。
- `rc`：启用 RC。
- `rough_only`：关闭 RC，只启用 roughness smoothness。
- 解释结果时应报告：roughness smoothness 是否单独复现了 RC 的 reflection-consistency 改善；FD-P2-lite 日志显示它没有稳定复现 RC 的一致性收益，见 `docs/superpowers/logs/rc-refgs-fd-p2-lite-final-results-analysis-2026-06-01.md:137`。

## 6. 模块 E：Confidence-Aware Mesh Extraction

### 目的

Confidence-aware mesh extraction 使用 alpha 和 normal agreement 过滤低可信深度，再做 TSDF fusion。它服务于 mesh extraction 的工程鲁棒性，不是 RC loss 的核心理论。

### 代码路径

- `utils/mesh_utils.py:109`：`GaussianExtractor.reconstruction` 遍历训练/测试相机 render。
- `utils/mesh_utils.py:120`：计算 `normal_agree = dot(rend_normal, surf_normal).clamp(0,1)`。
- `utils/mesh_utils.py:124`：计算 `conf = alpha * normal_agree`。
- `utils/mesh_utils.py:154`：`extract_mesh_bounded(..., conf_threshold=0.0)`。
- `utils/mesh_utils.py:183`：`depth[self.confmaps[i] < conf_threshold] = 0`。
- `utils/mesh_utils.py:194`：Open3D TSDF integrate。

### 置信度公式

```python
normal_agree = clamp(dot(rend_normal, surf_normal), 0, 1)
conf = rend_alpha * normal_agree
```

含义：

- `rend_alpha` 表示该像素是否有稳定可见表面。
- `normal_agree` 表示 primitive/raster normal 与 depth-derived normal 是否一致。
- 两者相乘后，低透明度或 normal 不一致区域会被视为低可信深度。

### TSDF 接入方式

当前 bounded path 不是把 `conf` 作为连续 TSDF 权重，而是门控 depth：

```python
if conf_threshold > 0:
    depth[conf < conf_threshold] = 0
tsdf_volume.integrate(rgbd, intrinsic, extrinsic)
```

这意味着：

- 低置信像素不参与融合。
- 真正 TSDF 权重和融合细节由 Open3D `ScalableTSDFVolume` 管理。
- 该模块可以独立迁移到任何能 render `alpha/depth/normal` 的方法。

### 适用范围

- 适合 2DGS/3DGS-like surfel/splatting 方法，因为这些方法通常能输出 alpha、depth、normal。
- NeRF-like 方法可用 accumulated opacity 和 density/depth normal 替代。
- 如果没有 depth-derived normal，可先用 depth finite difference 计算；但要验证坐标系。

### 贡献边界

实验/论文表达建议：

- 可以说：这是 confidence-aware TSDF 的工程增强模块。
- 不应说：RC loss 必然提升几何或 mesh quality。
- 只有在 Chamfer、normal consistency、mesh completeness 等几何指标上完成对照后，才可升级几何质量结论。
- 计划文档也把该模块列为 supporting component，不是 core novelty，见 `docs/superpowers/plans/2026-05-16-rc-refgs-research-plan.md:54`。

## 7. 模块 F：Evaluation and Ablation Protocol

### 指标

| 指标 | 代码/日志证据 | 方向 | 说明 |
| --- | --- | --- | --- |
| `mean_reflection_consistency` | `metrics/reflection_consistency_eval.py:251` | 越低越好 | pair views 上的 reflection consistency loss 均值。 |
| `reflective_region_psnr` | `metrics/reflection_consistency_eval.py:113` | 越高越好 | `alpha > threshold` 且 `roughness < threshold` 区域 PSNR。 |
| `full_psnr/full_ssim/full_lpips` | `metrics/render_quality_eval.py:116` | PSNR/SSIM 越高越好，LPIPS 越低越好 | 全图渲染质量。 |
| `reflective_psnr/ssim/lpips` | `metrics/render_quality_eval.py:77`、`metrics/render_quality_eval.py:116` | 同上 | 反射区域渲染质量。 |

### Runner variants

| Variant | 代码证据 | 含义 |
| --- | --- | --- |
| `base` | `scripts/run_rc_refgs_ablation_direct.py:137` | 默认 Ref-GS，不加 RC 训练参数。 |
| `rc` | `scripts/run_rc_refgs_ablation_direct.py:139` | 开启 reflection consistency loss。 |
| `wo_ref` | `scripts/run_rc_refgs_ablation_direct.py:155` | 关闭 RC loss，用于检查 RC loss 是否必要。 |
| `wo_conf` | `scripts/run_rc_refgs_ablation_direct.py:171` | 保留 RC loss，但使用 `wo_conf_gamma`；默认 gamma=0，去掉 roughness exponent 的置信度偏好，但仍保留 spec intensity 权重。 |
| `rough_only` | `scripts/run_rc_refgs_ablation_direct.py:188` | 关闭 RC loss，只启用 roughness smoothness。 |

### 实验结论边界

来自 `docs/superpowers/logs/rc-refgs-fd-p2-lite-final-results-analysis-2026-06-01.md` 和对应 CSV/JSON：

- FD-P2-lite/non-Shiny-Real 完成范围：main `28/28`、ablation `42/42`、complete-metric `70/70`；证据见 `docs/superpowers/logs/rc-refgs-fd-p2-lite-final-results-analysis-2026-06-01.md:5`。
- RC 在 scoped 数据上强支持改善 reflection consistency：train `14/14`、test `13/14` favor RC；证据见 `docs/superpowers/logs/rc-refgs-fd-p2-lite-final-results-analysis-2026-06-01.md:39` 和 `docs/superpowers/figures/fd-p2-lite/table2_rc_win_counts_by_metric.csv`。
- 渲染质量不是稳定全面改善：full PSNR/SSIM/LPIPS 多数但非全部 favor RC，reflective PSNR/SSIM/LPIPS 明显 mixed；证据见 `docs/superpowers/logs/rc-refgs-fd-p2-lite-final-results-analysis-2026-06-01.md:39`、`docs/superpowers/logs/rc-refgs-fd-p2-lite-final-results-analysis-2026-06-01.md:130`。
- RC 改善 consistency 同时至少恶化一个 render-quality metric 的 scene/split rows 为 `21`；证据见 `docs/superpowers/logs/rc-refgs-fd-p2-lite-final-results-analysis-2026-06-01.md:168`。
- Shiny Blender Real 未纳入 FD-P2-lite 完整结论，存在 persistent OOM blocker；证据见 `docs/superpowers/logs/rc-refgs-fd-p2-lite-final-results-analysis-2026-06-01.md:12`。
- 2026-05-29 时 full main/ablation 仍不完整：main `29/34`、complete pairs `14/17`、ablation `42/51`；证据见 `docs/superpowers/logs/rc-refgs-current-main-and-ablation-analysis-2026-05-29.md:13`。

### 推荐复现实验

迁移到其他方法时，建议至少跑以下 protocol：

1. `base`：目标方法原始训练。
2. `rc`：只加入 reflection consistency loss。
3. `wo_ref`：保留其他训练设置但关闭 RC loss。
4. `wo_conf`：去掉/中和 reflection confidence 或 roughness exponent。
5. `rough_only`：只加入 roughness/confidence smoothness。
6. 每个 variant 统计 train/test reflection consistency、full quality、reflective-region quality。
7. 对 OOM/incomplete/excluded scene 单独列出，不并入平均结论。

## 8. 迁移到其他方法的通用接口

### 抽象接口

```python
def render(view) -> dict:
    return {
        "rgb": rgb,                              # [3,H,W], final rendered color
        "alpha": alpha,                          # [1,H,W], opacity/visibility
        "depth": depth,                          # [1,H,W], projected surface depth
        "normal_render": normal_render,          # [3,H,W], representation/raster normal
        "normal_depth": normal_depth,            # [3,H,W], normal from depth or geometry
        "specular_rgb": specular_rgb,            # [3,H,W], specular branch or residual
        "roughness_or_confidence": confidence,   # [1,H,W], low value/high value convention documented
        "optional_reflection_dir": refl_dir,     # [3,H,W], optional diagnostic/encoding input
    }
```

### 字段 fallback

| 缺失字段 | 迁移建议 fallback | 风险 |
| --- | --- | --- |
| `specular_rgb` | 使用 view-dependent branch 输出；或 `rgb - diffuse_rgb`；或 learned specular residual。 | 必须确认 linear/sRGB 空间一致，否则 loss 会约束错误尺度。 |
| `roughness_map` | 使用 reflective mask、specular intensity、learned confidence、material classifier 或 uncertainty。 | 不能把 fallback 伪装成物理 roughness；阈值要重新校准。 |
| `alpha` | 使用 accumulated opacity、visibility、transmittance complement 或 valid depth mask。 | 半透明/体渲染中 alpha 与 surface visibility 不完全等价。 |
| `surf_depth` | 使用 z-buffer depth、expected depth、median depth 或 ray termination depth。 | expected vs median depth 的 occlusion 行为不同。 |
| `rend_normal` | 使用 primitive normal、surface normal、MLP normal 或 finite difference normal。 | 坐标系和朝向不一致会破坏 normal agreement。 |
| `surf_normal` | 从 depth map finite difference 得到；或用 geometry normal。 | noisy depth normal 会使 mask 过稀。 |
| `reflection_dir` | 可省略；如果 spec branch 需要，按 `reflect(-view_dir, normal)` 计算。 | view direction 和 normal 坐标系必须一致。 |

### 面向不同方法的接入要点

- 3DGS：需要额外输出 alpha、depth、normal；specular 可由颜色残差或新增 view-dependent branch 提供。
- 2DGS：通常更容易输出 surfel normal/depth，可直接迁移 normal agreement 和 confidence-aware TSDF。
- GaussianShader：可把 shader 的 specular 分支作为 `specular_rgb`，把 material roughness 或 learned confidence 作为 mask 权重。
- Deferred Reflection GS：可复用 deferred specular buffer；重点检查 reflection direction 和 depth projection convention。
- NeRF-like reflective reconstruction：可用 ray accumulated depth/opacity 和 specular head；但 pair projection 要处理 volumetric depth uncertainty。

## 9. 迁移步骤 Checklist

1. 暴露中间 buffers：至少 `rgb/alpha/depth/normal/specular/confidence`。
2. 实现 camera pair 选择：先用小 baseline 和角度阈值，再加入 overlap/depth-valid 过滤。
3. 实现 backproject/project/sample：统一 camera convention、NDC、`grid_sample align_corners`。
4. 实现 confidence mask：组合 visibility、depth consistency、normal agreement、reflective confidence。
5. 接入 scheduled loss：默认关闭，通过 lambda/start/every/max_angle/gamma 显式启用。
6. 添加 metrics：reflection consistency、reflective-region PSNR、full/reflective PSNR/SSIM/LPIPS。
7. 做 `base/rc/wo_ref/wo_conf/rough_only` ablation：把 RC loss、confidence weighting、roughness smoothness 的贡献拆开。
8. 写结果边界：明确 completed/incomplete/OOM/excluded scene，避免把局部结果写成全局结论。

## 10. 风险与常见坑

- target projection depth convention 不一致：source depth 反投影和 target projected depth 必须在同一相机/尺度约定下比较。
- normal 坐标系不一致：`rend_normal`、`surf_normal`、view direction 如果混用 world/view/camera 坐标，会导致 reflection direction 和 normal agreement 错误。
- specular 分支 linear/sRGB 混用：当前 `spec_light` 不是最终 sRGB composite，而 `pbr_rgb` 已做 `linear2srgb`；迁移时不要混用。
- roughness 不存在或不可解释：可以用 confidence fallback，但要标成迁移建议，不要声称是物理 roughness。
- pair camera 角度过大导致 false correspondence：当前默认最大角度 20 度，复杂遮挡/非凸物体应更保守。
- `grid_sample align_corners` 与 NDC 坐标：当前 `_sample_map` 使用 `align_corners=True`；换成 False 会改变坐标解释。
- 额外 render 导致显存/速度问题：RC 每次触发要多 render target view，应通过 `every`、batch size 或 gradient checkpoint 控制成本。
- reflective mask 过稀导致 loss 不稳定：如果 `weight.sum()==0` 当前返回 zero loss；迁移时应记录 valid pixel count，避免“loss 没报错但没训练信号”。
- 不要把诊断指标改善误写成 PSNR 必然改善：FD-P2-lite 支持 reflection consistency 改善，但 render quality 是 mixed/tradeoff。
- 不要把 Shiny Blender Real incomplete/OOM 写成完整结论：当前日志明确排除 Shiny Real 完整 claim。
- `wo_conf` 的含义要精确：当前 runner 通过 `gamma=0` 中和 roughness exponent，不是删除全部 confidence 权重。
- hard-coded training constants 要核查：当前 `train.py` 主路径中 normal regularization 使用硬编码 `lambda_normal=0.05`。

## 11. 可复制的最小伪代码

```python
class RCConfig:
    lambda_ref_consistency = 0.01
    start = 3000
    every = 4
    max_angle_deg = 20.0
    gamma = 2.0
    alpha_threshold = 0.2
    roughness_threshold = 0.6
    depth_tolerance = 0.02


def train_step(iteration, camera, camera_pool, model, optimizer, cfg):
    src = model.render(camera)
    pred = composite(src["rgb"], src["alpha"], background=1.0)
    loss = reconstruction_loss(pred, camera.gt_image)

    if "normal_render" in src and "normal_depth" in src:
        normal_agree = (src["normal_render"] * src["normal_depth"]).sum(0)
        loss = loss + 0.05 * (1.0 - normal_agree).mean()

    if should_use_roughness_tv(iteration, cfg) and "roughness_or_confidence" in src:
        loss = loss + cfg.lambda_roughness_tv * tv_loss(src["roughness_or_confidence"][None])

    if should_use_rc(iteration, cfg):
        pair = choose_pair_camera(camera_pool, camera, cfg.max_angle_deg)
        if pair is not None:
            tgt = model.render(pair)
            rc_loss = rc_loss_from_buffers(src, tgt, camera, pair, cfg)
            loss = loss + cfg.lambda_ref_consistency * rc_loss

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return loss


def rc_loss_from_buffers(src, tgt, src_cam, tgt_cam, cfg):
    points = backproject_depth(src_cam, src["depth"])
    grid, projected_depth, valid = project_points(tgt_cam, points)

    sampled_spec = sample_map(tgt["specular_rgb"], grid)
    sampled_alpha = sample_map(tgt["alpha"], grid)
    sampled_depth = sample_map(tgt["depth"], grid)

    src_spec = flatten_hw(src["specular_rgb"])
    src_alpha = flatten_hw(src["alpha"])
    src_depth = flatten_hw(src["depth"])
    src_conf = flatten_hw(src["roughness_or_confidence"])

    normal_agree = dot_hw(src["normal_render"], src["normal_depth"]).clamp(0, 1)
    depth_ok = (sampled_depth - projected_depth).abs() < cfg.depth_tolerance * projected_depth.abs().clamp_min(1e-6)

    reflective_conf = src_spec.mean(dim=-1).clamp(0, 1) * confidence_factor(src_conf, cfg.gamma)
    mask = (
        valid
        & depth_ok
        & (src_alpha[:, 0] > cfg.alpha_threshold)
        & (sampled_alpha[:, 0] > cfg.alpha_threshold)
        & reflective_region(src_conf, cfg.roughness_threshold)
        & (normal_agree > 0)
    )

    weight = mask.float() * src_alpha[:, 0] * sampled_alpha[:, 0] * normal_agree * reflective_conf
    if weight.sum() <= 1e-6:
        return src_spec.sum() * 0.0

    residual = (src_spec.detach() - sampled_spec).abs().mean(dim=-1)
    return (weight * residual).sum() / weight.sum().clamp_min(1e-6)
```

## 12. Codex 分析过程摘要

### 使用的 skills

- `using-superpowers`：用于确认当前 Codex skills 工作流，约束后续分析先读技能说明、再执行。
- `subagent-driven-development`：用于把只读分析拆成三个并行 explorer：renderer/training/arguments/representation、reflection/mesh/metrics/runner、plans/logs/results boundary。
- `executing-plans`：用于保持“先证据、再文档、再验证”的执行节奏；本任务不是执行已有代码实现计划，因此只采用其检查点风格。
- `verification-before-completion`：用于完成前运行用户指定的静态验证命令，并把失败原因作为事实记录。

### 已阅读/交叉核对的核心文件

- `README.md`
- `train.py`
- `arguments/__init__.py`
- `gaussian_renderer/__init__.py`
- `scene/gaussian_model.py`
- `utils/reflection_consistency.py`
- `utils/loss_utils.py`
- `utils/mesh_utils.py`
- `metrics/reflection_consistency_eval.py`
- `metrics/render_quality_eval.py`
- `scripts/run_rc_refgs_ablation_direct.py`

### 已阅读/交叉核对的计划与日志

- `docs/superpowers/plans/2026-05-16-rc-refgs-research-plan.md`
- `docs/superpowers/plans/2026-05-19-rc-refgs-full-implementation-and-experiment-roadmap.md`
- `docs/superpowers/logs/rc-refgs-current-main-and-ablation-analysis-2026-05-29.md`
- `docs/superpowers/logs/rc-refgs-fd-p2-lite-final-results-analysis-2026-06-01.md`
- `docs/superpowers/logs/rc-refgs-fd-p2-lite-final-main-summary-2026-06-01.csv`
- `docs/superpowers/logs/rc-refgs-fd-p2-lite-final-ablation-summary-2026-06-01.csv`
- `docs/superpowers/logs/rc-refgs-fd-p2-lite-final-tradeoff-summary-2026-06-01.csv`
- `docs/superpowers/figures/fd-p2-lite/table1_main_base_vs_rc_summary.csv`
- `docs/superpowers/figures/fd-p2-lite/table2_rc_win_counts_by_metric.csv`
- `docs/superpowers/figures/fd-p2-lite/table3_ablation_aggregate.csv`
- `docs/superpowers/figures/fd-p2-lite/table4_tradeoff_summary.csv`

### 代码证据 vs 迁移建议

- 代码证据：renderer buffers、training schedule、reflection consistency mask/weight/stop-gradient、roughness TV、confidence-aware TSDF、metrics 和 ablation runner 均来自仓库当前文件。
- 实验边界：FD-P2-lite/non-Shiny-Real 完成范围、RC consistency win counts、render-quality mixed/tradeoff 和 Shiny Real OOM/exclusion 均来自 `docs/superpowers/logs/` 与 `docs/superpowers/figures/fd-p2-lite/`。
- 迁移建议：通用 `render(view)->buffers` 接口、fallback 字段、lambda 起步范围、symmetric RC loss、overlap-aware pair selection 都是面向其他方法的工程建议，不是当前仓库已验证事实。

### 未能完成或不能确认的部分

- 未启动任何训练、评估或长时间实验；本任务只做代码理解、日志核对和文档生成。
- 未验证 Shiny Blender Real 完整结果；现有日志仍将其列为 OOM/incomplete/excluded 范围。
- 未确认 RC 对几何质量或 mesh quality 的独立提升；文档中只把 confidence-aware TSDF 写为工程增强。
- 未把迁移建议验证到 3DGS、2DGS、GaussianShader、Deferred Reflection GS 或 NeRF-like 方法；这些建议需要目标仓库内的 renderer/metric ablation 支持。
