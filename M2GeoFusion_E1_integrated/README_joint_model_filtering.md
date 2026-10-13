# M²GeoFusion：联合模型过滤实验

## 研究目的与边界

研究加入 LS 散斑输入后，Joint VGGT 模型相对于 LG-only 灰度模型产生了哪些三维几何差异，
再通过来源分类、语义区域限制和整体邻域支持过滤得到最终点云。
XD 是几何差分，不是已经验证正确的补点；模型重建误差、对齐残差也可能进入 XD。

本分支的新流程为：

```text
01 Grounded-SAM2 → LG target_mask.npy（可选膨胀 need_mask.npy）
02 VGGT → Gray-only [LG] 与 Joint [LG, LS]
03 同像素 LG 对应 → Robust Sim(3) → 整个 Joint 对齐
   → Gray/Joint 三维差分 XD → 分来源 → 仅对 LS XD 做 LG 语义 ROI 判断
   → Gray + LG XD + 通过 Mask 的 LS XD = Candidate
04 整个 Candidate 的 KNN/MAD 几何支持过滤 → Final
05 原始 Joint 诊断（包括 04_pointcloud/07_joint_demo.ply）
```

所有核心三维数据来自 `world_points_from_depth`（Depth Head + Camera Head + 反投影），
不用直接 Point Map Head。新 03 要求 NPZ 的 `camera_param_source=vggt_prediction`，
入口要求 `CAMERA_PARAM_MODE=0`；MATLAB 标定模式仍供旧实验使用，不混入新实验。
Track Head 可继续产生诊断图，但新 03/04 不读取其对应或置信度。
新算法只排除非有限 XYZ，不额外使用深度置信度筛选；07 的官方默认置信度筛选独立保留。

## 双分支与同像素对应

```python
Pg0 = gray["world_points_from_depth"][0]
Pgj = joint["world_points_from_depth"][0]
Psj = joint["world_points_from_depth"][1]
```

Gray-only 与 Joint 的第一张输入都是 LG。同一张 LG 的同一个预处理像素构成三维估计对应：

\[
P_G^0 \approx s R P_G^J+t.
\]

代码检查两幅 LG 点图尺寸相同，并检查保存的 `images[0]` 完全一致。
如果 Joint 的批量 padding 改变了 LG 网格，程序报错，不直接把不同网格按索引配对。
这只是同图像位置的重建对应，不意味着两次三维预测一定正确，也不意味着 LG 与 LS 有真实逐像素匹配。

## Robust Sim(3)

`common.robust_umeyama_similarity` 遵循行向量约定：

```python
dst ≈ scale * (src @ R.T) + t
```

1. 展平对应，删除任一侧含 NaN/Inf 的配对，要求至少 20 对。
2. 用 Umeyama 得到初始完整 Sim(3)，修正反射，要求 `scale > 0`、`det(R) > 0`。
3. 对全部有限对应计算残差 `e = ||dst - (s*(src@R.T)+t)||`。
4. 用 `median(e) + SIM3_MAD_FACTOR * 1.4826 * MAD(e)` 判定 inlier。
5. 用这些 inlier 重估 Sim(3)，默认执行 3 轮。各轮重新评估全部有限对应，允许点重新进入。

每次估计最多使用 `SIM3_MAX_POINTS` 个 inlier，抽样固定 `np.random.default_rng(0)`。
残差统计和 MAD 判定覆盖全部有限对应，抽样只限制拟合成本。
恰好拟合时附加机器精度量级容差，避免 MAD 接近零时数值舍入导致错误剔除。
共点、共线或不足 20 个 inlier 明确报错，不偷偷切换为其他算法。

同一个变换各应用一次：

```python
aligned_Pgj = apply_similarity(Pgj, scale, R, t)
aligned_Psj = apply_similarity(Psj, scale, R, t)
```

不再额外乘 scale，不单独配准 LS，不做 ICP。它只把 Joint 世界变到 Gray-only 世界，
保留 Joint 两帧内部的相对几何关系。

`sim3_transform.npz` 保存 `scale/R/t/inlier_mask/pixel_grid_shape`。
`inlier_mask` 与原始 LG 点图展平索引等长；无效配对为 False，True 表示最后一次拟合前
MAD 选出的 inlier（包含未被拟合抽样选中的 inlier）。
`sim3_report.json` 保存对应数、inlier 数/比例、scale、R、角度、t、平移模长，
对齐前 median，以及对齐后 median/mean/P90/max、inlier median 和逐轮记录。
总体残差统计包含被拒绝的有限对应，避免只报告 inlier 而掩盖模型形变。

## XD 三维差分与来源

Gray anchor 为全部有限的原始 `Pg0`。对它建立 `cKDTree`，计算自身最近其他点距离，
取有限正距离的中位数 `d_gray`；零距离重复采样不用于尺度估计。

\[
\tau_{diff}=\text{DIFF_FACTOR}\,d_{gray},\qquad
X_D=\{q\in\tilde P_J:d(q,P_G^0)>\tau_{diff}\}.
\]

`d(q,P_G^0)` 是 Joint 对齐点到 Gray anchor 的最近欧氏距离。
同时计算 Joint-LG 与 Joint-LS，保留以下来源标签：

| 标签 | 来源 | 诊断颜色 |
|---|---|---|
| 0 | Gray anchor | 灰色 (160,160,160) |
| 1 | XD Joint-LG | 蓝色 (0,0,255) |
| 2 | XD Joint-LS | 红色 (255,0,0) |

`difference_report.json` 包括所有有效点数、采样间距、差分阈值、XD 各来源数量、
LS Mask 保留/删除/无效投影数以及 Candidate 数量。
还记录 `diff_factor=1.0/1.5/2.0/3.0` 的 XD 数量敏感性，绝不影响选定参数的结果。

## 语义 Mask 与 Candidate

01 新增原始分辨率二值 `target_mask.npy`，保留原有 PNG、`need_mask.npy` 和 `overlay.jpg`。
默认选原始语义区域 target；`SEMANTIC_MASK_KIND="dilated"` 使用膨胀后的 need mask。

只有 LS 来源的 XD 接受 Mask 判断。将原始 Mask 按 Gray-only `[LG]` 的 resize/crop/pad
规则变换到 Gray 相机网格，并与 `Pg0` 尺寸校验。
使用 **Gray-only** `extrinsic[0]` / `intrinsic[0]` 投影已对齐的 LS XD：
相机 Z 必须大于零、投影有限、连续 uv 在图像内，最近像素取整后也在图像内，且 Mask 像素等于 1。
无效投影包含在 Mask 删除数量内，报告中另外计数；有效投影落在 ROI 外也单独计数。

该投影是在判定已有三维点是否落入 LG 语义 ROI；它不是 LG–LS 真实像素匹配。
不加入 LG depth consistency，不查询 Track 匹配。
LG 来源 XD 默认全部保留，`KEEP_XD_JOINT_LG=False` 可消融去掉这部分。

\[
P_{candidate}=P_G^0\cup X_D^G\cup X_{D,mask}^S.
\]

代码中的并集采用按来源顺序拼接，**不额外去重**。
`candidate_data.npz` 保存原始 RGB `colors`、`points`、`source_labels`，
以及 `gray_count/xd_lg_count/xd_ls_count`（实际进入 Candidate 的数量）。

## 整体几何过滤

\[
P_{final}=F_{shape}(P_{candidate}).
\]

04 只读取 `candidate_data.npz`。将 **整个 Candidate** 传入过滤函数；函数不接收来源标签，
Gray 不受保护，三类点共用邻居池、统计量和阈值。

每个点求 k 个非自身最近邻的平均距离 `mean_d`，然后在全体点上计算：

\[
m=median(\bar d),\quad \sigma_r=1.4826\,MAD(\bar d),\quad
\tau_{shape}=m+\text{SHAPE_MAD_FACTOR}\,\sigma_r.
\]

删除 `mean_d > tau_shape` 的点，保留等于阈值的点。
查询按 `KNN_CHUNK_SIZE` 分块，避免同时保存完整 N×K 距离/索引数组。
若开启多轮，每轮用当前保留的整体点云重新建树、计算共同阈值，索引始终映射回原 Candidate。
不足 k+1 点时使用 `min(k,N-1)` 个邻居；不足两个点时保留，分数记零并写明原因。
MAD 为零时直接使用 median 阈值，不额外设计百分位过滤。

当前 E1 使用 robust local-neighborhood support 近似主体几何一致性，后续可替换为管道圆柱、
曲面或学习式主体模型。它不强制拟合平面或圆柱，也不保证所有稀疏真实结构都会保留。

`shape_filter_report.json` 给出总数前/后/删除比例、k、median/MAD/sigma/阈值、
逐轮记录和三类来源的 before/removed/after 数量。
顶层距离统计对应最后执行的一轮；默认只有一轮。
`shape_filter_score.npy` 与原 Candidate 顺序一致，多轮时为各点最后一次被评估的分数。
`final_data.npz` 保留最终点、原 RGB、来源标签、原 Candidate 长度的 `candidate_keep_mask`
及 `removed_iteration`（0 表示保留），可恢复各点去留。

## 输出目录与所有 PLY

默认输出根目录独立于旧实验：`outputs/joint_model_filtering/test002/`。
其中实验名取自 `EXPERIMENT_NAME`，更换实验时修改入口设置。

| 目录/文件 | 含义和颜色 |
|---|---|
| 03_joint_filter/01_gray_anchor.ply | Gray-only 原始有限主干，输入 RGB |
| 03_joint_filter/02_joint_aligned_rgb.ply | 完整对齐 Joint，两帧输入 RGB |
| 03_joint_filter/03_joint_aligned_source_color.ply | 完整对齐 Joint，LG 蓝、LS 红 |
| 03_joint_filter/04_xd_all.ply | 全部几何差分 XD，按来源蓝/红 |
| 03_joint_filter/05_xd_joint_lg.ply | LG 来源 XD，蓝色，消融关闭时仍保存以供比较 |
| 03_joint_filter/06_xd_joint_ls_before_mask.ply | LS 来源 XD，Mask 前，红色 |
| 03_joint_filter/07_xd_joint_ls_after_mask.ply | LS XD 通过 Mask，输入 RGB |
| 03_joint_filter/08_xd_joint_ls_removed_by_mask.ply | LS XD 被 Mask 删除，含无效投影，输入 RGB |
| 03_joint_filter/09_candidate_before_shape_filter.ply | 完整 Candidate，Gray 灰/LG 蓝/LS 红 |
| 04_global_filter/01_candidate_before_filter.ply | 与 09 相同点及顺序，使用输入 RGB |
| 04_global_filter/02_final_filtered.ply | 整体几何过滤最终结果，输入 RGB |
| 04_global_filter/03_removed_outliers.ply | 从全部 Candidate 删除的点，输入 RGB |
| 04_global_filter/04_final_source_color.ply | 最终点，来源灰/蓝/红 |
| 04_pointcloud/07_joint_demo.ply | **原始未对齐 Joint 世界**的官方 Demo 对照，输入 RGB |
| 02_vggt/joint_view/01_joint_LG.ply | 旧诊断：原始 Joint-LG，输入 RGB |
| 02_vggt/joint_view/02_joint_LS.ply | 旧诊断：原始 Joint-LS，输入 RGB |
| 02_vggt/joint_view/03_joint_LG_gray_LS_red.ply | 旧诊断：原始 Joint，LG 灰/LS 红 |

`07_joint_demo.ply` 的函数从旧 04 **原样**提取到 `joint_demo_export.py`。
旧 04 继续导入调用；新流程由 05 调用，输出位置仍是 `04_pointcloud/07_joint_demo.ply`。
它读取 `world_points_from_depth/depth_conf/images`，对两帧联合展平的置信度求全局第 50 百分位，
保留 `conf >= percentile` 且 `conf > 1e-5`，排除非有限 XYZ。颜色转换及写出逻辑不变。
没有 Sim(3)、Mask、差分或整体过滤，不参与 Candidate 构建。
这是 `03_joint_filter/07_xd_joint_ls_after_mask.ply` 之外的另一个独立文件。

旧 `03_select_ls_points.py` 与 `04_merge_and_export.py` 保留。
只有显式开启 `RUN_LEGACY_POINT_COMPLETION` 后才按原 `RUN_SELECTION/RUN_MERGE` 开关调用它们；
旧 01～06 PLY 的计算与导出逻辑未改。

## 参数与运行

入口 `demo_all.py` 集中提供实验设置，`config.py` 提供单脚本默认值：

| 参数 | 默认值 |
|---|---|
| CAMERA_PARAM_MODE | 0，Camera Head |
| VGGT_PREPROCESS_MODE | crop |
| DIFF_FACTOR | 1.5 |
| SIM3_MAD_FACTOR | 3.5 |
| SIM3_ITERATIONS | 3 |
| SIM3_MAX_POINTS | 100000 |
| SEMANTIC_MASK_KIND | target（另可选 dilated） |
| KEEP_XD_JOINT_LG | True |
| SHAPE_K | 16 |
| SHAPE_MAD_FACTOR | 3.5 |
| SHAPE_FILTER_ITERATIONS | 1 |
| KNN_CHUNK_SIZE | 20000 |
| RUN_MASK / RUN_VGGT / RUN_JOINT_EXPORT | True |
| RUN_JOINT_DIFFERENCE_FILTER / RUN_GLOBAL_SHAPE_FILTER | True |
| RUN_LEGACY_POINT_COMPLETION | False |

语义检测沿用 `TEXT_PROMPT="wire harness. pipe."`、box threshold 0.35、text threshold 0.25；
dilated 消融沿用 `DILATE_RADIUS=20`。Track 配置只影响诊断。
入口会记录实验参数到 `experiment_manifest.json`。

```bash
conda activate grounded_sam2
cd '/home/wqy/workspace/M2GeoFusion_E1_integrated(1)/M2GeoFusion_E1_integrated'
python demo_all.py
```

入口用现有 conda 调用 Grounded-SAM2 和 VGGT 各自的环境；用户无需逐阶段传路径。
若复用已有输出，需将 `OUTPUT_ROOT/EXPERIMENT_NAME` 指向匹配数据并关闭对应 RUN 开关；
旧 Mask 输出若缺 `target_mask.npy`，需运行新增保存该字段的 01。

静态与数学验证命令（不加载 GPU 模型）：

```bash
python -m py_compile *.py
python sanity_check_sim3.py
```

合成检查包含已知 s/R/t 的精确恢复、噪声加 15% 离群对应、NaN/Inf 索引处理和固定 seed 抽样复现。
代码修改阶段只执行这些检查，不自动运行完整实验、提交或推送 Git。
