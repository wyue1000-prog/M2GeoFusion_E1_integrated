from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from config import DEDUP_FACTOR
from common import ensure_dir, write_ascii_ply, save_json
from joint_demo_export import export_joint_demo_ply


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection-data", required=True)
    ap.add_argument("--joint-npz", required=True,
                    help="02's original [LG, LS] Joint inference NPZ, used only for 07_joint_demo.ply")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dedup-factor", type=float, default=DEDUP_FACTOR)
    args = ap.parse_args()

    out_dir = ensure_dir(args.out)
    data = np.load(args.selection_data)

    joint_lg_count, joint_ls_count, joint_total, joint_conf_threshold = export_joint_demo_ply(
        args.joint_npz,
        out_dir / "07_joint_demo.ply",
    )

    # This is the finite-point subset from 03's gray_only Pg0 map.  It is the
    # sole source for both 01_gray_anchor.ply and the gray prefix of merged PLYs.
    gray_anchor = np.asarray(data["anchor_points"], dtype=np.float32)
    anchor_colors = np.asarray(data["anchor_colors"], dtype=np.uint8)

    ls_all = np.asarray(data["ls_all_points"], dtype=np.float32)
    ls_all_colors = np.asarray(data["ls_all_colors"], dtype=np.uint8)

    selected_ls = np.asarray(data["ls_selected_points"], dtype=np.float32)
    ls_need_colors = np.asarray(data["ls_selected_colors"], dtype=np.uint8)
    camera_param_source = str(np.asarray(data["camera_param_source"]).item()) if "camera_param_source" in data else "unknown"
    camera_param_mode = int(np.asarray(data["camera_param_mode"]).item()) if "camera_param_mode" in data else None
    translation_scale = (float(np.asarray(data["translation_scale_physical_to_vggt"]).item())
                         if "translation_scale_physical_to_vggt" in data else None)
    if translation_scale is not None and not np.isfinite(translation_scale):
        translation_scale = None

    if len(gray_anchor) < 2:
        raise RuntimeError("Anchor cloud is too small.")
    if len(selected_ls) == 0:
        raise RuntimeError(
            "No LS candidate passed the Need mask. "
            "请检查 need_mask / 投影关系 / prompt。"
        )

    # ------------------------------------------------------------
    # 1. Gray anchor 自己的典型点间距
    # ------------------------------------------------------------
    tree = cKDTree(gray_anchor)

    # 对全部 anchor 求 k=2；第 0 个邻居是自己，第 1 个才是最近其他点
    d2, _ = tree.query(gray_anchor, k=2, workers=-1)
    nn_spacing = d2[:, 1]
    nn_spacing = nn_spacing[np.isfinite(nn_spacing) & (nn_spacing > 0)]

    if len(nn_spacing) == 0:
        raise RuntimeError("Cannot estimate anchor nearest-neighbor spacing.")

    median_nn = float(np.median(nn_spacing))
    tau_dup = float(args.dedup_factor * median_nn)

    # ------------------------------------------------------------
    # 2. Need-selected LS 到 Gray Anchor 的最近距离
    # ------------------------------------------------------------
    dist_to_gray, _ = tree.query(selected_ls, k=1, workers=-1)
    finite_dist = dist_to_gray[np.isfinite(dist_to_gray)]
    dist_min = float(np.min(finite_dist)) if len(finite_dist) else float("nan")
    dist_median = float(np.median(finite_dist)) if len(finite_dist) else float("nan")
    dist_p90 = float(np.percentile(finite_dist, 90)) if len(finite_dist) else float("nan")
    dist_max = float(np.max(finite_dist)) if len(finite_dist) else float("nan")

    add_mask = np.isfinite(dist_to_gray) & (dist_to_gray > tau_dup)
    ls_add = selected_ls[add_mask]
    ls_add_colors = ls_need_colors[add_mask]

    # ------------------------------------------------------------
    # 3. 输出点云
    # ------------------------------------------------------------
    gray_color_for_vis = np.tile(
        np.array([[160, 160, 160]], dtype=np.uint8),
        (len(gray_anchor), 1),
    )
    red_need = np.tile(
        np.array([[255, 0, 0]], dtype=np.uint8),
        (len(selected_ls), 1),
    )
    red_add = np.tile(
        np.array([[255, 0, 0]], dtype=np.uint8),
        (len(ls_add), 1),
    )

    write_ascii_ply(
        out_dir / "01_gray_anchor.ply",
        gray_anchor,
        anchor_colors,
    )
    write_ascii_ply(
        out_dir / "02_all_ls_candidates.ply",
        ls_all,
        ls_all_colors,
    )
    write_ascii_ply(
        out_dir / "03_ls_need_candidates.ply",
        selected_ls,
        red_need,
    )

    # Raw: gray XYZ is copied verbatim before the selected LS suffix.
    merged_xyz = np.concatenate([gray_anchor, selected_ls], axis=0)
    assert np.allclose(
        merged_xyz[:len(gray_anchor)],
        gray_anchor,
        atol=1e-8,
    )
    gray_consistency_error = np.linalg.norm(
        merged_xyz[:len(gray_anchor)] - gray_anchor,
        axis=1,
    )
    gray_consistency_max_error = float(np.max(gray_consistency_error))
    gray_consistency_mean_error = float(np.mean(gray_consistency_error))
    gray_consistency_median_error = float(np.median(gray_consistency_error))

    raw_colors = np.concatenate([gray_color_for_vis, red_need], axis=0)
    write_ascii_ply(
        out_dir / "04_gray_plus_ls_raw.ply",
        merged_xyz,
        raw_colors,
    )

    # De-dup: gray 全保留 + 真正离开已有 gray 采样点一定距离的 LS
    dedup_points = np.concatenate([gray_anchor, ls_add], axis=0)
    dedup_colors = np.concatenate([gray_color_for_vis, red_add], axis=0)
    write_ascii_ply(
        out_dir / "05_gray_plus_ls_dedup.ply",
        dedup_points,
        dedup_colors,
    )

    # 单独保存真正被加进去的点
    write_ascii_ply(
        out_dir / "06_true_added_ls_points.ply",
        ls_add,
        red_add,
    )

    np.save(out_dir / "ls_need_distance_to_gray.npy", dist_to_gray.astype(np.float32))

    save_json(out_dir / "merge_report.json", {
        "camera_param_source": camera_param_source,
        "camera_param_mode": camera_param_mode,
        "translation_scale_physical_to_vggt": translation_scale,
        "anchor_count": int(len(gray_anchor)),
        "all_ls_valid_count": int(len(ls_all)),
        "need_selected_ls_count": int(len(selected_ls)),
        "median_gray_nn_spacing": median_nn,
        "dedup_factor": float(args.dedup_factor),
        "tau_dup": tau_dup,
        "true_added_ls_count": int(len(ls_add)),
        "true_added_ratio_of_need_selected": float(len(ls_add) / max(len(selected_ls), 1)),
        "raw_merged_count": int(len(merged_xyz)),
        "dedup_merged_count": int(len(dedup_points)),
        "gray_consistency_max_3d_error": gray_consistency_max_error,
        "gray_consistency_mean_3d_error": gray_consistency_mean_error,
        "gray_consistency_median_3d_error": gray_consistency_median_error,
        "ls_to_gray_min_distance": dist_min,
        "ls_to_gray_median_distance": dist_median,
        "ls_to_gray_p90_distance": dist_p90,
        "ls_to_gray_max_distance": dist_max,
    })

    print("[E1-04] --------------------------------------")
    print(f"[E1-04] Gray anchor              : {len(gray_anchor)}")
    print(f"[E1-04] Need-selected LS         : {len(selected_ls)}")
    print(f"[E1-04] Gray consistency max 3D error    : {gray_consistency_max_error:.8e}")
    print(f"[E1-04] Gray consistency mean 3D error   : {gray_consistency_mean_error:.8e}")
    print(f"[E1-04] Gray consistency median 3D error : {gray_consistency_median_error:.8e}")
    print(f"[E1-04] median gray NN spacing   : {median_nn:.8f}")
    print(f"[E1-04] tau_dup                  : {tau_dup:.8f}")
    print(f"[E1-04] LS->Gray min distance    : {dist_min:.8f}")
    print(f"[E1-04] LS->Gray median distance : {dist_median:.8f}")
    print(f"[E1-04] LS->Gray P90 distance    : {dist_p90:.8f}")
    print(f"[E1-04] LS->Gray max distance    : {dist_max:.8f}")
    print(f"[E1-04] true added LS            : {len(ls_add)}")
    print(f"[E1-04] true-added ratio         : {len(ls_add)/max(len(selected_ls),1):.4f}")
    print("[E1-04] 07 joint demo source      : world_points_from_depth")
    print("[E1-04] 07 confidence filtering   : depth_conf global 50th percentile "
          f"({joint_conf_threshold:.8g}), conf > 1e-5")
    print(f"[E1-04] 07 Joint-LG points       : {joint_lg_count}")
    print(f"[E1-04] 07 Joint-LS points       : {joint_ls_count}")
    print(f"[E1-04] 07 total points          : {joint_total}")
    print(f"[E1-04] saved 07                 : {out_dir / '07_joint_demo.ply'}")
    print(f"[E1-04] output dir               : {out_dir}")


if __name__ == "__main__":
    main()
