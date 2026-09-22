from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from config import ALIGNMENT_MODE, VGGT_PREPROCESS_MODE
from common import (
    ensure_dir,
    preprocess_mask_like_vggt,
    project_world_to_image,
    estimate_scale_from_local_edges,
    umeyama_similarity,
    apply_similarity,
    image_tensor_to_rgb_uint8,
    write_ascii_ply,
    save_json,
)


def get_world(npz):
    if "world_points_from_depth" not in npz:
        raise KeyError("NPZ missing 'world_points_from_depth'")
    return np.asarray(npz["world_points_from_depth"])


def get_camera_param_source(npz) -> str:
    if "camera_param_source" not in npz:
        raise KeyError("NPZ missing 'camera_param_source'; rerun 02_run_vggt.py with the same camera mode.")
    value = np.asarray(npz["camera_param_source"])
    return str(value.item() if value.ndim == 0 else value.reshape(-1)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lg", required=True, help="原始 LG 路径")
    ap.add_argument("--ls", required=True, help="原始 LS 路径")
    ap.add_argument("--calib", required=False, default=None, help="Deprecated alias for --calib-mat.")
    ap.add_argument("--calib-mat", required=False, default=None)
    ap.add_argument("--camera-param-mode", type=int, choices=[0, 1], default=0)
    ap.add_argument("--need-mask", required=True, help="原始分辨率 need_mask.npy")
    ap.add_argument("--gray-npz", required=True)
    ap.add_argument("--joint-npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--alignment",
        default=ALIGNMENT_MODE,
        choices=["scale", "sim3"],
    )
    ap.add_argument(
        "--preprocess-mode",
        default=VGGT_PREPROCESS_MODE,
        choices=["crop", "pad"],
    )
    args = ap.parse_args()

    lg = str(Path(args.lg).expanduser().resolve())
    ls = str(Path(args.ls).expanduser().resolve())
    out_dir = ensure_dir(args.out)

    need_original = np.load(args.need_mask)
    gray = np.load(args.gray_npz)
    joint = np.load(args.joint_npz)
    source_gray = get_camera_param_source(gray)
    source_joint = get_camera_param_source(joint)
    translation_scale = (float(np.asarray(joint["translation_scale_physical_to_vggt"]).item())
                         if "translation_scale_physical_to_vggt" in joint else None)
    expected_source = "calibration" if args.camera_param_mode == 1 else "vggt_prediction"
    if source_gray != expected_source or source_joint != expected_source:
        raise RuntimeError(
            f"Camera parameter source mismatch: requested {expected_source}, "
            f"gray={source_gray}, joint={source_joint}. Rerun stage 02 with the same mode."
        )
    print(f"[E1-03] camera parameter source = {source_joint}")
    if args.camera_param_mode == 1:
        print("[E1-03] projection uses calibrated LG K + calibrated LG world frame")

    Pg0 = get_world(gray)[0]      # H,W,3
    Pgj = get_world(joint)[0]     # joint LG
    Psj = get_world(joint)[1]     # joint LS

    if Pg0.shape != Pgj.shape:
        raise RuntimeError(
            "LG-only 与 Joint-LG 的点图尺寸不同。"
            f" gray={Pg0.shape}, joint={Pgj.shape}. "
            "第一轮 E1 请优先确保 LG/LS 原图分辨率与宽高比一致。"
        )

    h, w = Pgj.shape[:2]

    need_vggt = preprocess_mask_like_vggt(
        need_original,
        image_paths=[lg, ls],
        target_index=0,
        mode=args.preprocess_mode,
    )

    if need_vggt.shape != (h, w):
        raise RuntimeError(
            f"Need mask shape {need_vggt.shape} != VGGT pointmap {(h, w)}"
        )

    cv2.imwrite(
        str(out_dir / "need_mask_vggt.png"),
        need_vggt.astype(np.uint8) * 255,
    )

    # ------------------------------------------------------------
    # A. 两次独立 forward 对齐
    # ------------------------------------------------------------
    if args.alignment == "scale":
        scale = estimate_scale_from_local_edges(Pg0, Pgj)
        R_align = np.eye(3, dtype=np.float64)
        t_align = np.zeros(3, dtype=np.float64)

        aligned_Psj = apply_similarity(Psj, scale)
        aligned_Pgj = apply_similarity(Pgj, scale)

    else:
        src = Pgj.reshape(-1, 3)
        dst = Pg0.reshape(-1, 3)
        scale, R_align, t_align = umeyama_similarity(src, dst)
        aligned_Psj = apply_similarity(Psj, scale, R_align, t_align)
        aligned_Pgj = apply_similarity(Pgj, scale, R_align, t_align)

    print(f"[E1-03] alignment = {args.alignment}")
    print(f"[E1-03] scale = {scale:.8f}")

    # Diagnostic: same LG image reconstructed in two independent forwards.
    align_error = np.linalg.norm(
        aligned_Pgj.reshape(-1, 3) - Pg0.reshape(-1, 3), axis=1
    )
    align_error = align_error[np.isfinite(align_error)]
    align_median = float(np.median(align_error)) if len(align_error) else float("nan")
    align_mean = float(np.mean(align_error)) if len(align_error) else float("nan")
    align_p90 = float(np.percentile(align_error, 90)) if len(align_error) else float("nan")
    print(f"[E1-03] LG alignment median error = {align_median:.8f}")
    print(f"[E1-03] LG alignment mean error   = {align_mean:.8f}")
    print(f"[E1-03] LG alignment P90 error    = {align_p90:.8f}")

    # ------------------------------------------------------------
    # B. 注意：像素投影要在 Joint 自己的几何坐标里做，
    #    不能先只缩放 P 再拿原 joint extrinsic 投影。
    # ------------------------------------------------------------
    E_lg_joint = np.asarray(joint["extrinsic"])[0]
    K_lg_joint = np.asarray(joint["intrinsic"])[0]

    Psj_flat = Psj.reshape(-1, 3)
    uv, z_cam, _ = project_world_to_image(
        Psj_flat,
        E_lg_joint,
        K_lg_joint,
    )

    valid = (
        np.isfinite(Psj_flat).all(axis=1) &
        np.isfinite(uv).all(axis=1) &
        (z_cam > 1e-9)
    )

    u = np.rint(uv[:, 0]).astype(np.int64, copy=False)
    v = np.rint(uv[:, 1]).astype(np.int64, copy=False)

    inside = (
        valid &
        (u >= 0) & (u < w) &
        (v >= 0) & (v < h)
    )

    selected = np.zeros(len(Psj_flat), dtype=bool)
    idx_inside = np.where(inside)[0]
    selected[idx_inside] = need_vggt[v[idx_inside], u[idx_inside]] > 0

    selection_mask_ls = selected.reshape(Psj.shape[:2])
    cv2.imwrite(
        str(out_dir / "selection_mask_ls.png"),
        selection_mask_ls.astype(np.uint8) * 255,
    )
    np.save(
        out_dir / "selection_mask_ls.npy",
        selection_mask_ls.astype(np.uint8),
    )

    # ------------------------------------------------------------
    # C. 颜色
    # ------------------------------------------------------------
    gray_rgb = image_tensor_to_rgb_uint8(gray["images"][0])
    joint_lg_rgb = image_tensor_to_rgb_uint8(joint["images"][0])
    joint_ls_rgb = image_tensor_to_rgb_uint8(joint["images"][1])

    if gray_rgb.shape[:2] != Pg0.shape[:2]:
        raise RuntimeError("Gray image tensor and point map shape mismatch.")
    if joint_ls_rgb.shape[:2] != Psj.shape[:2]:
        raise RuntimeError("LS image tensor and point map shape mismatch.")

    anchor_flat = Pg0.reshape(-1, 3)
    anchor_color_flat = gray_rgb.reshape(-1, 3)

    aligned_ls_flat = aligned_Psj.reshape(-1, 3)
    ls_color_flat = joint_ls_rgb.reshape(-1, 3)

    anchor_valid = np.isfinite(anchor_flat).all(axis=1)
    ls_valid = np.isfinite(aligned_ls_flat).all(axis=1) & valid

    anchor_points = anchor_flat[anchor_valid].astype(np.float32)
    anchor_colors = anchor_color_flat[anchor_valid].astype(np.uint8)

    all_ls_points = aligned_ls_flat[ls_valid].astype(np.float32)
    all_ls_colors = ls_color_flat[ls_valid].astype(np.uint8)

    sel_valid = selected & np.isfinite(aligned_ls_flat).all(axis=1)
    selected_ls_points = aligned_ls_flat[sel_valid].astype(np.float32)
    selected_ls_colors = ls_color_flat[sel_valid].astype(np.uint8)

    # ------------------------------------------------------------
    # D. 投影诊断图：在 VGGT 预处理后的 LG 上画点
    # ------------------------------------------------------------
    overlay = cv2.cvtColor(joint_lg_rgb, cv2.COLOR_RGB2BGR)
    draw_idx = np.where(selected)[0]
    if len(draw_idx) > 15000:
        rng = np.random.default_rng(0)
        draw_idx = rng.choice(draw_idx, 15000, replace=False)

    for idx in draw_idx:
        cv2.circle(
            overlay,
            (int(u[idx]), int(v[idx])),
            1,
            (0, 0, 255),
            -1,
        )

    cv2.imwrite(str(out_dir / "projected_selected_points.png"), overlay)

    # 诊断 PLY
    write_ascii_ply(
        out_dir / "ls_all_candidates_aligned.ply",
        all_ls_points,
        all_ls_colors,
    )

    red = np.tile(np.array([[255, 0, 0]], dtype=np.uint8),
                  (len(selected_ls_points), 1))
    write_ascii_ply(
        out_dir / "ls_need_candidates_red.ply",
        selected_ls_points,
        red,
    )

    np.savez_compressed(
        out_dir / "selection_data.npz",
        anchor_points=anchor_points,
        anchor_colors=anchor_colors,
        ls_all_points=all_ls_points,
        ls_all_colors=all_ls_colors,
        ls_selected_points=selected_ls_points,
        ls_selected_colors=selected_ls_colors,
        camera_param_source=np.asarray(source_joint),
        camera_param_mode=np.asarray(args.camera_param_mode, dtype=np.int32),
        translation_scale_physical_to_vggt=np.asarray(translation_scale if translation_scale is not None else np.nan),
    )

    calib_arg = args.calib_mat or args.calib
    calib_path = None
    if calib_arg:
        calib_path = str(Path(calib_arg).expanduser().resolve())
        if not Path(calib_path).exists():
            raise FileNotFoundError(calib_path)

    save_json(out_dir / "selection_report.json", {
        "lg_path": lg,
        "ls_path": ls,
        "calibration_file": calib_path,
        "camera_param_mode": int(args.camera_param_mode),
        "camera_param_source": source_joint,
        "translation_scale_physical_to_vggt": translation_scale,
        "calibration_used_in_E1_core": bool(args.camera_param_mode == 1),
        "calibration_note": ("LS points are projected through calibrated LG K and calibrated LG-world frame."
                             if args.camera_param_mode == 1 else
                             "Legacy mode: Joint VGGT predicted camera parameters are used."),
        "alignment_mode": args.alignment,
        "scale": float(scale),
        "R_align": np.asarray(R_align).tolist(),
        "t_align": np.asarray(t_align).tolist(),
        "vggt_shape_hw": [int(h), int(w)],
        "all_ls_valid_count": int(ls_valid.sum()),
        "selected_ls_count": int(sel_valid.sum()),
        "selected_ratio_of_valid_ls": float(sel_valid.sum() / max(ls_valid.sum(), 1)),
        "lg_alignment_median_error": align_median,
        "lg_alignment_mean_error": align_mean,
        "lg_alignment_p90_error": align_p90,
    })

    print(f"[E1-03] valid LS candidates = {int(ls_valid.sum())}")
    print(f"[E1-03] selected by Need    = {int(sel_valid.sum())}")
    print(f"[E1-03] saved to: {out_dir}")


if __name__ == "__main__":
    main()
