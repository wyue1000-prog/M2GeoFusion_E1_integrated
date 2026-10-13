"""Joint-world -> Gray-world Sim3, geometric difference, and LS-only semantic ROI."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from config import (DIFF_FACTOR, SIM3_MAD_FACTOR, SIM3_ITERATIONS, SIM3_MAX_POINTS,
                    KEEP_XD_JOINT_LG, VGGT_PREPROCESS_MODE)
from common import (ensure_dir, save_json, write_ascii_ply, image_tensor_to_rgb_uint8,
                    preprocess_mask_like_vggt, robust_umeyama_similarity,
                    apply_similarity, provenance_colors)


def semantic_roi_membership(points, extrinsic, intrinsic, mask):
    """Test existing Gray-world 3D points against LG ROI; NOT cross-modal matching.

    No depth consistency test. Reject non-positive Z, non-finite projection,
    continuous coordinates outside the image and rounded pixels outside it.
    """
    E, K = np.asarray(extrinsic), np.asarray(intrinsic)
    if E.shape != (3, 4) or K.shape != (3, 3) or not np.isfinite(E).all() or not np.isfinite(K).all():
        raise ValueError("Invalid Gray-only Camera Head matrices.")
    camera = np.asarray(points, dtype=np.float64) @ E[:, :3].T + E[:, 3]
    projected = camera @ K.T
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        uv = projected[:, :2] / projected[:, 2:3]
    h, w = mask.shape
    valid = (np.isfinite(uv).all(axis=1) & np.isfinite(camera).all(axis=1)
             & (camera[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < w)
             & (uv[:, 1] >= 0) & (uv[:, 1] < h))
    ids = np.flatnonzero(valid)
    pixels = np.rint(uv[ids]).astype(np.int64)
    inside = (pixels[:, 0] < w) & (pixels[:, 1] < h)
    valid[ids[~inside]] = False
    ids, pixels = ids[inside], pixels[inside]
    keep = np.zeros(len(points), dtype=bool)
    keep[ids] = mask[pixels[:, 1], pixels[:, 0]] == 1
    return keep, valid


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for flag in ("lg", "ls", "semantic-mask", "gray-npz", "joint-npz", "out"):
        ap.add_argument("--" + flag, required=True)
    ap.add_argument("--preprocess-mode", choices=["crop", "pad"], default=VGGT_PREPROCESS_MODE)
    ap.add_argument("--diff-factor", type=float, default=DIFF_FACTOR)
    ap.add_argument("--sim3-mad-factor", type=float, default=SIM3_MAD_FACTOR)
    ap.add_argument("--sim3-iterations", type=int, default=SIM3_ITERATIONS)
    ap.add_argument("--sim3-max-points", type=int, default=SIM3_MAX_POINTS)
    ap.add_argument("--keep-xd-joint-lg", type=int, choices=[0, 1], default=int(KEEP_XD_JOINT_LG))
    args = ap.parse_args()
    if not np.isfinite(args.diff_factor) or args.diff_factor <= 0:
        raise ValueError("diff-factor must be finite and positive.")

    with np.load(args.gray_npz) as gray, np.load(args.joint_npz) as joint:
        for name, prediction, count in (("Gray-only", gray, 1), ("Joint", joint, 2)):
            if "camera_param_source" not in prediction or str(prediction["camera_param_source"].item()) != "vggt_prediction":
                raise ValueError(f"{name} must use Camera Head predictions (CAMERA_PARAM_MODE=0).")
            points = prediction["world_points_from_depth"]
            images = prediction["images"]
            if points.ndim != 4 or points.shape[0] != count or points.shape[-1] != 3:
                raise ValueError(f"Unexpected {name} depth point-map shape: {points.shape}")
            if images.shape != (count, 3, *points.shape[1:3]):
                raise ValueError(f"{name} images must match the saved depth point-map grid.")
        Pg0 = gray["world_points_from_depth"][0]
        Pgj, Psj = joint["world_points_from_depth"]
        if Pg0.shape != Pgj.shape or not np.array_equal(gray["images"][0], joint["images"][0]):
            raise ValueError("Gray-only and Joint-LG must contain exactly the same preprocessed LG grid; check inputs/crop/padding.")
        rgb0 = image_tensor_to_rgb_uint8(gray["images"][0]).reshape(-1, 3)
        rgbg = image_tensor_to_rgb_uint8(joint["images"][0]).reshape(-1, 3)
        rgbs = image_tensor_to_rgb_uint8(joint["images"][1]).reshape(-1, 3)
        E, K = gray["extrinsic"][0], gray["intrinsic"][0]

    # Projection uses the Gray-only grid, so preprocess the mask with [LG].
    raw_mask = np.load(args.semantic_mask)
    with Image.open(args.lg) as image:
        if raw_mask.shape != (image.height, image.width):
            raise ValueError("Semantic mask must have the original LG image resolution.")
    if not Path(args.ls).is_file():
        raise FileNotFoundError(args.ls)
    mask = preprocess_mask_like_vggt(raw_mask, [args.lg], mode=args.preprocess_mode)
    if mask.shape != Pg0.shape[:2]:
        raise ValueError("Semantic mask does not match the Gray-only camera pixel grid.")

    scale, R, t, inliers, sim3_report = robust_umeyama_similarity(
        Pgj, Pg0, args.sim3_mad_factor, args.sim3_iterations, args.sim3_max_points)
    # Exactly ONE shared Sim3; no subsequent scale, ICP, or per-view alignment.
    aligned_g = apply_similarity(Pgj, scale, R, t).reshape(-1, 3)
    aligned_s = apply_similarity(Psj, scale, R, t).reshape(-1, 3)
    anchor = Pg0.reshape(-1, 3)
    valid0 = np.isfinite(anchor).all(axis=1)
    validg = np.isfinite(aligned_g).all(axis=1)
    valids = np.isfinite(aligned_s).all(axis=1)
    anchor, rgb0 = anchor[valid0], rgb0[valid0]
    aligned_g, rgbg = aligned_g[validg], rgbg[validg]
    aligned_s, rgbs = aligned_s[valids], rgbs[valids]
    tree = cKDTree(anchor)
    nn = tree.query(anchor, k=2, workers=-1)[0][:, 1]
    nn = nn[np.isfinite(nn) & (nn > 0)]
    if not len(nn):
        raise ValueError("Cannot estimate a positive Gray anchor nearest-neighbor spacing.")
    spacing = float(np.median(nn))
    tau = args.diff_factor * spacing
    dg = tree.query(aligned_g, k=1, workers=-1)[0]
    ds = tree.query(aligned_s, k=1, workers=-1)[0]
    xd_g, xd_s = dg > tau, ds > tau
    xg, xs = aligned_g[xd_g], aligned_s[xd_s]
    cg, cs = rgbg[xd_g], rgbs[xd_s]
    mask_keep, projection_valid = semantic_roi_membership(xs, E, K, mask)
    kept_g = xg if args.keep_xd_joint_lg else xg[:0]
    kept_cg = cg if args.keep_xd_joint_lg else cg[:0]
    candidate = np.concatenate([anchor, kept_g, xs[mask_keep]])
    colors = np.concatenate([rgb0, kept_cg, cs[mask_keep]])
    labels = np.concatenate([np.zeros(len(anchor), np.uint8), np.ones(len(kept_g), np.uint8),
                             np.full(int(mask_keep.sum()), 2, np.uint8)])
    out = ensure_dir(args.out)
    np.savez_compressed(out / "sim3_transform.npz", scale=scale, R=R, t=t,
                        inlier_mask=inliers, pixel_grid_shape=Pg0.shape[:2])
    save_json(out / "sim3_report.json", sim3_report)
    np.save(out / "semantic_mask_vggt.npy", mask)
    np.savez_compressed(out / "candidate_data.npz", points=candidate, colors=colors,
                        source_labels=labels, gray_count=len(anchor), xd_lg_count=len(kept_g),
                        xd_ls_count=int(mask_keep.sum()))
    joint_points = np.concatenate([aligned_g, aligned_s])
    joint_labels = np.concatenate([np.ones(len(aligned_g), np.uint8), np.full(len(aligned_s), 2, np.uint8)])
    xd_points = np.concatenate([xg, xs])
    xd_labels = np.concatenate([np.ones(len(xg), np.uint8), np.full(len(xs), 2, np.uint8)])
    exports = [
        ("01_gray_anchor.ply", anchor, rgb0),
        ("02_joint_aligned_rgb.ply", joint_points, np.concatenate([rgbg, rgbs])),
        ("03_joint_aligned_source_color.ply", joint_points, provenance_colors(joint_labels)),
        ("04_xd_all.ply", xd_points, provenance_colors(xd_labels)),
        ("05_xd_joint_lg.ply", xg, provenance_colors(np.ones(len(xg), np.uint8))),
        ("06_xd_joint_ls_before_mask.ply", xs, provenance_colors(np.full(len(xs), 2, np.uint8))),
        ("07_xd_joint_ls_after_mask.ply", xs[mask_keep], cs[mask_keep]),
        ("08_xd_joint_ls_removed_by_mask.ply", xs[~mask_keep], cs[~mask_keep]),
        ("09_candidate_before_shape_filter.ply", candidate, provenance_colors(labels)),
    ]
    for name, xyz, rgb in exports:
        write_ascii_ply(out / name, xyz, rgb)
    report = {
        "point_source": "world_points_from_depth", "confidence_filtering": False,
        "gray_anchor_count": len(anchor), "joint_lg_valid_count": len(aligned_g),
        "joint_ls_valid_count": len(aligned_s), "median_gray_nn_spacing": spacing,
        "diff_factor": args.diff_factor, "tau_diff": tau, "xd_total_count": len(xg) + len(xs),
        "xd_joint_lg_count": len(xg), "xd_joint_ls_count": len(xs),
        "xd_joint_ls_mask_kept": int(mask_keep.sum()),
        "xd_joint_ls_mask_removed": int((~mask_keep).sum()),
        "xd_joint_ls_invalid_projection": int((~projection_valid).sum()),
        "xd_joint_ls_outside_semantic_roi": int((projection_valid & ~mask_keep).sum()),
        "keep_xd_joint_lg": bool(args.keep_xd_joint_lg), "candidate_xd_joint_lg_count": len(kept_g),
        "candidate_total_count": len(candidate), "semantic_mask": str(Path(args.semantic_mask).resolve()),
        "projection_camera": "Gray-only Camera Head; semantic ROI only, not cross-modal matching",
        "preprocess_mode": args.preprocess_mode,
        "threshold_sensitivity": {
            str(f): {"xd_joint_lg_count": int((dg > f * spacing).sum()),
                     "xd_joint_ls_count": int((ds > f * spacing).sum()),
                     "xd_total_count": int((dg > f * spacing).sum() + (ds > f * spacing).sum())}
            for f in (1.0, 1.5, 2.0, 3.0)},
        "note": "XD is geometric difference, not verified correct added geometry.",
    }
    save_json(out / "difference_report.json", report)
    print("[E1-J03] Robust Sim3:", sim3_report)
    print("[E1-J03] Geometric difference:", report)
    print(f"[E1-J03] saved: {out}")


if __name__ == "__main__":
    main()
