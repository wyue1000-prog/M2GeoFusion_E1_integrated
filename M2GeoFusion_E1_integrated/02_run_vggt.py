from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from config import (
    VGGT_ROOT,
    VGGT_CHECKPOINT,
    VGGT_PREPROCESS_MODE,
    TRACK_MAX_POINTS,
    TRACK_VIS_THRESHOLD,
    TRACK_CONF_THRESHOLD,
)
from common import ensure_dir, save_json, preprocess_mask_like_vggt, image_tensor_to_rgb_uint8


def unwrap_state_dict(obj):
    if isinstance(obj, dict):
        if "model" in obj and isinstance(obj["model"], dict):
            return obj["model"]
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
    return obj


def make_grid_queries(h, w, max_points=300, mask=None):
    """Create approximately uniform query points in model-input pixel coordinates."""
    max_points = max(int(max_points), 1)
    aspect = w / max(h, 1)
    nx = max(2, int(round(np.sqrt(max_points * aspect))))
    ny = max(2, int(round(max_points / nx)))
    xs = np.linspace(0.04 * (w - 1), 0.96 * (w - 1), nx)
    ys = np.linspace(0.04 * (h - 1), 0.96 * (h - 1), ny)
    xx, yy = np.meshgrid(xs, ys)
    q = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)

    if mask is not None:
        ui = np.clip(np.rint(q[:, 0]).astype(int), 0, w - 1)
        vi = np.clip(np.rint(q[:, 1]).astype(int), 0, h - 1)
        q_masked = q[mask[vi, ui] > 0]
        # If target ROI is too small, retain target queries but add uniform queries as fallback.
        if len(q_masked) >= 12:
            q = q_masked

    if len(q) > max_points:
        ids = np.linspace(0, len(q) - 1, max_points).astype(int)
        q = q[ids]
    return q.astype(np.float32)


def load_local_vggt(VGGT, checkpoint_path: Path, device: str):
    print(f"[E1-02] Loading VGGT from: {checkpoint_path}")
    model = VGGT()
    try:
        state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(str(checkpoint_path), map_location="cpu")
    state = unwrap_state_dict(state)
    if isinstance(state, dict) and state:
        first_key = next(iter(state.keys()))
        if first_key.startswith("module."):
            state = {k[len("module."):]: v for k, v in state.items()}
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print("[E1-02] WARNING: checkpoint key mismatch")
        print("  missing:", incompatible.missing_keys[:20])
        print("  unexpected:", incompatible.unexpected_keys[:20])
    model.eval().to(device)
    return model


def run_vggt(image_paths, model, device, load_and_preprocess_images, pose_encoding_to_extri_intri,
             unproject_depth_map_to_point_map, preprocess_mode="crop", query_points=None):
    image_paths = [str(Path(x).expanduser().resolve()) for x in image_paths]
    images = load_and_preprocess_images(image_paths, mode=preprocess_mode).to(device)
    print(f"[E1-02] input order = {image_paths}")
    print(f"[E1-02] preprocessed tensor = {tuple(images.shape)}")

    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    q_tensor = None
    if query_points is not None:
        q_tensor = torch.as_tensor(query_points, dtype=torch.float32, device=device)

    with torch.no_grad():
        if device == "cuda":
            with torch.autocast(device_type="cuda", dtype=dtype):
                if q_tensor is None:
                    pred = model(images)
                else:
                    try:
                        pred = model(images, query_points=q_tensor)
                    except TypeError:
                        # Compatibility fallback for older VGGT forward signatures.
                        pred = model(images)
                        if getattr(model, "track_head", None) is None:
                            raise RuntimeError("Current VGGT code/checkpoint has no usable track head.")
                        images_b = images.unsqueeze(0) if images.ndim == 4 else images
                        aggregated_tokens_list, patch_start_idx = model.aggregator(images_b)
                        track_list, vis, conf = model.track_head(
                            aggregated_tokens_list,
                            images=images_b,
                            patch_start_idx=patch_start_idx,
                            query_points=q_tensor.unsqueeze(0) if q_tensor.ndim == 2 else q_tensor,
                        )
                        pred["track"] = track_list[-1]
                        pred["vis"] = vis
                        pred["conf"] = conf
        else:
            pred = model(images, query_points=q_tensor) if q_tensor is not None else model(images)

    extrinsic, intrinsic = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])
    pred["extrinsic"] = extrinsic
    pred["intrinsic"] = intrinsic

    out = {}
    for k, v in pred.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.detach().float().cpu().numpy().squeeze(0)
    out["images"] = images.detach().cpu().numpy()
    if query_points is not None:
        out["query_points"] = np.asarray(query_points, dtype=np.float32)

    depth = out["depth"]
    out["world_points_from_depth"] = unproject_depth_map_to_point_map(
        depth, out["extrinsic"], out["intrinsic"]
    )
    out.pop("pose_enc_list", None)
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def save_npz(path, pred):
    clean = {k: v for k, v in pred.items() if isinstance(v, np.ndarray)}
    np.savez_compressed(path, **clean)


def to_homogeneous_extrinsic(E):
    T = np.eye(4, dtype=np.float64)
    T[:3, :4] = np.asarray(E, dtype=np.float64)
    return T


def matrix_text(M, fmt=" .9f"):
    return "\n".join(" ".join(format(float(x), fmt) for x in row) for row in np.asarray(M))


def camera_center_from_extrinsic(E):
    E = np.asarray(E, dtype=np.float64)
    R, t = E[:3, :3], E[:3, 3]
    return -R.T @ t


def save_pose_txt(out_dir: Path, gray, joint):
    Eg = np.asarray(gray["extrinsic"])[0]
    Kg = np.asarray(gray["intrinsic"])[0]
    EJg, EJs = np.asarray(joint["extrinsic"])[0], np.asarray(joint["extrinsic"])[1]
    KJg, KJs = np.asarray(joint["intrinsic"])[0], np.asarray(joint["intrinsic"])[1]

    Tg = to_homogeneous_extrinsic(EJg)
    Ts = to_homogeneous_extrinsic(EJs)
    # E is world->camera. Therefore this maps LS-camera coordinates into LG-camera coordinates.
    T_lg_from_ls = Tg @ np.linalg.inv(Ts)
    T_ls_from_lg = np.linalg.inv(T_lg_from_ls)

    lines = []
    lines.append("M2GeoFusion / VGGT predicted camera parameters\n")
    lines.append("Convention: extrinsic maps world -> camera: X_cam = R * X_world + t\n")
    lines.append("\n=== Pass A: LG-only ===\n")
    lines.append("Intrinsic K_LG_only (3x3):\n" + matrix_text(Kg) + "\n")
    lines.append("Extrinsic E_LG_only (3x4):\n" + matrix_text(Eg) + "\n")
    lines.append("Camera center C_LG_only (world coordinates):\n" + " ".join(f"{x:.9f}" for x in camera_center_from_extrinsic(Eg)) + "\n")
    lines.append("\n=== Pass B: Joint [LG, LS] ===\n")
    lines.append("Intrinsic K_joint_LG (3x3):\n" + matrix_text(KJg) + "\n")
    lines.append("Extrinsic E_joint_LG (3x4):\n" + matrix_text(EJg) + "\n")
    lines.append("Camera center C_joint_LG:\n" + " ".join(f"{x:.9f}" for x in camera_center_from_extrinsic(EJg)) + "\n")
    lines.append("\nIntrinsic K_joint_LS (3x3):\n" + matrix_text(KJs) + "\n")
    lines.append("Extrinsic E_joint_LS (3x4):\n" + matrix_text(EJs) + "\n")
    lines.append("Camera center C_joint_LS:\n" + " ".join(f"{x:.9f}" for x in camera_center_from_extrinsic(EJs)) + "\n")
    lines.append("\n=== Relative pose predicted by VGGT ===\n")
    lines.append("T_LG_from_LS (4x4), maps a 3D point from LS camera coordinates to LG camera coordinates:\n")
    lines.append(matrix_text(T_lg_from_ls) + "\n")
    lines.append("\nT_LS_from_LG (4x4), inverse transform:\n")
    lines.append(matrix_text(T_ls_from_lg) + "\n")

    (out_dir / "vggt_pose.txt").write_text("".join(lines), encoding="utf-8")
    (out_dir / "relative_pose_LS_to_LG.txt").write_text(
        "T_LG_from_LS\n" + matrix_text(T_lg_from_ls) + "\n",
        encoding="utf-8",
    )
    return T_lg_from_ls, T_ls_from_lg


def save_track_visualization(out_dir: Path, joint, vis_thr=0.5, conf_thr=0.5):
    if "track" not in joint or "vis" not in joint:
        print("[E1-02] WARNING: no track output found; track image was not generated.")
        return 0

    track = np.asarray(joint["track"])
    vis = np.asarray(joint["vis"])
    conf = np.asarray(joint.get("conf", np.ones_like(vis)))
    query = np.asarray(joint.get("query_points", track[0]), dtype=np.float32)

    if track.ndim != 3 or track.shape[0] < 2:
        raise RuntimeError(f"Unexpected track shape: {track.shape}; expected [S,N,2].")

    lg_rgb = image_tensor_to_rgb_uint8(joint["images"][0])
    ls_rgb = image_tensor_to_rgb_uint8(joint["images"][1])
    h1, w1 = lg_rgb.shape[:2]
    h2, w2 = ls_rgb.shape[:2]
    canvas_h = max(h1, h2)
    canvas = np.zeros((canvas_h, w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = cv2.cvtColor(lg_rgb, cv2.COLOR_RGB2BGR)
    canvas[:h2, w1:w1 + w2] = cv2.cvtColor(ls_rgb, cv2.COLOR_RGB2BGR)

    p_lg = track[0]
    p_ls = track[1]
    good = (
        np.isfinite(p_lg).all(axis=1) & np.isfinite(p_ls).all(axis=1)
        & (p_lg[:, 0] >= 0) & (p_lg[:, 0] < w1) & (p_lg[:, 1] >= 0) & (p_lg[:, 1] < h1)
        & (p_ls[:, 0] >= 0) & (p_ls[:, 0] < w2) & (p_ls[:, 1] >= 0) & (p_ls[:, 1] < h2)
        & (vis[0] >= vis_thr) & (vis[1] >= vis_thr)
        & (conf[0] >= conf_thr) & (conf[1] >= conf_thr)
    )
    ids = np.where(good)[0]

    rng = np.random.default_rng(7)
    for idx in ids:
        c = tuple(int(x) for x in rng.integers(70, 256, size=3))
        a = tuple(np.rint(p_lg[idx]).astype(int))
        b0 = np.rint(p_ls[idx]).astype(int)
        b = (int(b0[0] + w1), int(b0[1]))
        cv2.line(canvas, a, b, c, 1, cv2.LINE_AA)
        cv2.circle(canvas, a, 2, c, -1, cv2.LINE_AA)
        cv2.circle(canvas, b, 2, c, -1, cv2.LINE_AA)

    cv2.putText(canvas, "LG (query/reference)", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, "LS (tracked)", (w1 + 12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_dir / "track_LG_LS.png"), canvas)

    txt = out_dir / "track_LG_LS.txt"
    with open(txt, "w", encoding="utf-8") as f:
        f.write("# id lg_u lg_v ls_u ls_v vis_lg vis_ls conf_lg conf_ls keep\n")
        for i in range(track.shape[1]):
            f.write(
                f"{i} {p_lg[i,0]:.6f} {p_lg[i,1]:.6f} "
                f"{p_ls[i,0]:.6f} {p_ls[i,1]:.6f} "
                f"{vis[0,i]:.6f} {vis[1,i]:.6f} "
                f"{conf[0,i]:.6f} {conf[1,i]:.6f} {int(good[i])}\n"
            )

    np.savez_compressed(
        out_dir / "track_LG_LS.npz",
        query_points=query,
        track=track,
        vis=vis,
        conf=conf,
        good_mask=good.astype(np.uint8),
    )
    print(f"[E1-02] track matches kept = {len(ids)} / {track.shape[1]}")
    return int(len(ids))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lg", required=True)
    ap.add_argument("--ls", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vggt-root", default=str(VGGT_ROOT))
    ap.add_argument("--checkpoint", default=str(VGGT_CHECKPOINT))
    ap.add_argument("--preprocess-mode", default=VGGT_PREPROCESS_MODE, choices=["crop", "pad"])
    ap.add_argument("--need-mask", default=None, help="Optional original-resolution LG Need mask used only to place track queries.")
    ap.add_argument("--track-max-points", type=int, default=TRACK_MAX_POINTS)
    ap.add_argument("--track-vis-threshold", type=float, default=TRACK_VIS_THRESHOLD)
    ap.add_argument("--track-conf-threshold", type=float, default=TRACK_CONF_THRESHOLD)
    args = ap.parse_args()

    lg = Path(args.lg).expanduser().resolve()
    ls = Path(args.ls).expanduser().resolve()
    vggt_root = Path(args.vggt_root).expanduser().resolve()
    ckpt = Path(args.checkpoint).expanduser().resolve()
    out_dir = ensure_dir(args.out)
    for p in [lg, ls, vggt_root, ckpt]:
        if not p.exists():
            raise FileNotFoundError(p)

    sys.path.insert(0, str(vggt_root))
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.geometry import unproject_depth_map_to_point_map

    if not torch.cuda.is_available():
        raise RuntimeError("VGGT experiment requires CUDA in this setup.")
    device = "cuda"
    print("[E1-02] GPU:", torch.cuda.get_device_name(0))
    model = load_local_vggt(VGGT, ckpt, device)

    print("\n[E1-02] ===== Pass A: LG only =====")
    gray = run_vggt(
        [lg], model, device, load_and_preprocess_images, pose_encoding_to_extri_intri,
        unproject_depth_map_to_point_map, preprocess_mode=args.preprocess_mode,
    )
    save_npz(out_dir / "gray_only.npz", gray)
    gc.collect(); torch.cuda.empty_cache()

    # Prepare track queries in the exact joint-input pixel coordinate system.
    preview = load_and_preprocess_images([str(lg), str(ls)], mode=args.preprocess_mode)
    h, w = preview.shape[-2:]
    need_vggt = None
    if args.need_mask:
        mask_path = Path(args.need_mask).expanduser().resolve()
        if mask_path.exists():
            need_raw = np.load(mask_path)
            need_vggt = preprocess_mask_like_vggt(
                need_raw, image_paths=[str(lg), str(ls)], target_index=0,
                mode=args.preprocess_mode,
            )
    query_points = make_grid_queries(h, w, args.track_max_points, mask=need_vggt)
    print(f"[E1-02] track query points = {len(query_points)}")

    print("\n[E1-02] ===== Pass B: [LG, LS] + Track Head =====")
    joint = run_vggt(
        [lg, ls], model, device, load_and_preprocess_images, pose_encoding_to_extri_intri,
        unproject_depth_map_to_point_map, preprocess_mode=args.preprocess_mode,
        query_points=query_points,
    )
    save_npz(out_dir / "joint_LG_LS.npz", joint)

    T_lg_from_ls, _ = save_pose_txt(out_dir, gray, joint)
    kept = save_track_visualization(
        out_dir, joint,
        vis_thr=args.track_vis_threshold,
        conf_thr=args.track_conf_threshold,
    )

    save_json(out_dir / "vggt_info.json", {
        "lg_path": str(lg),
        "ls_path": str(ls),
        "checkpoint": str(ckpt),
        "preprocess_mode": args.preprocess_mode,
        "gray_world_shape": list(gray["world_points_from_depth"].shape),
        "joint_world_shape": list(joint["world_points_from_depth"].shape),
        "input_order_joint": ["LG", "LS"],
        "track_query_count": int(len(query_points)),
        "track_kept_count": int(kept),
        "track_vis_threshold": float(args.track_vis_threshold),
        "track_conf_threshold": float(args.track_conf_threshold),
        "T_LG_from_LS_predicted": np.asarray(T_lg_from_ls).tolist(),
        "note": "world_points_from_depth uses VGGT depth + predicted camera parameters; track queries use LG as the first/reference frame.",
    })

    print("\n[E1-02] saved:")
    for name in [
        "gray_only.npz", "joint_LG_LS.npz", "vggt_pose.txt",
        "relative_pose_LS_to_LG.txt", "track_LG_LS.png", "track_LG_LS.txt",
    ]:
        print(" ", out_dir / name)


if __name__ == "__main__":
    main()
