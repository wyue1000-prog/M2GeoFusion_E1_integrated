from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from common import ensure_dir, image_tensor_to_rgb_uint8, write_ascii_ply
from joint_demo_export import export_joint_demo_ply


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joint-npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--joint-demo-out", help="Independent original Joint Demo PLY path")
    args = ap.parse_args()

    out_dir = ensure_dir(args.out)
    joint = np.load(args.joint_npz)
    source = str(np.asarray(joint["camera_param_source"]).item()) if "camera_param_source" in joint else "unknown"
    P_lg = np.asarray(joint["world_points_from_depth"])[0]
    P_ls = np.asarray(joint["world_points_from_depth"])[1]
    rgb_lg = image_tensor_to_rgb_uint8(joint["images"][0])
    rgb_ls = image_tensor_to_rgb_uint8(joint["images"][1])

    P_lg_flat, P_ls_flat = P_lg.reshape(-1, 3), P_ls.reshape(-1, 3)
    C_lg_flat, C_ls_flat = rgb_lg.reshape(-1, 3), rgb_ls.reshape(-1, 3)
    valid_lg = np.isfinite(P_lg_flat).all(axis=1)
    valid_ls = np.isfinite(P_ls_flat).all(axis=1)
    P_lg_valid, P_ls_valid = P_lg_flat[valid_lg].astype(np.float32), P_ls_flat[valid_ls].astype(np.float32)
    C_lg_valid, C_ls_valid = C_lg_flat[valid_lg].astype(np.uint8), C_ls_flat[valid_ls].astype(np.uint8)

    write_ascii_ply(out_dir / "01_joint_LG.ply", P_lg_valid, C_lg_valid)
    write_ascii_ply(out_dir / "02_joint_LS.ply", P_ls_valid, C_ls_valid)
    gray = np.tile(np.array([[180, 180, 180]], np.uint8), (len(P_lg_valid), 1))
    red = np.tile(np.array([[255, 0, 0]], np.uint8), (len(P_ls_valid), 1))
    write_ascii_ply(
        out_dir / "03_joint_LG_gray_LS_red.ply",
        np.concatenate([P_lg_valid, P_ls_valid], axis=0),
        np.concatenate([gray, red], axis=0),
    )
    if source == "calibration":
        write_ascii_ply(
            out_dir / "03_joint_LG_gray_LS_red_calib.ply",
            np.concatenate([P_lg_valid, P_ls_valid], axis=0),
            np.concatenate([gray, red], axis=0),
        )
    print(f"[E1-05] camera parameter source = {source}")
    print(f"[E1-05] Joint-LG points = {len(P_lg_valid)}")
    print(f"[E1-05] Joint-LS points = {len(P_ls_valid)}")
    print(f"[E1-05] saved to: {out_dir}")
    if args.joint_demo_out:
        lg_count, ls_count, total, threshold = export_joint_demo_ply(args.joint_npz, args.joint_demo_out)
        print("[E1-05] 07 joint demo source : world_points_from_depth (original Joint frame)")
        print(f"[E1-05] 07 global confidence percentile 50 : {threshold:.8g}")
        print(f"[E1-05] 07 Joint-LG / Joint-LS / total : {lg_count} / {ls_count} / {total}")
        print(f"[E1-05] saved 07 : {args.joint_demo_out}")


if __name__ == "__main__":
    main()
