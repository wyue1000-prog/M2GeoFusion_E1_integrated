from __future__ import annotations

"""
M²GeoFusion E1 integrated demo
==============================
Only edit the USER SETTINGS section below, then run:

    python demo_all.py

The demo automatically calls Grounded-SAM2 and VGGT in their own conda envs.
You no longer need to run 01/02/03/04/05 one by one.
"""

import json
import os
import shlex
import subprocess
from pathlib import Path

# ============================================================
#                    USER SETTINGS
# ============================================================

# 1) This code project directory (usually no need to edit)
PROJECT_ROOT = Path(__file__).resolve().parent

# 2) Experiment name. Change this for every new pair to avoid overwriting old results.
EXPERIMENT_NAME = "test002"

# 3) Input LG / LS images
LG_PATH = PROJECT_ROOT / "data/test002/LG/Camera3_416.bmp"
LS_PATH = PROJECT_ROOT / "data/test002/LS/Camera4_416.bmp"

# 4) Camera-parameter source. Change only this switch and, for mode 1, the MAT path.
# 0 = VGGT Camera Head predicted intrinsics/extrinsics (legacy behaviour)
# 1 = MATLAB calibrated intrinsics/extrinsics; VGGT Camera Head is not used for geometry
CAMERA_PARAM_MODE = 0
CALIB_MAT_PATH = PROJECT_ROOT / "calibration/LG_LS_calibration_numeric.mat"

# 5) VGGT paths
VGGT_ROOT = Path("/home/wqy/workspace/vggt")
VGGT_CHECKPOINT = VGGT_ROOT / "checkpoints/model.pt"

# 6) Grounded-SAM2 paths
GROUNDED_SAM_ROOT = Path("/home/wqy/workspace/Grounded-SAM-2")
SAM2_CHECKPOINT = GROUNDED_SAM_ROOT / "checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = GROUNDED_SAM_ROOT / "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = GROUNDED_SAM_ROOT / "gdino_checkpoints/groundingdino_swint_ogc.pth"

# 7) Conda
CONDA_EXE = Path("/home/wqy/anaconda3/bin/conda")
GROUNDED_SAM_ENV = "grounded_sam2"
VGGT_ENV = "vggt"

# 8) Output: same structure as previous E1
OUTPUT_ROOT = PROJECT_ROOT / "outputs/test002"

# 9) Need-mask settings
TEXT_PROMPT = "wire harness. pipe."
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
DILATE_RADIUS = 20

# 10) VGGT / alignment settings
VGGT_PREPROCESS_MODE = "crop"      # crop | pad
ALIGNMENT = "scale"                # scale | sim3
DEDUP_FACTOR = 1.5

# 11) Track-head settings
# Query points are placed on LG (first/reference frame), preferentially inside Need Mask.
TRACK_MAX_POINTS = 300
TRACK_VIS_THRESHOLD = 0.50
TRACK_CONF_THRESHOLD = 0.001

# 12) Stage switches. Normally keep all True.
RUN_MASK = True
RUN_VGGT = True
RUN_SELECTION = True
RUN_MERGE = True
RUN_JOINT_EXPORT = True

# ============================================================
#                END OF USER SETTINGS
# ============================================================


def run(cmd):
    print("\n[M2GF-DEMO] >>>", " ".join(shlex.quote(str(x)) for x in cmd))
    subprocess.run([str(x) for x in cmd], check=True)


def conda_python(env_name, script, *script_args):
    return [
        CONDA_EXE, "run", "--no-capture-output", "-n", env_name,
        "python", script, *script_args,
    ]


def assert_exists(path: Path, label: str, required=True):
    if required and not path.expanduser().exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def main():
    lg = LG_PATH.expanduser().resolve()
    ls = LS_PATH.expanduser().resolve()
    if CAMERA_PARAM_MODE not in (0, 1):
        raise ValueError("CAMERA_PARAM_MODE must be 0 or 1")
    calib = Path(CALIB_MAT_PATH).expanduser().resolve() if CAMERA_PARAM_MODE == 1 else None
    out = (OUTPUT_ROOT / EXPERIMENT_NAME).expanduser().resolve()

    assert_exists(CONDA_EXE, "conda")
    assert_exists(lg, "LG image")
    assert_exists(ls, "LS image")
    assert_exists(VGGT_ROOT, "VGGT root")
    assert_exists(VGGT_CHECKPOINT, "VGGT checkpoint")
    assert_exists(GROUNDED_SAM_ROOT, "Grounded-SAM2 root")
    assert_exists(SAM2_CHECKPOINT, "SAM2 checkpoint")
    assert_exists(GROUNDING_DINO_CONFIG, "GroundingDINO config")
    assert_exists(GROUNDING_DINO_CHECKPOINT, "GroundingDINO checkpoint")
    if CAMERA_PARAM_MODE == 1 and not os.path.isfile(calib):
        raise FileNotFoundError(calib)

    if CAMERA_PARAM_MODE == 0:
        print("[M2GF-DEMO] CAMERA PARAM SOURCE = VGGT predicted intrinsics/extrinsics")
    else:
        print("[M2GF-DEMO] CAMERA PARAM SOURCE = MATLAB calibrated intrinsics/extrinsics")
        print("[M2GF-DEMO] VGGT Camera Head parameters are NOT used for geometry")

    mask_dir = out / "01_mask"
    vggt_dir = out / "02_vggt"
    select_dir = out / "03_selection"
    cloud_dir = out / "04_pointcloud"
    joint_view_dir = vggt_dir / "joint_view"
    for d in [mask_dir, vggt_dir, select_dir, cloud_dir, joint_view_dir]:
        d.mkdir(parents=True, exist_ok=True)

    manifest = {
        "experiment_name": EXPERIMENT_NAME,
        "LG": str(lg),
        "LS": str(ls),
        "calibration": str(calib) if calib else None,
        "camera_param_mode": CAMERA_PARAM_MODE,
        "camera_param_source": "calibration" if CAMERA_PARAM_MODE == 1 else "vggt_prediction",
        "prompt": TEXT_PROMPT,
        "dilate_radius": DILATE_RADIUS,
        "alignment": ALIGNMENT,
        "dedup_factor": DEDUP_FACTOR,
        "track_max_points": TRACK_MAX_POINTS,
        "track_vis_threshold": TRACK_VIS_THRESHOLD,
        "track_conf_threshold": TRACK_CONF_THRESHOLD,
        "grounded_sam_env": GROUNDED_SAM_ENV,
        "vggt_env": VGGT_ENV,
    }
    (out / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if RUN_MASK:
        run(conda_python(
            GROUNDED_SAM_ENV, PROJECT_ROOT / "01_make_need_mask.py",
            "--lg", lg,
            "--out", mask_dir,
            "--prompt", TEXT_PROMPT,
            "--dilate-radius", DILATE_RADIUS,
            "--box-threshold", BOX_THRESHOLD,
            "--text-threshold", TEXT_THRESHOLD,
            "--grounded-sam-root", GROUNDED_SAM_ROOT,
            "--sam2-checkpoint", SAM2_CHECKPOINT,
            "--sam2-model-config", SAM2_MODEL_CONFIG,
            "--grounding-dino-config", GROUNDING_DINO_CONFIG,
            "--grounding-dino-checkpoint", GROUNDING_DINO_CHECKPOINT,
        ))

    if RUN_VGGT:
        cmd02 = conda_python(
            VGGT_ENV, PROJECT_ROOT / "02_run_vggt.py",
            "--lg", lg,
            "--ls", ls,
            "--out", vggt_dir,
            "--vggt-root", VGGT_ROOT,
            "--checkpoint", VGGT_CHECKPOINT,
            "--preprocess-mode", VGGT_PREPROCESS_MODE,
            "--need-mask", mask_dir / "need_mask.npy",
            "--track-max-points", TRACK_MAX_POINTS,
            "--track-vis-threshold", TRACK_VIS_THRESHOLD,
            "--track-conf-threshold", TRACK_CONF_THRESHOLD,
            "--camera-param-mode", CAMERA_PARAM_MODE,
        )
        if CAMERA_PARAM_MODE == 1:
            cmd02 += ["--calib-mat", calib]
        run(cmd02)

    if RUN_SELECTION:
        cmd = conda_python(
            GROUNDED_SAM_ENV, PROJECT_ROOT / "03_select_ls_points.py",
            "--lg", lg,
            "--ls", ls,
            "--need-mask", mask_dir / "need_mask.npy",
            "--gray-npz", vggt_dir / "gray_only.npz",
            "--joint-npz", vggt_dir / "joint_LG_LS.npz",
            "--out", select_dir,
            "--alignment", ALIGNMENT,
            "--preprocess-mode", VGGT_PREPROCESS_MODE,
            "--camera-param-mode", CAMERA_PARAM_MODE,
        )
        if CAMERA_PARAM_MODE == 1:
            cmd += ["--calib-mat", calib]
        run(cmd)

    if RUN_MERGE:
        run(conda_python(
            GROUNDED_SAM_ENV, PROJECT_ROOT / "04_merge_and_export.py",
            "--selection-data", select_dir / "selection_data.npz",
            "--out", cloud_dir,
            "--dedup-factor", DEDUP_FACTOR,
        ))

    if RUN_JOINT_EXPORT:
        run(conda_python(
            GROUNDED_SAM_ENV, PROJECT_ROOT / "05_export_joint_views.py",
            "--joint-npz", vggt_dir / "joint_LG_LS.npz",
            "--out", joint_view_dir,
        ))

    print("\n[M2GF-DEMO] =============================================")
    print("[M2GF-DEMO] ALL DONE")
    print(f"[M2GF-DEMO] Output: {out}")
    print("\nNew diagnostic outputs:")
    print("  Pose TXT      :", vggt_dir / "vggt_pose.txt")
    print("  Relative pose :", vggt_dir / "relative_pose_LS_to_LG.txt")
    print("  Track image   :", vggt_dir / "track_LG_LS.png")
    print("  Track TXT     :", vggt_dir / "track_LG_LS.txt")
    print("  Joint PLY     :", joint_view_dir / "03_joint_LG_gray_LS_red.ply")
    print("\nExisting E1 outputs are still kept in the same 01/02/03/04 folders.")


if __name__ == "__main__":
    main()
