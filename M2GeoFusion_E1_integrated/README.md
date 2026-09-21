# M²GeoFusion E1 integrated demo

This version keeps the original E1 output structure and adds:

- `02_vggt/vggt_pose.txt`: all predicted intrinsics/extrinsics plus joint relative pose.
- `02_vggt/relative_pose_LS_to_LG.txt`: predicted transform from LS camera coordinates to LG camera coordinates.
- `02_vggt/track_LG_LS.png`: VGGT Track Head correspondence visualization (LG on left, LS on right).
- `02_vggt/track_LG_LS.txt` and `.npz`: numeric track coordinates, visibility and confidence.
- `02_vggt/joint_view/01_joint_LG.ply`
- `02_vggt/joint_view/02_joint_LS.ply`
- `02_vggt/joint_view/03_joint_LG_gray_LS_red.ply`

## How to use

1. Put the whole folder under your project, e.g. `/home/wqy/workspace/M2GeoFusion_E1_code/M2GeoFusion_E1/`.
2. Edit **only the USER SETTINGS at the top of `demo_all.py`**.
3. Run from any normal shell / base conda environment:

```bash
python demo_all.py
```

The demo itself uses `conda run` to invoke:

- `grounded_sam2` for Grounded-SAM2 and post-processing.
- `vggt` for VGGT inference and Track Head.

## Output structure

```text
outputs/<EXPERIMENT_NAME>/
├── experiment_manifest.json
├── 01_mask/
│   ├── target_mask.png
│   ├── need_mask.png
│   ├── need_mask.npy
│   ├── overlay.jpg
│   └── mask_info.json
├── 02_vggt/
│   ├── gray_only.npz
│   ├── joint_LG_LS.npz
│   ├── vggt_pose.txt
│   ├── relative_pose_LS_to_LG.txt
│   ├── track_LG_LS.png
│   ├── track_LG_LS.txt
│   ├── track_LG_LS.npz
│   ├── vggt_info.json
│   └── joint_view/
│       ├── 01_joint_LG.ply
│       ├── 02_joint_LS.ply
│       └── 03_joint_LG_gray_LS_red.ply
├── 03_selection/
└── 04_pointcloud/
```

### Track convention

The first image is LG and is the Track Head reference/query frame. `track_LG_LS.png` shows LG on the left and LS on the right. Only matches passing both visibility and confidence thresholds are drawn.

### Pose convention

VGGT extrinsics are treated as `world -> camera` matrices. The reported `T_LG_from_LS` is computed as:

```text
T_LG_from_LS = T_world_to_LG @ inverse(T_world_to_LS)
```

and therefore maps a 3D point represented in LS-camera coordinates into LG-camera coordinates.
