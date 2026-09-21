from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Tuple, Dict, Any

import cv2
import numpy as np
from PIL import Image


def ensure_dir(path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path, obj: Dict[str, Any]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_ascii_ply(path, points, colors=None):
    """保存 XYZ / RGB ASCII PLY，便于直接用 CloudCompare 打开。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    valid = np.isfinite(points).all(axis=1)
    points = points[valid]

    if colors is not None:
        colors = np.asarray(colors).reshape(-1, 3)[valid]
        if np.issubdtype(colors.dtype, np.floating):
            if colors.size and colors.max() <= 1.0 + 1e-6:
                colors = colors * 255.0
        colors = np.clip(colors, 0, 255).astype(np.uint8)

    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")

        if colors is None:
            for p in points:
                f.write(f"{p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n")
        else:
            for p, c in zip(points, colors):
                f.write(
                    f"{p[0]:.8f} {p[1]:.8f} {p[2]:.8f} "
                    f"{int(c[0])} {int(c[1])} {int(c[2])}\n"
                )


def _processed_size_for_vggt(image_path, mode="crop", target_size=518):
    """复制 VGGT load_and_preprocess_images() 的尺寸计算规则。"""
    with Image.open(image_path) as img:
        width, height = img.size

    if mode == "pad":
        if width >= height:
            new_width = target_size
            new_height = round(height * (new_width / width) / 14) * 14
        else:
            new_height = target_size
            new_width = round(width * (new_height / height) / 14) * 14
    elif mode == "crop":
        new_width = target_size
        new_height = round(height * (new_width / width) / 14) * 14
    else:
        raise ValueError("mode must be 'crop' or 'pad'")

    return width, height, int(new_width), int(new_height)


def preprocess_mask_like_vggt(
    mask: np.ndarray,
    image_paths: Iterable[str],
    target_index: int = 0,
    mode: str = "crop",
    target_size: int = 518,
):
    """
    把原始 LG 二值 mask 做与 VGGT load_and_preprocess_images() 等价的
    resize / center-crop / batch-padding。

    image_paths 必须和真正送入 VGGT 的顺序完全一致。
    target_index=0 表示 mask 属于第一张 LG。
    """
    image_paths = list(image_paths)
    infos = [_processed_size_for_vggt(p, mode, target_size) for p in image_paths]

    processed_shapes = []
    for _, _, nw, nh in infos:
        if mode == "crop" and nh > target_size:
            ph = target_size
        else:
            ph = nh
        if mode == "pad":
            ph = target_size
            pw = target_size
        else:
            pw = nw
        processed_shapes.append((ph, pw))

    max_h = max(x[0] for x in processed_shapes)
    max_w = max(x[1] for x in processed_shapes)

    _, _, new_w, new_h = infos[target_index]
    m = (mask > 0).astype(np.uint8) * 255
    m = cv2.resize(m, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    if mode == "crop" and new_h > target_size:
        start_y = (new_h - target_size) // 2
        m = m[start_y:start_y + target_size, :]

    if mode == "pad":
        h_padding = target_size - m.shape[0]
        w_padding = target_size - m.shape[1]
        top = h_padding // 2
        bottom = h_padding - top
        left = w_padding // 2
        right = w_padding - left
        m = cv2.copyMakeBorder(
            m, top, bottom, left, right,
            borderType=cv2.BORDER_CONSTANT, value=0
        )

    # VGGT 会把 batch 内不同 H/W 的图继续 pad 到相同尺寸。
    h_padding = max_h - m.shape[0]
    w_padding = max_w - m.shape[1]
    if h_padding > 0 or w_padding > 0:
        top = h_padding // 2
        bottom = h_padding - top
        left = w_padding // 2
        right = w_padding - left
        m = cv2.copyMakeBorder(
            m, top, bottom, left, right,
            borderType=cv2.BORDER_CONSTANT, value=0
        )

    return (m > 0).astype(np.uint8)


def image_tensor_to_rgb_uint8(image_chw):
    x = np.asarray(image_chw)
    if x.ndim != 3 or x.shape[0] != 3:
        raise ValueError(f"Expected CHW RGB, got {x.shape}")
    x = np.transpose(x, (1, 2, 0))
    x = np.clip(x, 0, 1)
    return (x * 255.0 + 0.5).astype(np.uint8)


def project_world_to_image(points_world, extrinsic_3x4, intrinsic_3x3):
    """
    VGGT 的 extrinsic 按 world -> camera 使用。
    Xc = R Xw + t，再由 K 投影。
    """
    p = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    E = np.asarray(extrinsic_3x4, dtype=np.float64)
    K = np.asarray(intrinsic_3x3, dtype=np.float64)

    R = E[:3, :3]
    t = E[:3, 3]
    pc = p @ R.T + t[None, :]
    z = pc[:, 2]

    uv = np.full((len(p), 2), np.nan, dtype=np.float64)
    valid_z = np.isfinite(pc).all(axis=1) & (z > 1e-9)

    q = pc[valid_z] @ K.T
    uv[valid_z, 0] = q[:, 0] / q[:, 2]
    uv[valid_z, 1] = q[:, 1] / q[:, 2]
    return uv, z, pc


def estimate_scale_from_local_edges(anchor_map, joint_lg_map, max_samples=300000):
    """
    用同一 LG 像素网格的局部 3D 边长比估尺度：
        s = median( ||dP_anchor|| / ||dP_joint|| )
    对平移不敏感，且不会使用物理标定尺度。
    """
    A = np.asarray(anchor_map, dtype=np.float64)
    J = np.asarray(joint_lg_map, dtype=np.float64)
    if A.shape != J.shape:
        raise ValueError(f"Point-map shape mismatch: anchor {A.shape}, joint-LG {J.shape}")

    ratios = []

    for a1, a2, j1, j2 in [
        (A[:, 1:, :], A[:, :-1, :], J[:, 1:, :], J[:, :-1, :]),
        (A[1:, :, :], A[:-1, :, :], J[1:, :, :], J[:-1, :, :]),
    ]:
        da = np.linalg.norm(a1 - a2, axis=-1)
        dj = np.linalg.norm(j1 - j2, axis=-1)
        v = (
            np.isfinite(da) & np.isfinite(dj) &
            (da > 1e-9) & (dj > 1e-9)
        )
        r = da[v] / dj[v]
        # 去掉极端离群
        r = r[np.isfinite(r) & (r > 1e-5) & (r < 1e5)]
        if r.size:
            ratios.append(r)

    if not ratios:
        raise RuntimeError("No valid local-edge ratios for scale alignment.")

    ratios = np.concatenate(ratios)
    if ratios.size > max_samples:
        rng = np.random.default_rng(0)
        ratios = rng.choice(ratios, size=max_samples, replace=False)

    q1, q99 = np.percentile(ratios, [1, 99])
    ratios = ratios[(ratios >= q1) & (ratios <= q99)]
    return float(np.median(ratios))


def umeyama_similarity(src, dst, max_points=50000):
    """
    估计 dst ~= scale * (src @ R.T) + t
    src/dst: (N,3) 对应点。
    """
    src = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 3)
    valid = np.isfinite(src).all(axis=1) & np.isfinite(dst).all(axis=1)
    src = src[valid]
    dst = dst[valid]

    if len(src) < 20:
        raise RuntimeError("Too few correspondences for Sim(3).")

    if len(src) > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(src), max_points, replace=False)
        src = src[idx]
        dst = dst[idx]

    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    xs = src - mu_s
    xd = dst - mu_d

    cov = (xd.T @ xs) / len(src)
    U, S, Vt = np.linalg.svd(cov)

    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[-1, -1] = -1

    R = U @ D @ Vt
    var_s = np.mean(np.sum(xs * xs, axis=1))
    scale = float(np.trace(np.diag(S) @ D) / max(var_s, 1e-12))
    t = mu_d - scale * (R @ mu_s)

    return scale, R, t


def apply_similarity(points, scale, R=None, t=None):
    p = np.asarray(points, dtype=np.float64)
    shape = p.shape
    p2 = p.reshape(-1, 3)

    if R is None:
        out = scale * p2
    else:
        out = scale * (p2 @ np.asarray(R).T)
        if t is not None:
            out = out + np.asarray(t)[None, :]

    return out.reshape(shape).astype(np.float32)


def finite_points_and_colors(points, colors=None):
    p = np.asarray(points).reshape(-1, 3)
    v = np.isfinite(p).all(axis=1)
    p = p[v]
    if colors is None:
        return p, None
    c = np.asarray(colors).reshape(-1, 3)[v]
    return p, c
