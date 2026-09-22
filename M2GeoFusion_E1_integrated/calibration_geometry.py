"""MATLAB stereo calibration helpers for the calibration-driven E1 path.

All public matrices use the OpenCV column-vector convention:
``X_ls = R_lg_to_ls @ X_lg + t_lg_to_ls``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.optimize import minimize_scalar
from scipy.spatial import cKDTree


@dataclass
class CalibrationData:
    K_lg: np.ndarray
    K_ls: np.ndarray
    R_lg_to_ls: np.ndarray
    t_lg_to_ls: np.ndarray
    image_size_lg_hw: tuple[int, int] | None = None
    image_size_ls_hw: tuple[int, int] | None = None
    source_description: str = "MATLAB stereo calibration"


def _fields(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "_fieldnames"):
        return {name: getattr(obj, name) for name in obj._fieldnames}
    if isinstance(obj, np.void) and obj.dtype.names:
        return {name: obj[name] for name in obj.dtype.names}
    if isinstance(obj, np.ndarray) and obj.size == 1:
        return _fields(obj.reshape(-1)[0])
    return {}


def _field(obj: Any, *names: str) -> Any:
    values = _fields(obj)
    lower = {str(key).lower(): value for key, value in values.items()}
    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def _find_stereo_container(obj: Any, depth: int = 0) -> Any | None:
    """Find the saved stereo-parameter struct without assuming its MAT variable name."""
    if depth > 4:
        return None
    if _field(obj, "CameraParameters1") is not None and _field(obj, "CameraParameters2") is not None:
        return obj
    for value in _fields(obj).values():
        found = _find_stereo_container(value, depth + 1)
        if found is not None:
            return found
    return None


def _matrix(value: Any, shape: tuple[int, int], label: str) -> np.ndarray:
    if value is None:
        raise KeyError(f"MAT file does not contain {label}")
    a = np.asarray(value, dtype=np.float64).squeeze()
    if a.shape != shape:
        raise ValueError(f"{label} must be {shape}, got {a.shape}")
    return a


def _vector(value: Any, label: str) -> np.ndarray:
    if value is None:
        raise KeyError(f"MAT file does not contain {label}")
    a = np.asarray(value, dtype=np.float64).reshape(-1)
    if a.shape != (3,):
        raise ValueError(f"{label} must have 3 values, got {a.shape}")
    return a


def _image_size_hw(camera: Any) -> tuple[int, int] | None:
    value = _field(camera, "ImageSize", "imageSize")
    if value is None:
        return None
    a = np.asarray(value).reshape(-1)
    return (int(a[0]), int(a[1])) if a.size >= 2 else None


def _camera_intrinsic(camera: Any, label: str) -> np.ndarray:
    """Read MATLAB K and convert its 1-based principal point to OpenCV pixels."""
    k = _field(camera, "K")
    if k is not None:
        K = _matrix(k, (3, 3), f"{label}.K")
    else:
        # Legacy MATLAB's IntrinsicMatrix is the transpose of usual OpenCV K.
        K = _matrix(_field(camera, "IntrinsicMatrix"), (3, 3), f"{label}.IntrinsicMatrix").T
    K = K.copy()
    K[0, 2] -= 1.0
    K[1, 2] -= 1.0
    if not (K[0, 0] > 0 and K[1, 1] > 0):
        raise ValueError(f"{label} has invalid focal lengths: {K[0, 0]}, {K[1, 1]}")
    return K


def _load_mat(path: Path) -> dict[str, Any]:
    try:
        from scipy.io import loadmat
        return {k: v for k, v in loadmat(path, squeeze_me=True, struct_as_record=False).items() if not k.startswith("__")}
    except NotImplementedError:
        pass
    except ValueError as exc:
        if "7.3" not in str(exc):
            raise
    # v7.3 MAT is HDF5. h5py preserves references, so expose top-level datasets
    # and fail with an actionable error if its schema needs a project-specific adapter.
    import h5py
    with h5py.File(path, "r") as f:
        return {k: np.array(v) for k, v in f.items() if not k.startswith("#")}


def load_user_calibration(mat_path: str | Path) -> CalibrationData:
    """Load numeric or standard MATLAB Stereo Camera Calibrator parameters.

    A numeric export uses ``K_LG``, ``K_LS``, ``R_LG_to_LS`` and
    ``t_LG_to_LS`` directly in the OpenCV column-vector convention.  In that
    form Camera1 is LG and Camera2 is LS.  Older ``stereoParameters`` MAT
    exports remain supported below.
    """
    path = Path(mat_path).expanduser().resolve()
    mat = _load_mat(path)

    # The current calibration export is a normal MATLAB -v7 numeric MAT file,
    # rather than a stereoParameters object.  Its K matrices are already in
    # their intended convention: do not transpose or apply MATLAB pixel-index
    # adjustments here.
    numeric_fields = ("K_LG", "K_LS", "R_LG_to_LS", "t_LG_to_LS")
    if all(name in mat for name in numeric_fields):
        K_lg = np.asarray(mat["K_LG"], dtype=np.float64)
        K_ls = np.asarray(mat["K_LS"], dtype=np.float64)
        R_lg_to_ls = np.asarray(
            mat["R_LG_to_LS"],
            dtype=np.float64,
        )
        t_lg_to_ls = np.asarray(
            mat["t_LG_to_LS"],
            dtype=np.float64,
        ).reshape(3)

        if K_lg.shape != (3, 3):
            raise ValueError(f"K_LG must be (3, 3), got {K_lg.shape}")
        if K_ls.shape != (3, 3):
            raise ValueError(f"K_LS must be (3, 3), got {K_ls.shape}")
        if R_lg_to_ls.shape != (3, 3):
            raise ValueError(f"R_LG_to_LS must be (3, 3), got {R_lg_to_ls.shape}")
        for label, K in (("K_LG", K_lg), ("K_LS", K_ls)):
            if not np.isclose(K[2, 2], 1.0, rtol=0.0, atol=1e-6):
                raise ValueError(f"{label}[2, 2] must be close to 1, got {K[2, 2]}")
            if not (K[0, 0] > 0 and K[1, 1] > 0):
                raise ValueError(f"{label} has invalid focal lengths: {K[0, 0]}, {K[1, 1]}")
        det_R = float(np.linalg.det(R_lg_to_ls))
        if not np.isclose(det_R, 1.0, rtol=0.0, atol=1e-2):
            raise ValueError(f"R_LG_to_LS is not a proper rotation (det={det_R:.6f})")

        image_size_lg = np.asarray(mat["imageSize_LG"], dtype=np.float64)
        image_size_ls = np.asarray(mat["imageSize_LS"], dtype=np.float64)
        print("K_LG:\n", K_lg)
        print("K_LS:\n", K_ls)
        print("R_LG_to_LS:\n", R_lg_to_ls)
        print("t_LG_to_LS:\n", t_lg_to_ls)
        print("det(R):", det_R)
        print("imageSize_LG:", image_size_lg)
        print("imageSize_LS:", image_size_ls)
        return CalibrationData(
            K_lg,
            K_ls,
            R_lg_to_ls,
            t_lg_to_ls,
            _image_size_hw({"imageSize": image_size_lg}),
            _image_size_hw({"imageSize": image_size_ls}),
            "MATLAB numeric K_LG/K_LS/R_LG_to_LS/t_LG_to_LS",
        )

    root = _find_stereo_container(mat)
    if root is None:
        opaque = [key for key, value in mat.items() if type(value).__name__ == "MatlabOpaque"]
        if opaque:
            raise ValueError(
                "This MAT contains a MATLAB MCOS calibrationSession object, not exported numeric stereo parameters "
                f"(opaque variable(s): {opaque}). In MATLAB Camera Calibrator, use Export Camera Parameters and save "
                "the resulting stereoParams (with CameraParameters1/CameraParameters2) to a new MAT file; then set "
                "CALIB_MAT_PATH to that exported file."
            )
        raise KeyError(
            "Could not find a stereo-parameter struct containing CameraParameters1/CameraParameters2. "
            f"Top-level MAT keys: {sorted(mat)}"
        )
    cam1 = _field(root, "CameraParameters1", "cameraParameters1")
    cam2 = _field(root, "CameraParameters2", "cameraParameters2")
    if cam1 is None or cam2 is None:
        raise KeyError(
            "Expected CameraParameters1/CameraParameters2 (possibly under stereoParams). "
            f"Top-level MAT keys: {sorted(mat)}"
        )

    K_lg = _camera_intrinsic(cam1, "CameraParameters1")
    K_ls = _camera_intrinsic(cam2, "CameraParameters2")
    legacy_R = _field(root, "RotationOfCamera2")
    legacy_t = _field(root, "TranslationOfCamera2")
    if legacy_R is not None and legacy_t is not None:
        R = _matrix(legacy_R, (3, 3), "RotationOfCamera2").T
        t = _vector(legacy_t, "TranslationOfCamera2")
        source = "MATLAB legacy RotationOfCamera2/TranslationOfCamera2 (R transposed from postmultiply convention)"
    else:
        pose = _field(root, "PoseCamera2", "poseCamera2")
        R = _matrix(_field(pose, "R", "RotationMatrix"), (3, 3), "PoseCamera2.R")
        t = _vector(_field(pose, "Translation", "t"), "PoseCamera2.Translation")
        source = "MATLAB PoseCamera2.R/Translation"

    if abs(np.linalg.det(R) - 1.0) >= 1e-2:
        raise ValueError(f"Calibration rotation is not a proper rotation (det={np.linalg.det(R):.6f})")
    return CalibrationData(K_lg, K_ls, R, t, _image_size_hw(cam1), _image_size_hw(cam2), source)


def compute_vggt_preprocess_meta(image_paths: list[str | Path], mode: str = "crop", target_size: int = 518) -> list[dict[str, Any]]:
    """Replicate VGGT's resize, crop and within-batch centre padding exactly."""
    base: list[dict[str, Any]] = []
    for path in image_paths:
        with Image.open(path) as img:
            original_w, original_h = img.size
        if mode == "pad":
            if original_w >= original_h:
                resized_w = target_size
                resized_h = round(original_h * (resized_w / original_w) / 14) * 14
            else:
                resized_h = target_size
                resized_w = round(original_w * (resized_h / original_h) / 14) * 14
        elif mode == "crop":
            resized_w = target_size
            resized_h = round(original_h * (resized_w / original_w) / 14) * 14
        else:
            raise ValueError("mode must be 'crop' or 'pad'")
        crop_top = (resized_h - target_size) // 2 if mode == "crop" and resized_h > target_size else 0
        processed_h = target_size if mode == "pad" or resized_h > target_size else resized_h
        processed_w = target_size if mode == "pad" else resized_w
        pad_top = (target_size - processed_h) // 2 if mode == "pad" else 0
        pad_left = (target_size - processed_w) // 2 if mode == "pad" else 0
        base.append({"original_hw": (original_h, original_w), "resized_hw": (resized_h, resized_w),
                     "processed_hw": (processed_h, processed_w), "scale_xy": (resized_w / original_w, resized_h / original_h),
                     "crop_top": crop_top, "pad_top": pad_top, "pad_left": pad_left})
    max_h = max(x["processed_hw"][0] for x in base)
    max_w = max(x["processed_hw"][1] for x in base)
    for item in base:
        h, w = item["processed_hw"]
        item["batch_pad_top"] = (max_h - h) // 2
        item["batch_pad_left"] = (max_w - w) // 2
        item["final_hw"] = (max_h, max_w)
    return base


def adjust_intrinsics_to_vggt_grid(K_original: np.ndarray, meta: dict[str, Any]) -> np.ndarray:
    K = np.asarray(K_original, dtype=np.float64).copy()
    sx, sy = meta["scale_xy"]
    K[0, :] *= sx
    K[1, :] *= sy
    K[2, :] = [0.0, 0.0, 1.0]
    K[1, 2] -= meta["crop_top"]
    K[0, 2] += meta["pad_left"] + meta["batch_pad_left"]
    K[1, 2] += meta["pad_top"] + meta["batch_pad_top"]
    return K


def make_gray_only_calibrated_extrinsics() -> np.ndarray:
    return np.concatenate([np.eye(3), np.zeros((3, 1))], axis=1)[None].astype(np.float64)


def make_calibrated_extrinsics(R_lg_to_ls: np.ndarray, t_lg_to_ls: np.ndarray, scale: float) -> np.ndarray:
    E_lg = make_gray_only_calibrated_extrinsics()[0]
    E_ls = np.concatenate([np.asarray(R_lg_to_ls, dtype=np.float64), (float(scale) * np.asarray(t_lg_to_ls, dtype=np.float64)).reshape(3, 1)], axis=1)
    return np.stack([E_lg, E_ls], axis=0)


def _depth_to_camera(depth: np.ndarray, K: np.ndarray, confidence: np.ndarray | None, max_points: int, seed: int) -> np.ndarray:
    d = np.asarray(depth, dtype=np.float64).squeeze()
    if d.ndim != 2:
        raise ValueError(f"Depth must reduce to HxW, got {d.shape}")
    h, w = d.shape
    yy, xx = np.indices((h, w))
    valid = np.isfinite(d) & (d > 1e-9)
    if confidence is not None:
        c = np.asarray(confidence, dtype=np.float64).squeeze()
        if c.shape == d.shape:
            valid &= np.isfinite(c) & (c >= np.nanpercentile(c[np.isfinite(c)], 25))
    ids = np.flatnonzero(valid)
    if len(ids) < 64:
        raise RuntimeError("Too few valid VGGT depth samples to estimate calibration translation scale.")
    if len(ids) > max_points:
        ids = np.random.default_rng(seed).choice(ids, max_points, replace=False)
    z, u, v = d.reshape(-1)[ids], xx.reshape(-1)[ids], yy.reshape(-1)[ids]
    return np.column_stack(((u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z))


def estimate_translation_scale_to_vggt_depth(depth_lg: np.ndarray, depth_ls: np.ndarray, K_lg: np.ndarray, K_ls: np.ndarray,
                                               R_lg_to_ls: np.ndarray, t_lg_to_ls_physical: np.ndarray,
                                               conf_lg: np.ndarray | None = None, conf_ls: np.ndarray | None = None) -> dict[str, float]:
    """Fit the one allowed physical-to-VGGT scale from symmetric NN consistency."""
    P_lg = _depth_to_camera(depth_lg, K_lg, conf_lg, 12000, 0)
    P_ls = _depth_to_camera(depth_ls, K_ls, conf_ls, 12000, 1)
    tree_lg = cKDTree(P_lg)
    R = np.asarray(R_lg_to_ls, dtype=np.float64)
    t = _vector(t_lg_to_ls_physical, "t_lg_to_ls_physical")
    baseline = float(np.linalg.norm(t))
    if baseline <= 1e-12:
        raise ValueError("Calibration translation baseline is zero; cannot fit scale.")
    depth_scale = float(np.median(np.linalg.norm(P_lg, axis=1)))
    guess = max(depth_scale / baseline, 1e-12)

    def loss(scale: float) -> float:
        # X_lg = R.T @ (X_ls - s*t), expressed with row vectors.
        ls_in_lg = (P_ls - scale * t) @ R
        d1 = tree_lg.query(ls_in_lg, k=1, workers=-1)[0]
        tree_ls = cKDTree(ls_in_lg)
        d2 = tree_ls.query(P_lg, k=1, workers=-1)[0]
        values = np.concatenate([d1, d2])
        return float(np.median(values[np.isfinite(values)]))

    result = minimize_scalar(loss, bounds=(0.0, guess * 8.0), method="bounded", options={"xatol": max(guess * 1e-5, 1e-10)})
    if not result.success or not np.isfinite(result.x):
        raise RuntimeError(f"Translation-scale optimization failed: {result.message}")
    return {"scale_physical_translation_to_vggt": float(result.x), "symmetric_nn_median_error": float(result.fun),
            "depth_median": depth_scale, "physical_baseline": baseline, "sample_count_lg": int(len(P_lg)), "sample_count_ls": int(len(P_ls))}


def save_calibration_report(path: str | Path, calib: CalibrationData, K_lg_vggt: np.ndarray, K_ls_vggt: np.ndarray,
                            scale_info: dict[str, float], extrinsic_joint: np.ndarray, preprocess_meta: list[dict[str, Any]]) -> None:
    report = {"convention": "X_LS = R_LG_to_LS @ X_LG + t_LG_to_LS; OpenCV 0-based pixels",
              "source_description": calib.source_description, "K_lg_original": calib.K_lg.tolist(), "K_ls_original": calib.K_ls.tolist(),
              "K_lg_vggt": np.asarray(K_lg_vggt).tolist(), "K_ls_vggt": np.asarray(K_ls_vggt).tolist(),
              "R_lg_to_ls": calib.R_lg_to_ls.tolist(), "t_lg_to_ls_physical": calib.t_lg_to_ls.tolist(),
              "t_lg_to_ls_vggt_normalized": (scale_info["scale_physical_translation_to_vggt"] * calib.t_lg_to_ls).tolist(),
              "translation_scale": scale_info, "extrinsic_joint_used": np.asarray(extrinsic_joint).tolist(), "preprocess": preprocess_meta}
    Path(path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
