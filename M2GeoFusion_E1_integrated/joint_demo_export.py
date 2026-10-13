"""Original Joint Demo diagnostic, independent of either filtering pipeline."""

import numpy as np

from common import write_ascii_ply


def export_joint_demo_ply(joint_npz, out_path):
    """Export the official Demo's depth-and-camera point cloud as ASCII PLY.

    This deliberately has no connection to E1's Need-mask, alignment, or
    de-duplication path. It reproduces visual_util.predictions_to_glb() with
    the web Demo defaults: Depthmap and Camera Branch, all frames, and a 50th
    percentile confidence threshold.
    """
    joint = np.load(joint_npz)
    required = ("world_points_from_depth", "depth_conf", "images")
    missing = [key for key in required if key not in joint]
    if missing:
        raise KeyError(
            f"Joint NPZ is missing {missing}; rerun 02 so it saves the "
            "Depth Head confidence and preprocessed RGB images."
        )

    point_maps = np.asarray(joint["world_points_from_depth"], dtype=np.float32)
    depth_conf = np.asarray(joint["depth_conf"], dtype=np.float32)
    images = np.asarray(joint["images"])
    if point_maps.ndim != 4 or point_maps.shape[-1] != 3 or point_maps.shape[0] != 2:
        raise ValueError(
            "Expected Joint [LG, LS] world_points_from_depth with shape "
            f"(2, H, W, 3), got {point_maps.shape}."
        )
    if depth_conf.shape == point_maps.shape[:-1] + (1,):
        depth_conf = depth_conf[..., 0]
    if depth_conf.shape != point_maps.shape[:-1]:
        raise ValueError(
            f"depth_conf shape {depth_conf.shape} does not match point maps {point_maps.shape[:-1]}."
        )
    if images.ndim == 3 and images.shape[0] == 2:  # (S, H, W) grayscale
        rgb = np.repeat(images[..., None], 3, axis=-1)
    elif images.ndim != 4 or images.shape[0] != 2:
        raise ValueError(f"Expected Joint [LG, LS] image tensor, got {images.shape}.")
    elif images.shape[1] == 3:  # Official Demo accepts NCHW here.
        rgb = np.transpose(images, (0, 2, 3, 1))
    elif images.shape[1] == 1:  # (S, 1, H, W) grayscale
        rgb = np.repeat(np.transpose(images, (0, 2, 3, 1)), 3, axis=-1)
    elif images.shape[-1] == 3:
        rgb = images
    elif images.shape[-1] == 1:  # (S, H, W, 1) grayscale
        rgb = np.repeat(images, 3, axis=-1)
    else:
        raise ValueError(f"Expected RGB Joint images, got {images.shape}.")
    if rgb.shape[:3] != point_maps.shape[:3]:
        raise ValueError(f"Image shape {rgb.shape} does not match point maps {point_maps.shape}.")

    # Exact official filtering: flatten all selected frames, calculate the
    # percentile globally, then retain >= threshold and confidence > 1e-5.
    flat_conf = depth_conf.reshape(-1)
    conf_threshold = float(np.percentile(flat_conf, 50.0))
    conf_mask = (flat_conf >= conf_threshold) & (flat_conf > 1e-5)
    flat_points = point_maps.reshape(-1, 3)
    flat_colors = np.clip(rgb.reshape(-1, 3) * 255.0, 0, 255).astype(np.uint8)

    # write_ascii_ply also rejects non-finite XYZ, as a PLY cannot represent
    # those points. This is the only additional validity constraint.
    finite_mask = np.isfinite(flat_points).all(axis=1)
    keep = conf_mask & finite_mask
    frame_pixels = point_maps.shape[1] * point_maps.shape[2]
    lg_count = int(np.count_nonzero(keep[:frame_pixels]))
    ls_count = int(np.count_nonzero(keep[frame_pixels:]))
    write_ascii_ply(out_path, flat_points[keep], flat_colors[keep])
    return lg_count, ls_count, int(np.count_nonzero(keep)), conf_threshold
