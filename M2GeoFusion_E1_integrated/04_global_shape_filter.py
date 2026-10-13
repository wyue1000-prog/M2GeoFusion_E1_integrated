"""Whole-cloud geometry filter: all Candidate sources share one KNN/MAD rule."""
from __future__ import annotations

import argparse
import numpy as np
from scipy.spatial import cKDTree

from common import ensure_dir, write_ascii_ply, save_json, provenance_colors
from config import SHAPE_K, SHAPE_MAD_FACTOR, SHAPE_FILTER_ITERATIONS, KNN_CHUNK_SIZE


def whole_cloud_geometry_filter(points, k=SHAPE_K, mad_factor=SHAPE_MAD_FACTOR,
                                iterations=SHAPE_FILTER_ITERATIONS, chunk_size=KNN_CHUNK_SIZE):
    """Use robust local-neighborhood support to approximate main geometry.

    Later versions may replace this with cylinder, surface, or learned models.
    There is deliberately no source-label input: Gray has no protected status.
    Scores retain original Candidate indices, storing the last evaluated score.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Candidate must be a finite (N, 3) array.")
    if k < 1 or iterations < 1 or chunk_size < 1 or not np.isfinite(mad_factor) or mad_factor < 0:
        raise ValueError("Invalid KNN/MAD filter parameters.")
    keep = np.ones(len(points), dtype=bool)
    scores = np.zeros(len(points), dtype=np.float64)
    removed_iteration = np.zeros(len(points), dtype=np.int32)
    history = []
    for iteration in range(iterations):
        ids = np.flatnonzero(keep)
        active = points[ids]
        effective_k = min(k, max(len(active) - 1, 0))
        if not effective_k:
            history.append({"iteration": iteration + 1, "total_before": len(ids),
                            "total_after": len(ids), "total_removed": 0, "effective_k": 0,
                            "median_knn_distance": 0.0, "mad_knn_distance": 0.0,
                            "robust_sigma": 0.0, "tau_shape": 0.0,
                            "note": "Fewer than two points: no neighborhood, retained."})
            break
        tree = cKDTree(active)
        for start in range(0, len(active), chunk_size):
            stop = min(start + chunk_size, len(active))
            distances, _ = tree.query(active[start:stop], k=effective_k + 1, workers=-1)
            # Drop one zero-distance self neighbor; duplicate points remain neighbors.
            scores[ids[start:stop]] = distances[:, 1:].mean(axis=1)
        values = scores[ids]
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        sigma = 1.4826 * mad
        threshold = median + mad_factor * sigma
        rejected = ids[values > threshold]
        keep[rejected] = False
        removed_iteration[rejected] = iteration + 1
        history.append({"iteration": iteration + 1, "total_before": len(ids),
                        "total_after": int(keep.sum()), "total_removed": len(rejected),
                        "effective_k": effective_k, "median_knn_distance": median,
                        "mad_knn_distance": mad, "robust_sigma": sigma, "tau_shape": threshold})
    return keep, scores, removed_iteration, history


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate-data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shape-k", type=int, default=SHAPE_K)
    ap.add_argument("--shape-mad-factor", type=float, default=SHAPE_MAD_FACTOR)
    ap.add_argument("--shape-filter-iterations", type=int, default=SHAPE_FILTER_ITERATIONS)
    ap.add_argument("--knn-chunk-size", type=int, default=KNN_CHUNK_SIZE)
    args = ap.parse_args()
    with np.load(args.candidate_data) as data:
        points, colors, labels = data["points"], data["colors"], data["source_labels"]
    if colors.shape != points.shape or labels.shape != (len(points),):
        raise ValueError("Candidate points/colors/source_labels are not aligned.")
    if not np.issubdtype(labels.dtype, np.integer) or not np.isin(labels, [0, 1, 2]).all():
        raise ValueError("Expected source labels 0=Gray, 1=XD-LG, 2=XD-LS.")
    keep, scores, removed_iteration, history = whole_cloud_geometry_filter(
        points, args.shape_k, args.shape_mad_factor, args.shape_filter_iterations, args.knn_chunk_size)
    out = ensure_dir(args.out)
    write_ascii_ply(out / "01_candidate_before_filter.ply", points, colors)
    write_ascii_ply(out / "02_final_filtered.ply", points[keep], colors[keep])
    write_ascii_ply(out / "03_removed_outliers.ply", points[~keep], colors[~keep])
    write_ascii_ply(out / "04_final_source_color.ply", points[keep], provenance_colors(labels[keep]))
    np.save(out / "shape_filter_score.npy", scores)
    np.savez_compressed(out / "final_data.npz", points=points[keep], colors=colors[keep],
                        source_labels=labels[keep], candidate_keep_mask=keep,
                        removed_iteration=removed_iteration)
    report = {
        "total_before": len(points), "total_after": int(keep.sum()),
        "total_removed": int((~keep).sum()), "removed_ratio": float((~keep).sum() / max(len(points), 1)),
        "k": args.shape_k, "shape_mad_factor": args.shape_mad_factor,
        "iterations": args.shape_filter_iterations, "knn_chunk_size": args.knn_chunk_size,
        "filter_scope": "entire Candidate, including Gray anchor; no source protected",
        "score_definition": "mean distance to k non-self neighbors; last evaluation at original Candidate index",
        "summary_statistics_scope": "last executed iteration", "iteration_reports": history,
    }
    for key in ("median_knn_distance", "mad_knn_distance", "robust_sigma", "tau_shape"):
        report[key] = history[-1][key]
    for label, prefix in ((0, "gray"), (1, "xd_lg"), (2, "xd_ls")):
        source = labels == label
        report[prefix + "_before"] = int(source.sum())
        report[prefix + "_removed"] = int((source & ~keep).sum())
        report[prefix + "_after"] = int((source & keep).sum())
    save_json(out / "shape_filter_report.json", report)
    print("[E1-J04] Whole-cloud Geometry Filter:", report)
    print(f"[E1-J04] saved: {out}")


if __name__ == "__main__":
    main()
