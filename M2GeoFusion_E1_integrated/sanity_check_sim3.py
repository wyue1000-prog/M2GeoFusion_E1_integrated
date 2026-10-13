"""Small CPU-only synthetic check; no model, checkpoint, or experiment input."""
import numpy as np

from common import robust_umeyama_similarity
from config import SIM3_MAD_FACTOR, SIM3_ITERATIONS, SIM3_MAX_POINTS


def main():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(2000, 3))
    angle = np.deg2rad(28.0)
    expected_R = np.array([[np.cos(angle), -np.sin(angle), 0],
                           [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    expected_s, expected_t = 1.7, np.array([0.4, -0.7, 1.2])
    truth = expected_s * (src @ expected_R.T) + expected_t
    for noisy in (False, True):
        dst = truth.copy()
        outliers = np.zeros(len(src), dtype=bool)
        if noisy:
            dst += rng.normal(scale=0.001, size=dst.shape)
            outliers[rng.choice(len(src), 300, replace=False)] = True
            dst[outliers] += rng.normal(scale=4.0, size=(300, 3))
        # Non-finite correspondence indices must remain identifiable in mask.
        a = np.concatenate([src, [[np.nan, 0, 0], [0, 0, 0]]])
        b = np.concatenate([dst, [[0, 0, 0], [np.inf, 0, 0]]])
        s, R, t, mask, report = robust_umeyama_similarity(
            a, b, SIM3_MAD_FACTOR, SIM3_ITERATIONS, SIM3_MAX_POINTS)
        residual = np.linalg.norm(truth - (s * (src @ R.T) + t), axis=1)
        errors = {"scale_error": abs(s - expected_s),
                  "R_frobenius_error": float(np.linalg.norm(R - expected_R)),
                  "t_error": float(np.linalg.norm(t - expected_t)),
                  "clean_truth_median_residual": float(np.median(residual)),
                  "inliers": int(mask.sum()), "det_R": float(np.linalg.det(R))}
        tolerance = 5e-4 if noisy else 1e-10
        assert errors["scale_error"] < tolerance, errors
        assert errors["R_frobenius_error"] < tolerance, errors
        assert errors["t_error"] < tolerance, errors
        assert errors["clean_truth_median_residual"] < tolerance, errors
        assert s > 0 and np.linalg.det(R) > 0
        assert mask.shape == (len(a),) and not mask[-2:].any()
        assert report["correspondence_count"] == len(src)
        if noisy:
            assert not mask[:len(src)][outliers].any()
        print("PASS", "noise + 15% outliers" if noisy else "exact transform", errors)

    # Exercise the fixed-seed sampling path and its reproducibility.
    result1 = robust_umeyama_similarity(src, truth, max_points=200)
    result2 = robust_umeyama_similarity(src, truth, max_points=200)
    for x, y in zip(result1[:4], result2[:4]):
        np.testing.assert_array_equal(x, y)
    np.testing.assert_allclose(result1[0], expected_s, atol=1e-10)
    np.testing.assert_allclose(result1[1], expected_R, atol=1e-10)
    np.testing.assert_allclose(result1[2], expected_t, atol=1e-10)
    print("PASS fixed-seed subsampling (200 / 2000), identical repeated results")


if __name__ == "__main__":
    main()
