"""Print the parsed calibration convention before an E1 calibration-mode run."""
from __future__ import annotations

import argparse
from calibration_geometry import load_user_calibration


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mat_path")
    args = parser.parse_args()
    calibration = load_user_calibration(args.mat_path)
    print("Convention: X_LS = R_LG_to_LS @ X_LG + t_LG_to_LS (OpenCV, 0-based pixels)")
    print("Source:", calibration.source_description)
    print("K_LG:\n", calibration.K_lg)
    print("K_LS:\n", calibration.K_ls)
    print("R_LG_to_LS:\n", calibration.R_lg_to_ls)
    print("t_LG_to_LS:", calibration.t_lg_to_ls)
    print("LG image size H,W:", calibration.image_size_lg_hw)
    print("LS image size H,W:", calibration.image_size_ls_hw)


if __name__ == "__main__":
    main()
