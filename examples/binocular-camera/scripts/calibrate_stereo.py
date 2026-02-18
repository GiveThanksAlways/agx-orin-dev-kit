#!/usr/bin/env python3
"""
Stereo Camera Calibration — Waveshare Binocular IMX219 on Jetson Orin AGX.

Uses a checkerboard pattern to compute:
  1. Intrinsic parameters (focal length, principal point, distortion) per camera
  2. Extrinsic parameters (rotation & translation between cameras)
  3. Rectification transforms for epipolar-aligned stereo

The calibration data is saved as a NumPy .npz file for use by depth_map.py
and other stereo scripts.

Usage:
    # First capture calibration images:
    python3 capture_stereo.py --save calibration/images --count 30

    # Then calibrate:
    python3 calibrate_stereo.py calibration/images --output calibration/stereo_calib.npz

    # Verify with rectified preview:
    python3 calibrate_stereo.py calibration/images --verify
"""
import argparse, glob, os, sys
import numpy as np

def find_checkerboard_corners(images, board_size, square_size_mm=25.0):
    """Find checkerboard corners in a list of images.

    Args:
        images: list of image paths
        board_size: (cols, rows) inner corners of checkerboard
        square_size_mm: physical size of each square in mm

    Returns:
        obj_points: list of 3D world points
        img_points: list of 2D image points
        img_size: (width, height)
    """
    import cv2

    # Prepare object points: (0,0,0), (1,0,0), ... scaled by square_size
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= square_size_mm

    obj_points = []
    img_points = []
    img_size = None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for img_path in images:
        img = cv2.imread(img_path)
        if img is None:
            print(f"  Warning: cannot read {img_path}, skipping")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_size is None:
            img_size = (gray.shape[1], gray.shape[0])

        ret, corners = cv2.findChessboardCorners(gray, board_size, None)
        if ret:
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            obj_points.append(objp)
            img_points.append(corners_refined)
        else:
            print(f"  No checkerboard found in {os.path.basename(img_path)}")

    return obj_points, img_points, img_size

def calibrate_single_camera(obj_points, img_points, img_size):
    """Calibrate a single camera. Returns camera_matrix, dist_coeffs, rvecs, tvecs."""
    import cv2
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, img_size, None, None
    )
    print(f"  Single camera RMS reprojection error: {ret:.4f}")
    return mtx, dist, rvecs, tvecs

def calibrate_stereo(obj_points_l, img_points_l, obj_points_r, img_points_r,
                     mtx_l, dist_l, mtx_r, dist_r, img_size):
    """Full stereo calibration. Returns R, T, E, F."""
    import cv2

    # Only use images where both cameras found the checkerboard
    # (We assume they're already matched by index)
    assert len(obj_points_l) == len(obj_points_r), \
        f"Mismatch: {len(obj_points_l)} left vs {len(obj_points_r)} right"

    flags = (cv2.CALIB_FIX_INTRINSIC)  # Use pre-computed intrinsics

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

    ret, mtx_l, dist_l, mtx_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
        obj_points_l, img_points_l, img_points_r,
        mtx_l, dist_l, mtx_r, dist_r,
        img_size, criteria=criteria, flags=flags
    )
    print(f"  Stereo RMS reprojection error: {ret:.4f}")
    return R, T, E, F

def compute_rectification(mtx_l, dist_l, mtx_r, dist_r, img_size, R, T):
    """Compute rectification transforms for aligned stereo."""
    import cv2
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        mtx_l, dist_l, mtx_r, dist_r, img_size, R, T,
        alpha=0.0  # 0=crop all invalid pixels, 1=keep all pixels
    )
    return R1, R2, P1, P2, Q, roi1, roi2

def main():
    parser = argparse.ArgumentParser(description="Stereo camera calibration")
    parser.add_argument("image_dir", help="Directory with left_NNNN.png and right_NNNN.png")
    parser.add_argument("--output", "-o", default="calibration/stereo_calib.npz",
                       help="Output calibration file")
    parser.add_argument("--board-cols", type=int, default=9,
                       help="Checkerboard inner corners (columns)")
    parser.add_argument("--board-rows", type=int, default=6,
                       help="Checkerboard inner corners (rows)")
    parser.add_argument("--square-size", type=float, default=25.0,
                       help="Square size in mm")
    parser.add_argument("--verify", action="store_true",
                       help="Show rectified stereo pair after calibration")
    args = parser.parse_args()

    import cv2

    board_size = (args.board_cols, args.board_rows)

    # Find matching stereo pairs
    left_images = sorted(glob.glob(os.path.join(args.image_dir, "left_*.png")))
    right_images = sorted(glob.glob(os.path.join(args.image_dir, "right_*.png")))

    if not left_images or not right_images:
        print(f"Error: No stereo pairs found in {args.image_dir}")
        print("Expected files: left_0000.png, right_0000.png, ...")
        sys.exit(1)

    # Match pairs by index
    left_idxs = {os.path.basename(p).replace("left_", "").replace(".png", ""): p
                 for p in left_images}
    right_idxs = {os.path.basename(p).replace("right_", "").replace(".png", ""): p
                  for p in right_images}
    common = sorted(set(left_idxs.keys()) & set(right_idxs.keys()))
    print(f"Found {len(common)} matched stereo pairs")

    if len(common) < 10:
        print("Warning: At least 15-20 pairs recommended for good calibration")

    # Process left images
    print("\nCalibrating LEFT camera...")
    left_matched = [left_idxs[k] for k in common]
    obj_l, img_l, img_size = find_checkerboard_corners(
        left_matched, board_size, args.square_size)
    print(f"  Found checkerboard in {len(obj_l)}/{len(common)} images")

    # Process right images
    print("\nCalibrating RIGHT camera...")
    right_matched = [right_idxs[k] for k in common]
    obj_r, img_r, _ = find_checkerboard_corners(
        right_matched, board_size, args.square_size)
    print(f"  Found checkerboard in {len(obj_r)}/{len(common)} images")

    # Single camera calibration
    print("\nSingle camera calibration (LEFT)...")
    mtx_l, dist_l, _, _ = calibrate_single_camera(obj_l, img_l, img_size)
    print(f"  Focal length: fx={mtx_l[0,0]:.1f}, fy={mtx_l[1,1]:.1f}")
    print(f"  Principal point: cx={mtx_l[0,2]:.1f}, cy={mtx_l[1,2]:.1f}")

    print("\nSingle camera calibration (RIGHT)...")
    mtx_r, dist_r, _, _ = calibrate_single_camera(obj_r, img_r, img_size)
    print(f"  Focal length: fx={mtx_r[0,0]:.1f}, fy={mtx_r[1,1]:.1f}")
    print(f"  Principal point: cx={mtx_r[0,2]:.1f}, cy={mtx_r[1,2]:.1f}")

    # Stereo calibration
    print("\nStereo calibration...")
    R, T, E, F = calibrate_stereo(obj_l, img_l, obj_r, img_r,
                                   mtx_l, dist_l, mtx_r, dist_r, img_size)
    baseline_mm = np.linalg.norm(T)
    print(f"  Baseline: {baseline_mm:.1f} mm")
    print(f"  Translation: [{T[0,0]:.1f}, {T[1,0]:.1f}, {T[2,0]:.1f}] mm")

    # Rectification
    print("\nComputing rectification transforms...")
    R1, R2, P1, P2, Q, roi1, roi2 = compute_rectification(
        mtx_l, dist_l, mtx_r, dist_r, img_size, R, T)

    # Save calibration
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.savez(args.output,
             mtx_l=mtx_l, dist_l=dist_l,
             mtx_r=mtx_r, dist_r=dist_r,
             R=R, T=T, E=E, F=F,
             R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
             roi1=roi1, roi2=roi2,
             img_size=img_size, baseline_mm=baseline_mm)
    print(f"\nCalibration saved to {args.output}")
    print(f"  Image size: {img_size[0]}x{img_size[1]}")
    print(f"  Baseline: {baseline_mm:.1f} mm")

    # Verify with rectified display
    if args.verify:
        print("\nShowing rectified stereo pair (press any key to close)...")
        calib = np.load(args.output)

        # Build undistortion maps
        map1_l, map2_l = cv2.initUndistortRectifyMap(
            mtx_l, dist_l, R1, P1, img_size, cv2.CV_16SC2)
        map1_r, map2_r = cv2.initUndistortRectifyMap(
            mtx_r, dist_r, R2, P2, img_size, cv2.CV_16SC2)

        # Use first pair for demo
        img_l = cv2.imread(left_matched[0])
        img_r = cv2.imread(right_matched[0])
        rect_l = cv2.remap(img_l, map1_l, map2_l, cv2.INTER_LINEAR)
        rect_r = cv2.remap(img_r, map1_r, map2_r, cv2.INTER_LINEAR)

        # Draw horizontal lines to verify alignment
        combined = np.hstack([rect_l, rect_r])
        for y in range(0, combined.shape[0], 40):
            cv2.line(combined, (0, y), (combined.shape[1], y), (0, 255, 0), 1)

        cv2.imshow("Rectified Stereo (epipolar lines should be horizontal)", combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
