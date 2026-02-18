#!/usr/bin/env python3
"""
GPU-Accelerated Depth Map — tinygrad stereo matching on Jetson Orin AGX.

Computes dense depth maps from rectified stereo image pairs using:
  1. OpenCV's StereoSGBM as baseline
  2. A tinygrad-based cost volume approach (GPU-accelerated via NV backend)

The tinygrad approach builds a disparity cost volume and performs
winner-take-all (WTA) matching — this is the core of all stereo methods
and runs naturally on the GPU tensor engine.

Usage:
    python3 depth_map.py --calib calibration/stereo_calib.npz \\
                         --left captures/left_0000.png \\
                         --right captures/right_0000.png

    # Live stereo depth from cameras
    python3 depth_map.py --calib calibration/stereo_calib.npz --live

    # Benchmark tinygrad vs OpenCV
    python3 depth_map.py --calib calibration/stereo_calib.npz --bench
"""
import argparse, os, sys, time
import numpy as np

# ---------------------------------------------------------------------------
# OpenCV baseline
# ---------------------------------------------------------------------------
def stereo_sgbm(left_gray, right_gray, num_disparities=128, block_size=5):
    """OpenCV Semi-Global Block Matching (CPU)."""
    import cv2
    sgbm = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * block_size * block_size,
        P2=32 * block_size * block_size,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    disparity = sgbm.compute(left_gray, right_gray).astype(np.float32) / 16.0
    return disparity

# ---------------------------------------------------------------------------
# Tinygrad GPU stereo matching
# ---------------------------------------------------------------------------
def stereo_cost_volume_tinygrad(left_gray, right_gray, max_disp=128, block_radius=2):
    """
    GPU-accelerated stereo matching via tinygrad.

    Method: Builds a 3D cost volume C[d, y, x] = SAD(left, right_shifted_by_d)
    over a local window, then selects the disparity with minimum cost (WTA).

    This is a classic block matching algorithm but runs entirely on the GPU
    using tinygrad's tensor operations.
    """
    from tinygrad import Tensor, dtypes

    H, W = left_gray.shape

    # Upload images to GPU as float tensors
    left_t = Tensor(left_gray.astype(np.float32)).reshape(1, 1, H, W)
    right_t = Tensor(right_gray.astype(np.float32)).reshape(1, 1, H, W)

    # Box filter kernel for local aggregation
    ks = 2 * block_radius + 1
    box_kernel = Tensor.ones(1, 1, ks, ks) / (ks * ks)

    # Build cost volume: for each disparity d, shift right image left by d pixels
    # and compute Sum of Absolute Differences (SAD) in a local window
    costs = []
    for d in range(max_disp):
        if d == 0:
            diff = (left_t - right_t).abs()
        else:
            # Shift right image by d pixels (left shift = look further right)
            right_shifted = right_t[:, :, :, d:]
            left_cropped = left_t[:, :, :, :W - d]
            diff = (left_cropped - right_shifted).abs()
            # Pad to original width (fill with large cost)
            pad = Tensor.ones(1, 1, H, d) * 255.0
            diff = diff.cat(pad, dim=3)

        # Local aggregation via box filter (average pooling approximation)
        # Using conv2d with a uniform kernel = box filter
        cost = diff.pad2d((block_radius, block_radius, block_radius, block_radius),
                          mode="constant", value=255.0)
        cost = cost.conv2d(box_kernel, padding=0)
        costs.append(cost)

    # Stack into cost volume [max_disp, H, W] and find min-cost disparity
    cost_volume = Tensor.stack(*costs, dim=0).reshape(max_disp, H, W)
    disparity = cost_volume.argmin(axis=0).cast(dtypes.float32)

    return disparity.numpy()

def disparity_to_depth(disparity, Q):
    """Convert disparity map to 3D depth using the Q (reprojection) matrix.

    Q is from cv2.stereoRectify:
        [X, Y, Z, W]^T = Q * [x, y, disp, 1]^T
        depth = Z/W = -focal*baseline / (disp - cx_diff)

    Args:
        disparity: HxW disparity map
        Q: 4x4 reprojection matrix from calibration

    Returns:
        depth_m: HxW depth in meters
    """
    import cv2
    points_3d = cv2.reprojectImageTo3D(disparity, Q)
    depth = points_3d[:, :, 2]
    # Clean up: invalid disparities give nonsensical depth
    depth[disparity <= 0] = 0
    depth[depth > 50.0] = 0  # Max 50m
    depth[depth < 0] = 0
    return depth

def colorize_depth(depth, max_depth=5.0):
    """Create a colorized depth visualization."""
    import cv2
    depth_clipped = np.clip(depth, 0, max_depth)
    depth_norm = (depth_clipped / max_depth * 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
    # Make invalid pixels black
    depth_color[depth <= 0] = 0
    return depth_color

def main():
    parser = argparse.ArgumentParser(description="Stereo depth estimation")
    parser.add_argument("--calib", required=True, help="Calibration .npz file")
    parser.add_argument("--left", help="Left image path")
    parser.add_argument("--right", help="Right image path")
    parser.add_argument("--live", action="store_true", help="Live depth from cameras")
    parser.add_argument("--bench", action="store_true", help="Benchmark tinygrad vs OpenCV")
    parser.add_argument("--max-disp", type=int, default=64, help="Max disparity (default: 64)")
    parser.add_argument("--method", choices=["sgbm", "tinygrad", "both"], default="both")
    parser.add_argument("--headless", action="store_true", help="No display")
    args = parser.parse_args()

    import cv2

    # Load calibration
    calib = np.load(args.calib)
    mtx_l, dist_l = calib["mtx_l"], calib["dist_l"]
    mtx_r, dist_r = calib["mtx_r"], calib["dist_r"]
    R1, R2 = calib["R1"], calib["R2"]
    P1, P2 = calib["P1"], calib["P2"]
    Q = calib["Q"]
    img_size = tuple(calib["img_size"])

    # Build undistortion + rectification maps
    map1_l, map2_l = cv2.initUndistortRectifyMap(mtx_l, dist_l, R1, P1, img_size, cv2.CV_16SC2)
    map1_r, map2_r = cv2.initUndistortRectifyMap(mtx_r, dist_r, R2, P2, img_size, cv2.CV_16SC2)

    print(f"Calibration loaded: {img_size[0]}x{img_size[1]}, "
          f"baseline={float(calib['baseline_mm']):.1f}mm")

    if args.bench:
        print(f"\n=== Benchmark: max_disp={args.max_disp} ===")
        # Create synthetic stereo pair for benchmarking
        if args.left and args.right:
            img_l = cv2.imread(args.left, cv2.IMREAD_GRAYSCALE)
            img_r = cv2.imread(args.right, cv2.IMREAD_GRAYSCALE)
            rect_l = cv2.remap(img_l, map1_l, map2_l, cv2.INTER_LINEAR)
            rect_r = cv2.remap(img_r, map1_r, map2_r, cv2.INTER_LINEAR)
        else:
            print("Using synthetic test images (random texture)")
            rect_l = np.random.randint(0, 255, (img_size[1], img_size[0]), dtype=np.uint8)
            rect_r = np.roll(rect_l, 20, axis=1)  # 20px disparity

        # OpenCV SGBM
        print("\nOpenCV SGBM (CPU):")
        times = []
        for i in range(5):
            t0 = time.time()
            disp_sgbm = stereo_sgbm(rect_l, rect_r, args.max_disp)
            times.append(time.time() - t0)
        print(f"  Time: {np.mean(times)*1000:.1f} ± {np.std(times)*1000:.1f} ms")
        print(f"  FPS:  {1.0/np.mean(times):.1f}")

        # Tinygrad GPU
        backend = "NV" if os.environ.get("NV") == "1" else "CUDA" if os.environ.get("CUDA") == "1" else "CPU"
        print(f"\nTinygrad ({backend}):")
        # Warmup
        _ = stereo_cost_volume_tinygrad(rect_l, rect_r, args.max_disp)
        times = []
        for i in range(5):
            t0 = time.time()
            disp_tg = stereo_cost_volume_tinygrad(rect_l, rect_r, args.max_disp)
            times.append(time.time() - t0)
        print(f"  Time: {np.mean(times)*1000:.1f} ± {np.std(times)*1000:.1f} ms")
        print(f"  FPS:  {1.0/np.mean(times):.1f}")

        return

    if args.live:
        from capture_stereo import get_v4l2_capture, capture_stereo_pair
        cap_l = get_v4l2_capture(0, img_size[0], img_size[1])
        cap_r = get_v4l2_capture(1, img_size[0], img_size[1])
        print("Live depth estimation (press 'q' to quit)...")

        while True:
            frame_l, frame_r = capture_stereo_pair(cap_l, cap_r)
            gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
            rect_l = cv2.remap(gray_l, map1_l, map2_l, cv2.INTER_LINEAR)
            rect_r = cv2.remap(gray_r, map1_r, map2_r, cv2.INTER_LINEAR)

            if args.method in ("tinygrad", "both"):
                disp = stereo_cost_volume_tinygrad(rect_l, rect_r, args.max_disp)
            else:
                disp = stereo_sgbm(rect_l, rect_r, args.max_disp)

            depth = disparity_to_depth(disp, Q)
            depth_vis = colorize_depth(depth)

            if not args.headless:
                combined = np.hstack([cv2.cvtColor(rect_l, cv2.COLOR_GRAY2BGR), depth_vis])
                cv2.imshow("Left | Depth", combined)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap_l.release()
        cap_r.release()
        cv2.destroyAllWindows()
    else:
        # Single pair mode
        if not args.left or not args.right:
            print("Error: Provide --left and --right images, or use --live")
            sys.exit(1)

        img_l = cv2.imread(args.left, cv2.IMREAD_GRAYSCALE)
        img_r = cv2.imread(args.right, cv2.IMREAD_GRAYSCALE)
        rect_l = cv2.remap(img_l, map1_l, map2_l, cv2.INTER_LINEAR)
        rect_r = cv2.remap(img_r, map1_r, map2_r, cv2.INTER_LINEAR)

        if args.method in ("sgbm", "both"):
            t0 = time.time()
            disp_sgbm = stereo_sgbm(rect_l, rect_r, args.max_disp)
            print(f"SGBM: {(time.time()-t0)*1000:.1f}ms")
            depth_sgbm = disparity_to_depth(disp_sgbm, Q)
            cv2.imwrite("depth_sgbm.png", colorize_depth(depth_sgbm))
            print("Saved depth_sgbm.png")

        if args.method in ("tinygrad", "both"):
            t0 = time.time()
            disp_tg = stereo_cost_volume_tinygrad(rect_l, rect_r, args.max_disp)
            print(f"Tinygrad: {(time.time()-t0)*1000:.1f}ms")
            depth_tg = disparity_to_depth(disp_tg, Q)
            cv2.imwrite("depth_tinygrad.png", colorize_depth(depth_tg))
            print("Saved depth_tinygrad.png")

if __name__ == "__main__":
    main()
