#!/usr/bin/env python3
"""
3D Hand Tracking with Stereo Depth — tinygrad on Jetson Orin AGX.

Detects hand keypoints in 2D using a lightweight CNN (tinygrad) and
lifts them to 3D using stereo depth. This enables gesture recognition,
sign language, or robot teleoperation with depth awareness.

Architecture:
  - Backbone: Simple CNN (4 conv layers + 2 FC layers)
  - Output: 21 keypoints × 2 (x, y) = 42 values per hand
  - 3D: Each keypoint gets z-coordinate from stereo depth map
  - Total: 21 keypoints × 3 (x, y, z) = 63 values per hand

The 21 keypoints follow the MediaPipe hand landmark convention:
  0: wrist, 1-4: thumb, 5-8: index, 9-12: middle, 13-16: ring, 17-20: pinky

Usage:
    NV=1 python3 hand_tracking_3d.py --calib calibration/stereo_calib.npz --live
    NV=1 python3 hand_tracking_3d.py --bench  # Benchmark inference speed
"""
import argparse, os, sys, time
import numpy as np

NUM_KEYPOINTS = 21  # MediaPipe convention

KEYPOINT_NAMES = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]

# Skeleton connections for visualization
SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),# Ring
    (0, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (5, 9), (9, 13), (13, 17),             # Palm
]

def build_hand_keypoint_model(input_size=128):
    """
    Lightweight hand keypoint detector.

    Input: 128×128 RGB crop of hand region
    Output: 21 keypoints × 2 (normalized x, y coordinates)

    Architecture designed for real-time on Jetson iGPU:
    ~200K params, ~50M FLOPS → should run at 30+ FPS.
    """
    from tinygrad import Tensor
    from tinygrad.nn import Conv2d, Linear

    class HandKeypointNet:
        def __init__(self):
            # Feature extractor
            self.conv1 = Conv2d(3, 32, 3, stride=2, padding=1)    # 128→64
            self.conv2 = Conv2d(32, 64, 3, stride=2, padding=1)   # 64→32
            self.conv3 = Conv2d(64, 128, 3, stride=2, padding=1)  # 32→16
            self.conv4 = Conv2d(128, 256, 3, stride=2, padding=1) # 16→8

            # Keypoint regression head
            # 256 × 8 × 8 = 16384 → FC layers
            self.fc1 = Linear(256 * 8 * 8, 512)
            self.fc2 = Linear(512, NUM_KEYPOINTS * 2)  # 21 × 2 = 42

        def __call__(self, x):
            """x: [B, 3, 128, 128] → [B, 42] keypoint coords."""
            x = self.conv1(x).relu()
            x = self.conv2(x).relu()
            x = self.conv3(x).relu()
            x = self.conv4(x).relu()
            x = x.reshape(x.shape[0], -1)  # Flatten
            x = self.fc1(x).relu()
            x = self.fc2(x).sigmoid()  # Normalized [0, 1] coordinates
            return x

        def predict(self, x):
            """Run inference and decode to keypoint array."""
            coords = self(x).numpy()  # [B, 42]
            B = coords.shape[0]
            keypoints = coords.reshape(B, NUM_KEYPOINTS, 2)
            return keypoints  # [B, 21, 2] normalized (x, y) per keypoint

    return HandKeypointNet()

def lift_to_3d(keypoints_2d, depth_map, img_size, Q):
    """
    Lift 2D keypoints to 3D using stereo depth.

    Args:
        keypoints_2d: (21, 2) normalized coordinates
        depth_map: (H, W) depth in meters
        img_size: (W, H)
        Q: 4x4 reprojection matrix

    Returns:
        keypoints_3d: (21, 3) in meters
    """
    H, W = depth_map.shape
    keypoints_3d = np.zeros((NUM_KEYPOINTS, 3))

    fx = abs(Q[2, 3]) if Q[2, 3] != 0 else 1.0
    cx0 = -Q[0, 3]
    cy0 = -Q[1, 3]

    for i in range(NUM_KEYPOINTS):
        nx, ny = keypoints_2d[i]
        px, py = int(nx * W), int(ny * H)
        px = max(0, min(W - 1, px))
        py = max(0, min(H - 1, py))

        # Sample depth in a small neighborhood (3x3) for robustness
        r = 2
        y1, y2 = max(0, py - r), min(H, py + r + 1)
        x1, x2 = max(0, px - r), min(W, px + r + 1)
        patch = depth_map[y1:y2, x1:x2]
        valid = patch[patch > 0]

        if len(valid) > 0:
            z = float(np.median(valid))
        else:
            z = 0.0

        # Back-project to 3D
        x_3d = (px - cx0) * z / fx
        y_3d = (py - cy0) * z / fx
        keypoints_3d[i] = [x_3d, y_3d, z]

    return keypoints_3d

def compute_finger_distances(keypoints_3d):
    """Compute distances between fingertips for gesture recognition."""
    tips = {
        "thumb": keypoints_3d[4],
        "index": keypoints_3d[8],
        "middle": keypoints_3d[12],
        "ring": keypoints_3d[16],
        "pinky": keypoints_3d[20],
    }

    distances = {}
    names = list(tips.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d = np.linalg.norm(tips[names[i]] - tips[names[j]])
            distances[f"{names[i]}-{names[j]}"] = d

    return distances

def recognize_gesture(keypoints_3d):
    """
    Simple rule-based gesture recognition from 3D hand keypoints.

    Returns gesture name and confidence.
    """
    tips = keypoints_3d[[4, 8, 12, 16, 20]]  # Fingertips
    wrist = keypoints_3d[0]
    palm_center = np.mean(keypoints_3d[[0, 5, 9, 13, 17]], axis=0)

    # Finger extension: tip further from wrist than MCP joint
    mcps = keypoints_3d[[2, 5, 9, 13, 17]]
    tip_dists = np.linalg.norm(tips - wrist, axis=1)
    mcp_dists = np.linalg.norm(mcps - wrist, axis=1)
    extended = tip_dists > mcp_dists * 1.1  # 10% margin

    num_extended = np.sum(extended)

    # Gesture classification
    if num_extended == 0:
        return "fist", 0.8
    elif num_extended == 5:
        return "open_palm", 0.8
    elif num_extended == 1 and extended[1]:  # Index only
        return "pointing", 0.7
    elif num_extended == 2 and extended[1] and extended[2]:  # Index + middle
        return "peace", 0.7
    elif num_extended == 1 and extended[0]:  # Thumb only
        # Check direction
        if tips[0][1] < wrist[1]:  # Thumb above wrist
            return "thumbs_up", 0.6
        else:
            return "thumbs_down", 0.6
    elif num_extended == 3 and extended[1] and extended[2] and extended[3]:
        return "three", 0.6
    else:
        return "unknown", 0.3

def draw_hand_3d(frame, keypoints_2d, keypoints_3d, gesture, img_size):
    """Draw hand skeleton with 3D annotations on the frame."""
    import cv2
    H, W = frame.shape[:2]

    # Draw skeleton
    for i, j in SKELETON:
        pt1 = (int(keypoints_2d[i, 0] * W), int(keypoints_2d[i, 1] * H))
        pt2 = (int(keypoints_2d[j, 0] * W), int(keypoints_2d[j, 1] * H))
        cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

    # Draw keypoints with depth color
    for i in range(NUM_KEYPOINTS):
        px = int(keypoints_2d[i, 0] * W)
        py = int(keypoints_2d[i, 1] * H)
        z = keypoints_3d[i, 2]

        # Color by depth: close=red, far=blue
        if z > 0:
            t = min(z / 2.0, 1.0)  # 0-2m range
            color = (int(255 * t), 0, int(255 * (1 - t)))
        else:
            color = (128, 128, 128)

        radius = 6 if i in [4, 8, 12, 16, 20] else 4  # Bigger for fingertips
        cv2.circle(frame, (px, py), radius, color, -1)
        cv2.circle(frame, (px, py), radius, (255, 255, 255), 1)

    # Gesture label
    gesture_name, conf = gesture
    cv2.putText(frame, f"Gesture: {gesture_name} ({conf:.1f})",
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Show wrist depth
    wrist_z = keypoints_3d[0, 2]
    if wrist_z > 0:
        cv2.putText(frame, f"Wrist depth: {wrist_z:.2f}m",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    return frame

def preprocess_hand(frame, size=128):
    """Preprocess frame for hand keypoint model."""
    import cv2
    from tinygrad import Tensor
    resized = cv2.resize(frame, (size, size))
    rgb = resized[:, :, ::-1].astype(np.float32) / 255.0
    chw = np.transpose(rgb, (2, 0, 1))
    return Tensor(chw).reshape(1, 3, size, size)

def main():
    parser = argparse.ArgumentParser(description="3D Hand Tracking")
    parser.add_argument("--calib", help="Calibration .npz file")
    parser.add_argument("--live", action="store_true", help="Live from cameras")
    parser.add_argument("--bench", action="store_true", help="Benchmark inference")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    import cv2

    print("Building hand keypoint model...")
    model = build_hand_keypoint_model()
    print("Model ready (random weights — for demo/benchmark)")
    print("Note: For real tracking, train on hand datasets or load weights\n")

    if args.bench:
        from tinygrad import Tensor
        print("=== Hand Keypoint Model Benchmark ===")
        backend = "NV" if os.environ.get("NV") == "1" else \
                  "CUDA" if os.environ.get("CUDA") == "1" else "CPU"
        print(f"Backend: {backend}")

        x = Tensor.randn(1, 3, 128, 128)

        # Warmup
        for _ in range(3):
            model(x).numpy()

        times = []
        for i in range(50):
            t0 = time.time()
            kpts = model.predict(x)
            times.append(time.time() - t0)

        print(f"\nKeypoint prediction (128×128 input):")
        print(f"  Mean: {np.mean(times)*1000:.2f}ms")
        print(f"  Min:  {np.min(times)*1000:.2f}ms")
        print(f"  Std:  {np.std(times)*1000:.2f}ms")
        print(f"  FPS:  {1.0/np.mean(times):.1f}")

        # Test with batch
        for bs in [1, 2, 4]:
            x_batch = Tensor.randn(bs, 3, 128, 128)
            _ = model.predict(x_batch)  # warmup
            t0 = time.time()
            for _ in range(10):
                model.predict(x_batch)
            t_avg = (time.time() - t0) / 10
            print(f"  Batch={bs}: {t_avg*1000:.2f}ms ({bs/t_avg:.1f} hands/s)")

        # Gesture recognition benchmark
        print(f"\nGesture recognition (from keypoints):")
        fake_kpts = np.random.rand(21, 3).astype(np.float32)
        t0 = time.time()
        for _ in range(10000):
            recognize_gesture(fake_kpts)
        t_gesture = (time.time() - t0) / 10000
        print(f"  Time: {t_gesture*1e6:.1f}µs per frame")

        return

    if not args.calib:
        print("Error: --calib required for live/image mode (not needed for --bench)")
        sys.exit(1)

    calib = np.load(args.calib)
    Q = calib["Q"]
    mtx_l, dist_l = calib["mtx_l"], calib["dist_l"]
    mtx_r, dist_r = calib["mtx_r"], calib["dist_r"]
    R1, R2, P1, P2 = calib["R1"], calib["R2"], calib["P1"], calib["P2"]
    img_size = tuple(calib["img_size"])
    map1_l, map2_l = cv2.initUndistortRectifyMap(mtx_l, dist_l, R1, P1, img_size, cv2.CV_16SC2)
    map1_r, map2_r = cv2.initUndistortRectifyMap(mtx_r, dist_r, R2, P2, img_size, cv2.CV_16SC2)

    if args.live:
        from capture_stereo import get_v4l2_capture, capture_stereo_pair
        from depth_map import stereo_cost_volume_tinygrad, disparity_to_depth

        cap_l = get_v4l2_capture(0, img_size[0], img_size[1])
        cap_r = get_v4l2_capture(1, img_size[0], img_size[1])
        print("3D Hand Tracking (press 'q' to quit)...")

        while True:
            t0 = time.time()

            frame_l, frame_r = capture_stereo_pair(cap_l, cap_r)
            gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
            rect_l = cv2.remap(gray_l, map1_l, map2_l, cv2.INTER_LINEAR)
            rect_r = cv2.remap(gray_r, map1_r, map2_r, cv2.INTER_LINEAR)

            # Depth
            disp = stereo_cost_volume_tinygrad(rect_l, rect_r, 64)
            depth = disparity_to_depth(disp, Q)

            # Hand keypoints
            input_t = preprocess_hand(frame_l)
            kpts_2d = model.predict(input_t)[0]  # (21, 2)
            kpts_3d = lift_to_3d(kpts_2d, depth, img_size, Q)
            gesture = recognize_gesture(kpts_3d)

            fps = 1.0 / (time.time() - t0)

            if not args.headless:
                vis = draw_hand_3d(frame_l.copy(), kpts_2d, kpts_3d, gesture, img_size)
                cv2.putText(vis, f"FPS: {fps:.1f}", (10, vis.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("3D Hand Tracking", vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                gesture_name, conf = gesture
                print(f"Gesture: {gesture_name} ({conf:.1f}), "
                      f"Wrist: ({kpts_3d[0,0]:.2f}, {kpts_3d[0,1]:.2f}, {kpts_3d[0,2]:.2f})m, "
                      f"FPS: {fps:.1f}")

        cap_l.release()
        cap_r.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
