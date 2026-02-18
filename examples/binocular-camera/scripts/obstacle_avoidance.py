#!/usr/bin/env python3
"""
Obstacle Avoidance with Stereo Depth — tinygrad on Jetson Orin AGX.

Real-time obstacle detection for robot/drone navigation using stereo depth.
Divides the depth image into a grid of zones and computes clearance for each.

Algorithm:
  1. Capture stereo pair → rectify → compute depth map (tinygrad GPU)
  2. Divide depth map into NxM grid zones
  3. For each zone: compute distance statistics (min, median, percentile)
  4. Flag zones with obstacles closer than threshold
  5. Output clearance map + suggested heading

This is a foundation for autonomous navigation — connect the output to
motor controllers for a real robot.

Usage:
    NV=1 python3 obstacle_avoidance.py --calib calibration/stereo_calib.npz --live
    NV=1 python3 obstacle_avoidance.py --calib calibration/stereo_calib.npz --bench
"""
import argparse, os, sys, time
import numpy as np

# Grid configuration
GRID_COLS = 5   # Horizontal zones (left → right)
GRID_ROWS = 3   # Vertical zones (top, middle, bottom)

# Distance thresholds (meters)
DANGER_DIST = 0.5     # Red: immediate collision risk
WARNING_DIST = 1.5    # Yellow: slow down
CLEAR_DIST = 3.0      # Green: safe to proceed

def compute_clearance_grid(depth_map, grid_rows=GRID_ROWS, grid_cols=GRID_COLS):
    """
    Divide depth map into a grid and compute clearance statistics per zone.

    Returns:
        grid: (grid_rows, grid_cols) array of ClearanceInfo dicts
    """
    H, W = depth_map.shape
    cell_h = H // grid_rows
    cell_w = W // grid_cols

    grid = []
    for r in range(grid_rows):
        row = []
        for c in range(grid_cols):
            y1, y2 = r * cell_h, (r + 1) * cell_h
            x1, x2 = c * cell_w, (c + 1) * cell_w
            zone = depth_map[y1:y2, x1:x2]
            valid = zone[zone > 0]

            if len(valid) < 10:
                # Not enough valid depth data
                info = {"min": float('inf'), "median": float('inf'),
                        "p10": float('inf'), "density": 0.0, "status": "unknown"}
            else:
                info = {
                    "min": float(np.min(valid)),
                    "median": float(np.median(valid)),
                    "p10": float(np.percentile(valid, 10)),  # 10th percentile = closest cluster
                    "density": len(valid) / zone.size,
                    "status": "clear"
                }
                # Status based on 10th percentile (robust min)
                if info["p10"] < DANGER_DIST:
                    info["status"] = "danger"
                elif info["p10"] < WARNING_DIST:
                    info["status"] = "warning"

            row.append(info)
        grid.append(row)

    return grid

def suggest_heading(grid):
    """
    Simple heading suggestion based on clearance grid.

    Looks at the bottom two rows (ground-level obstacles) and picks
    the column with the most clearance.
    """
    cols = len(grid[0])
    rows = len(grid)

    # Weight bottom rows more (they're closer obstacles)
    col_scores = []
    for c in range(cols):
        score = 0
        for r in range(rows):
            weight = 1.0 + r  # Bottom rows weighted higher
            status = grid[r][c]["status"]
            if status == "clear":
                score += 3 * weight
            elif status == "warning":
                score += 1 * weight
            elif status == "danger":
                score -= 5 * weight
            # unknown: 0
        col_scores.append(score)

    best_col = np.argmax(col_scores)
    # Map column to heading: -1.0 (hard left) to +1.0 (hard right)
    heading = (best_col / (cols - 1)) * 2.0 - 1.0 if cols > 1 else 0.0

    labels = ["HARD LEFT", "LEFT", "STRAIGHT", "RIGHT", "HARD RIGHT"]
    if cols == 5:
        label = labels[best_col]
    else:
        label = f"COL {best_col}"

    return heading, label, col_scores

def draw_clearance_overlay(frame, grid, heading_label):
    """Draw the clearance grid overlay on the frame."""
    import cv2
    H, W = frame.shape[:2]
    rows, cols = len(grid), len(grid[0])
    cell_h, cell_w = H // rows, W // cols

    colors = {
        "clear": (0, 255, 0),      # Green
        "warning": (0, 255, 255),   # Yellow
        "danger": (0, 0, 255),      # Red
        "unknown": (128, 128, 128), # Gray
    }

    overlay = frame.copy()

    for r in range(rows):
        for c in range(cols):
            x1, y1 = c * cell_w, r * cell_h
            x2, y2 = x1 + cell_w, y1 + cell_h
            info = grid[r][c]
            color = colors.get(info["status"], (128, 128, 128))

            # Semi-transparent colored rectangle
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

            # Distance label
            dist = info["p10"]
            if dist < 100:
                dist_str = f"{dist:.1f}m"
            else:
                dist_str = "??"
            cv2.putText(overlay, dist_str, (x1 + 5, y1 + cell_h // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Blend overlay with original
    result = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

    # Draw heading suggestion
    cv2.putText(result, f"Heading: {heading_label}", (10, H - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    return result

def print_ascii_grid(grid, heading_label):
    """Print ASCII representation of clearance grid (for headless mode)."""
    symbols = {"clear": ".", "warning": "o", "danger": "X", "unknown": "?"}
    print("\n--- Clearance Grid ---")
    for row in grid:
        line = ""
        for cell in row:
            sym = symbols.get(cell["status"], "?")
            dist = cell["p10"]
            if dist < 100:
                line += f" {sym}{dist:4.1f}m"
            else:
                line += f" {sym} ??? "
        print(line)
    print(f"Heading: {heading_label}")
    print("(. = clear, o = warning, X = DANGER)")

def main():
    parser = argparse.ArgumentParser(description="Stereo obstacle avoidance")
    parser.add_argument("--calib", required=True, help="Calibration .npz file")
    parser.add_argument("--live", action="store_true", help="Live from cameras")
    parser.add_argument("--left", help="Left image")
    parser.add_argument("--right", help="Right image")
    parser.add_argument("--max-disp", type=int, default=64)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--bench", action="store_true", help="Benchmark loop timing")
    args = parser.parse_args()

    import cv2
    from depth_map import stereo_cost_volume_tinygrad, disparity_to_depth

    # Load calibration
    calib = np.load(args.calib)
    mtx_l, dist_l = calib["mtx_l"], calib["dist_l"]
    mtx_r, dist_r = calib["mtx_r"], calib["dist_r"]
    R1, R2, P1, P2, Q = calib["R1"], calib["R2"], calib["P1"], calib["P2"], calib["Q"]
    img_size = tuple(calib["img_size"])
    map1_l, map2_l = cv2.initUndistortRectifyMap(mtx_l, dist_l, R1, P1, img_size, cv2.CV_16SC2)
    map1_r, map2_r = cv2.initUndistortRectifyMap(mtx_r, dist_r, R2, P2, img_size, cv2.CV_16SC2)

    def process_frame(gray_l, gray_r):
        """Full pipeline: rectify → depth → clearance → heading."""
        rect_l = cv2.remap(gray_l, map1_l, map2_l, cv2.INTER_LINEAR)
        rect_r = cv2.remap(gray_r, map1_r, map2_r, cv2.INTER_LINEAR)
        disp = stereo_cost_volume_tinygrad(rect_l, rect_r, args.max_disp)
        depth = disparity_to_depth(disp, Q)
        grid = compute_clearance_grid(depth)
        heading, label, scores = suggest_heading(grid)
        return depth, grid, heading, label

    if args.bench:
        print("Benchmarking obstacle avoidance pipeline...")
        # Synthetic test
        gray_l = np.random.randint(0, 255, (img_size[1], img_size[0]), dtype=np.uint8)
        gray_r = np.roll(gray_l, 20, axis=1)

        # Warmup
        process_frame(gray_l, gray_r)

        times = []
        for i in range(20):
            t0 = time.time()
            depth, grid, heading, label = process_frame(gray_l, gray_r)
            times.append(time.time() - t0)
            if (i + 1) % 5 == 0:
                print(f"  Iter {i+1}/20: {times[-1]*1000:.1f}ms")

        print(f"\nPipeline timing:")
        print(f"  Mean: {np.mean(times)*1000:.1f}ms")
        print(f"  Min:  {np.min(times)*1000:.1f}ms")
        print(f"  Max:  {np.max(times)*1000:.1f}ms")
        print(f"  FPS:  {1.0/np.mean(times):.1f}")
        return

    if args.live:
        from capture_stereo import get_v4l2_capture, capture_stereo_pair
        cap_l = get_v4l2_capture(0, img_size[0], img_size[1])
        cap_r = get_v4l2_capture(1, img_size[0], img_size[1])
        print("Obstacle avoidance (press 'q' to quit)...")

        while True:
            frame_l, frame_r = capture_stereo_pair(cap_l, cap_r)
            gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

            depth, grid, heading, label = process_frame(gray_l, gray_r)

            if args.headless:
                print_ascii_grid(grid, label)
            else:
                vis = draw_clearance_overlay(frame_l, grid, label)
                from depth_map import colorize_depth
                depth_vis = colorize_depth(depth)
                combined = np.hstack([vis, depth_vis])
                cv2.imshow("Obstacle Avoidance | Depth", combined)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap_l.release()
        cap_r.release()
        cv2.destroyAllWindows()
    else:
        if not args.left or not args.right:
            print("Provide --left/--right or --live")
            sys.exit(1)

        img_l = cv2.imread(args.left, cv2.IMREAD_GRAYSCALE)
        img_r = cv2.imread(args.right, cv2.IMREAD_GRAYSCALE)
        depth, grid, heading, label = process_frame(img_l, img_r)
        print_ascii_grid(grid, label)

if __name__ == "__main__":
    main()
