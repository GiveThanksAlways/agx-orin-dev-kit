#!/usr/bin/env python3
"""
Stereo Camera Capture — Waveshare Binocular IMX219 on Jetson Orin AGX.

Captures synchronized image pairs from left and right cameras.
Supports V4L2 (direct) and GStreamer (nvarguscamerasrc) pipelines.

The Waveshare dual IMX219 module presents as two independent CSI cameras:
  - /dev/video0 (left)
  - /dev/video1 (right)

Usage:
    python3 capture_stereo.py                       # Interactive preview
    python3 capture_stereo.py --save output_dir     # Save stereo pairs
    python3 capture_stereo.py --gstreamer           # Use nvarguscamerasrc
"""
import argparse, os, sys, time
import numpy as np

def get_v4l2_capture(device_id, width=1280, height=720):
    """Open camera via V4L2 (OpenCV) backend."""
    import cv2
    cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera /dev/video{device_id}")
    return cap

def get_gstreamer_capture(sensor_id, width=1280, height=720, fps=30):
    """Open camera via GStreamer nvarguscamerasrc (hardware ISP)."""
    import cv2
    pipeline = (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM),width={width},height={height},"
        f"framerate={fps}/1,format=NV12 ! "
        f"nvvidconv ! video/x-raw,format=BGRx ! "
        f"videoconvert ! video/x-raw,format=BGR ! appsink"
    )
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open GStreamer pipeline for sensor {sensor_id}")
    return cap

def capture_stereo_pair(cap_left, cap_right):
    """Capture one synchronized stereo pair."""
    ret_l, frame_l = cap_left.read()
    ret_r, frame_r = cap_right.read()
    if not ret_l or not ret_r:
        raise RuntimeError("Failed to read from one or both cameras")
    return frame_l, frame_r

def main():
    parser = argparse.ArgumentParser(description="Stereo camera capture")
    parser.add_argument("--left", type=int, default=0, help="Left camera device ID (default: 0)")
    parser.add_argument("--right", type=int, default=1, help="Right camera device ID (default: 1)")
    parser.add_argument("--width", type=int, default=1280, help="Frame width")
    parser.add_argument("--height", type=int, default=720, help="Frame height")
    parser.add_argument("--save", type=str, default=None, help="Directory to save stereo pairs")
    parser.add_argument("--count", type=int, default=10, help="Number of pairs to save")
    parser.add_argument("--gstreamer", action="store_true", help="Use GStreamer nvarguscamerasrc")
    parser.add_argument("--headless", action="store_true", help="No display (for SSH)")
    args = parser.parse_args()

    import cv2

    print(f"Opening cameras (left={args.left}, right={args.right})...")
    if args.gstreamer:
        cap_l = get_gstreamer_capture(args.left, args.width, args.height)
        cap_r = get_gstreamer_capture(args.right, args.width, args.height)
        print("Using GStreamer nvarguscamerasrc pipeline")
    else:
        cap_l = get_v4l2_capture(args.left, args.width, args.height)
        cap_r = get_v4l2_capture(args.right, args.width, args.height)
        print("Using V4L2 direct capture")

    if args.save:
        os.makedirs(args.save, exist_ok=True)
        print(f"Saving {args.count} stereo pairs to {args.save}/")

        for i in range(args.count):
            frame_l, frame_r = capture_stereo_pair(cap_l, cap_r)
            cv2.imwrite(os.path.join(args.save, f"left_{i:04d}.png"), frame_l)
            cv2.imwrite(os.path.join(args.save, f"right_{i:04d}.png"), frame_r)
            print(f"  Saved pair {i+1}/{args.count}", flush=True)
            time.sleep(0.5)  # Half-second between captures for calibration

        print(f"Done! Saved {args.count} stereo pairs.")
    else:
        # Live preview mode
        print("Live preview (press 'q' to quit, 's' to save a pair)")
        pair_idx = 0
        fps_times = []

        while True:
            t0 = time.time()
            frame_l, frame_r = capture_stereo_pair(cap_l, cap_r)
            fps_times.append(time.time() - t0)
            if len(fps_times) > 30: fps_times.pop(0)
            fps = len(fps_times) / sum(fps_times) if fps_times else 0

            if not args.headless:
                # Side-by-side display
                combined = np.hstack([frame_l, frame_r])
                cv2.putText(combined, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Stereo Capture (Left | Right)", combined)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    os.makedirs("captures", exist_ok=True)
                    cv2.imwrite(f"captures/left_{pair_idx:04d}.png", frame_l)
                    cv2.imwrite(f"captures/right_{pair_idx:04d}.png", frame_r)
                    print(f"Saved pair #{pair_idx}")
                    pair_idx += 1
            else:
                # Headless: just print FPS
                if len(fps_times) % 30 == 0:
                    print(f"FPS: {fps:.1f}", flush=True)

    cap_l.release()
    cap_r.release()
    if not args.headless:
        import cv2
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
