#!/usr/bin/env python3
"""YOLOv8 live camera demo on Jetson AGX Orin with TinyGrad.

Works with the MMlove USB stereo global-shutter camera (UVC).
Supports both the stereo pair (/dev/video0) and headless (save-to-disk) mode.

Usage:
  # Live display (needs X11/Wayland):
  python3 demo_yolov8_camera.py

  # Headless — save annotated frames to disk:
  python3 demo_yolov8_camera.py --headless --frames 30

  # Pick a different camera / variant / resolution:
  python3 demo_yolov8_camera.py --device 2 --variant s --width 1280 --height 720

  # Use the left lens only (crop left half of stereo pair):
  python3 demo_yolov8_camera.py --stereo left
"""

import argparse, sys, time
from pathlib import Path

import cv2
import numpy as np

# ---- TinyGrad imports (resolved relative to the examples/ dir) ----
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tinygrad.helpers import fetch
from tinygrad.nn.state import safe_load, load_state_dict

# Import YOLOv8 building blocks from the upstream example
from examples.yolov8 import (
    YOLOv8,
    get_variant_multiples,
    get_weights_location,
    preprocess,
    postprocess,
    scale_boxes,
    draw_bounding_boxes_and_save,
)


def parse_args():
    p = argparse.ArgumentParser(description="YOLOv8 live camera demo (TinyGrad)")
    p.add_argument("--device", type=int, default=0, help="V4L2 camera index (default: 0)")
    p.add_argument("--variant", type=str, default="n", choices=["n", "s", "m", "l", "x"],
                    help="YOLOv8 variant (default: n)")
    p.add_argument("--width", type=int, default=2560, help="Capture width (stereo double-wide, default: 2560)")
    p.add_argument("--height", type=int, default=720, help="Capture height (default: 720)")
    p.add_argument("--fps", type=int, default=60, help="Requested capture FPS")
    p.add_argument("--stereo", type=str, default="left", choices=["left", "right", "none"],
                    help="Crop left or right half of side-by-side stereo frame (default: left)")
    p.add_argument("--headless", action="store_true", help="No display — save frames to disk")
    p.add_argument("--frames", type=int, default=0,
                    help="Stop after N frames (0 = run until 'q' / Ctrl-C)")
    p.add_argument("--outdir", type=str, default="outputs_camera",
                    help="Directory for headless frame saves")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    return p.parse_args()


def open_camera(device: int, width: int, height: int, fps: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"Error: cannot open /dev/video{device}")
        sys.exit(1)
    # Request MJPEG to get high-FPS from the stereo camera
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera opened: /dev/video{device}  {actual_w}x{actual_h} @ {actual_fps:.1f} FPS")
    return cap


def crop_stereo(frame: np.ndarray, side: str) -> np.ndarray:
    """Extract left or right half of a side-by-side stereo frame."""
    h, w = frame.shape[:2]
    half = w // 2
    if side == "left":
        return frame[:, :half]
    return frame[:, half:]


def load_model(variant: str):
    depth, width, ratio = get_variant_multiples(variant)
    model = YOLOv8(w=width, r=ratio, d=depth, num_classes=80)
    state_dict = safe_load(get_weights_location(variant))
    load_state_dict(model, state_dict)
    print(f"YOLOv8-{variant} loaded")
    return model


def load_labels():
    return fetch(
        "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
    ).read_text().split("\n")


def main():
    args = parse_args()

    # ---- Model + labels ----
    model = load_model(args.variant)
    labels = load_labels()

    # ---- Camera ----
    cap = open_camera(args.device, args.width, args.height, args.fps)

    if args.headless:
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)

    frame_count = 0
    fps_smooth = 0.0
    try:
        while True:
            _ = cap.grab()  # drop oldest buffered frame
            ret, frame = cap.read()
            if not ret:
                print("Camera read failed, exiting")
                break

            if args.stereo and args.stereo != "none":
                frame = crop_stereo(frame, args.stereo)

            # ---- Inference ----
            t0 = time.perf_counter()
            pre = preprocess([frame])
            preds = model(pre).numpy()
            preds = postprocess(preds, conf_threshold=args.conf)
            preds = scale_boxes(pre.shape[2:], preds, frame.shape)
            dt = time.perf_counter() - t0
            fps_smooth = 0.9 * fps_smooth + 0.1 * (1.0 / max(dt, 1e-6))

            # ---- Annotate ----
            annotated = frame.copy()
            for pred in preds:
                conf = float(pred[4])
                if conf == 0:
                    continue
                x1, y1, x2, y2 = map(int, pred[:4])
                cls_id = int(pred[5])
                label = f"{labels[cls_id]} {conf:.2f}"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, label, (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.putText(annotated, f"FPS: {fps_smooth:.1f}  inf: {dt*1000:.0f}ms",
                        (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            frame_count += 1

            if args.headless:
                out_path = outdir / f"frame_{frame_count:05d}.jpg"
                cv2.imwrite(str(out_path), annotated)
                print(f"\r[{frame_count}] {fps_smooth:.1f} FPS  {dt*1000:.0f}ms", end="", flush=True)
            else:
                cv2.imshow("YOLOv8 — TinyGrad (q to quit)", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if args.frames > 0 and frame_count >= args.frames:
                break

    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()

    print(f"\nDone — {frame_count} frames, avg {fps_smooth:.1f} FPS")


if __name__ == "__main__":
    main()
