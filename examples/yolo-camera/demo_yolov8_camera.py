#!/usr/bin/env python3
"""YOLOv8 live camera demo — TinyGrad NV=1 on Jetson AGX Orin.

Works with the MMlove USB stereo global-shutter camera (UVC).
Uses one eye from the stereo pair for object detection.

═══════════════════════════════════════════════════════════════
 SETUP (SSH into the Orin, then):
═══════════════════════════════════════════════════════════════

  cd ~/agx-orin-dev-kit/examples/yolo-camera
  nix develop

═══════════════════════════════════════════════════════════════
 STREAMING OVER SSH (view in browser on your laptop):
═══════════════════════════════════════════════════════════════

  1. On the Orin (inside nix develop):
       NV=1 python3 demo_yolov8_camera.py --stream

  2. On your laptop (new terminal / PowerShell):
       ssh -L 9999:localhost:8090 Orin-AGX-NixOS

  3. Open in browser:  http://localhost:9999/

  NOTE: NixOS firewall blocks 8090 externally, so the SSH
  tunnel is required. Port 8090 may conflict — use any free
  local port (e.g. 9999, 5555).

═══════════════════════════════════════════════════════════════
 PERFORMANCE — making inference faster:
═══════════════════════════════════════════════════════════════

  # Baseline (~1 FPS, ~1100ms/frame):
  NV=1 python3 demo_yolov8_camera.py --stream

  # JITBEAM kernel auto-tuning (tries N kernel variants, caches best):
  JITBEAM=2 NV=1 python3 demo_yolov8_camera.py --stream

  # Aggressive beam search (slower first run, fastest steady-state):
  JITBEAM=4 NV=1 python3 demo_yolov8_camera.py --stream

  # Half precision (fp16) — halves memory bandwidth, ~2x faster:
  HALF=1 JITBEAM=2 NV=1 python3 demo_yolov8_camera.py --stream

  # All-out fastest (fp16 + beam + smaller input):
  HALF=1 JITBEAM=4 NV=1 python3 demo_yolov8_camera.py --stream --size 320

═══════════════════════════════════════════════════════════════
 OTHER USAGE:
═══════════════════════════════════════════════════════════════

  # Save annotated MP4 video:
  NV=1 python3 demo_yolov8_camera.py --video demo.mp4 --frames 60

  # Headless — save annotated frames to disk:
  NV=1 python3 demo_yolov8_camera.py --headless --frames 10

  # Run on a single image:
  NV=1 python3 demo_yolov8_camera.py --image dog.jpg

  # Use right eye, larger model variant:
  NV=1 python3 demo_yolov8_camera.py --stereo right --variant s
"""
import argparse, sys, time, os, threading
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler

import cv2
import numpy as np

# tinygrad imports (PYTHONPATH set by flake shellHook)
from tinygrad import Tensor, dtypes
from tinygrad.helpers import fetch, getenv
from tinygrad.nn.state import safe_load, load_state_dict
from tinygrad.engine.jit import TinyJit

# YOLOv8 building blocks from tinygrad's examples/
from examples.yolov8 import (
    YOLOv8,
    get_variant_multiples,
    get_weights_location,
    preprocess,
    postprocess,
    scale_boxes,
)


def parse_args():
    p = argparse.ArgumentParser(description="YOLOv8 camera demo (TinyGrad NV=1)")
    p.add_argument("--device", type=int, default=0, help="V4L2 camera index")
    p.add_argument("--variant", type=str, default="n", choices=["n", "s", "m", "l", "x"],
                    help="YOLOv8 variant (n=nano, fastest)")
    p.add_argument("--width", type=int, default=2560, help="Capture width (stereo double-wide)")
    p.add_argument("--height", type=int, default=720, help="Capture height")
    p.add_argument("--fps", type=int, default=60, help="Capture FPS")
    p.add_argument("--stereo", type=str, default="left", choices=["left", "right", "none"],
                    help="Which eye to use from stereo pair")
    p.add_argument("--headless", action="store_true", help="Save frames to disk (no display)")
    p.add_argument("--frames", type=int, default=0, help="Stop after N frames (0=unlimited)")
    p.add_argument("--outdir", type=str, default="outputs", help="Output directory")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--image", type=str, default=None, help="Run on a single image instead of camera")
    p.add_argument("--video", type=str, default=None, help="Save annotated video to file (e.g. demo.mp4)")
    p.add_argument("--stream", action="store_true", help="MJPEG HTTP stream on --stream-port")
    p.add_argument("--stream-port", type=int, default=8090, help="Port for MJPEG stream (default 8090)")
    p.add_argument("--size", type=int, default=640, help="YOLO input size (320=fastest, 640=default, 416=balanced)")
    p.add_argument("--warmup", type=int, default=3, help="Warmup frames (JIT compile + BEAM search)")
    return p.parse_args()


def load_model(variant: str):
    print(f"Loading YOLOv8-{variant} weights...")
    depth, width, ratio = get_variant_multiples(variant)
    model = YOLOv8(w=width, r=ratio, d=depth, num_classes=80)
    state_dict = safe_load(get_weights_location(variant))
    load_state_dict(model, state_dict)
    print(f"YOLOv8-{variant} ready")
    return model

def make_jit_inference(model):
    """Wrap model in @TinyJit for static graph replay (HCQGraph on NV=1).

    After warmup, TinyGrad captures the full compute graph and replays it
    as a single HCQ graph submission — same mechanism as the C Hot Path.
    This eliminates per-frame scheduling/compilation overhead.
    """
    @TinyJit
    def run(x: Tensor) -> Tensor:
        return model(x)
    return run


# ---------------------------------------------------------------------------
# MJPEG HTTP streaming server (zero dependencies beyond stdlib)
# ---------------------------------------------------------------------------
_stream_frame = None
_stream_lock = threading.Lock()

class MJPEGHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/":
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while True:
                with _stream_lock:
                    jpg = _stream_frame
                if jpg is None:
                    time.sleep(0.05)
                    continue
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode())
                self.wfile.write(jpg)
                self.wfile.write(b"\r\n")
                time.sleep(0.03)  # ~30 fps cap for browser
        except (BrokenPipeError, ConnectionResetError):
            pass
    def log_message(self, fmt, *a): pass  # silence per-request logs

def start_stream_server(port: int):
    server = HTTPServer(("", port), MJPEGHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server

def update_stream_frame(frame):
    global _stream_frame
    _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    with _stream_lock:
        _stream_frame = jpg.tobytes()


COCO_LABELS = None
def get_labels():
    global COCO_LABELS
    if COCO_LABELS is None:
        COCO_LABELS = fetch(
            "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
        ).read_text().split("\n")
    return COCO_LABELS


def annotate(frame, preds, dt):
    """Draw bounding boxes and FPS on a frame."""
    labels = get_labels()
    for pred in preds:
        conf = float(pred[4])
        if conf == 0:
            continue
        x1, y1, x2, y2 = map(int, pred[:4])
        cls_id = int(pred[5])
        label = f"{labels[cls_id]} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    fps = 1.0 / max(dt, 1e-6)
    cv2.putText(frame, f"FPS: {fps:.1f}  inf: {dt*1000:.0f}ms",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return frame


def run_on_image(args):
    """Single image inference."""
    model = load_model(args.variant)
    img_path = args.image

    if img_path.startswith("http"):
        img_data = np.frombuffer(fetch(img_path).read_bytes(), np.uint8)
        img = cv2.imdecode(img_data, 1)
    else:
        img = cv2.imread(img_path)

    if img is None:
        print(f"Error: cannot load image {img_path}")
        sys.exit(1)

    pre = preprocess([img])
    t0 = time.perf_counter()
    preds = model(pre).numpy()
    dt = time.perf_counter() - t0
    preds = scale_boxes(pre.shape[2:], preds, img.shape)

    labels = get_labels()
    print(f"\nInference: {dt*1000:.0f}ms")
    detected = {}
    for pred in preds:
        if float(pred[4]) == 0:
            continue
        name = labels[int(pred[5])]
        detected[name] = detected.get(name, 0) + 1
    for name, count in detected.items():
        print(f"  {name}: {count}")

    out = annotate(img.copy(), preds, dt)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"detect_{Path(img_path).stem}.jpg"
    cv2.imwrite(str(out_path), out)
    print(f"Saved: {out_path}")


def run_camera(args):
    """Live camera loop."""
    model = load_model(args.variant)
    jit_run = make_jit_inference(model)

    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"Error: cannot open /dev/video{args.device}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera: /dev/video{args.device}  {actual_w}x{actual_h} @ {actual_fps:.0f} FPS")
    print(f"YOLO input: {args.size}x{args.size}  variant: {args.variant}")
    jitbeam = getenv("JITBEAM", 0)
    if jitbeam: print(f"JITBEAM={jitbeam} — first {args.warmup} frames will be slow (kernel tuning)")
    else: print("Tip: set JITBEAM=2 for 3-6x faster inference (slower warmup)")

    outdir = Path(args.outdir)
    if args.headless:
        outdir.mkdir(parents=True, exist_ok=True)

    # Video writer
    video_writer = None
    if args.video:
        Path(args.video).parent.mkdir(parents=True, exist_ok=True)

    # MJPEG stream
    stream_server = None
    if args.stream:
        stream_server = start_stream_server(args.stream_port)
        print(f"MJPEG stream: http://0.0.0.0:{args.stream_port}/")
        print("Open that URL in a browser on your laptop. Ctrl-C to stop.")

    frame_count = 0
    fps_smooth = 0.0

    try:
        while True:
            _ = cap.grab()  # drop buffered frame
            ret, frame = cap.read()
            if not ret:
                print("Camera read failed")
                break

            # Crop stereo pair to single eye
            if args.stereo != "none":
                h, w = frame.shape[:2]
                half = w // 2
                frame = frame[:, :half] if args.stereo == "left" else frame[:, half:]

            # Inference (uses @TinyJit — graph captured on frame 1, replayed after)
            t0 = time.perf_counter()
            pre = preprocess([frame], imgsz=args.size)
            preds = jit_run(pre).numpy()
            preds = scale_boxes(pre.shape[2:], preds, frame.shape)
            dt = time.perf_counter() - t0

            # Skip smoothing during warmup
            if frame_count < args.warmup:
                print(f"\r  warmup [{frame_count+1}/{args.warmup}] {dt*1000:.0f}ms", end="", flush=True)
                frame_count += 1
                continue
            elif frame_count == args.warmup:
                fps_smooth = 1.0 / max(dt, 1e-6)
                print(f"\r  warmup done — first real frame: {dt*1000:.0f}ms          ")
            else:
                fps_smooth = 0.9 * fps_smooth + 0.1 * (1.0 / max(dt, 1e-6))

            out = annotate(frame.copy(), preds, dt)
            frame_count += 1

            # Video output
            if args.video:
                if video_writer is None:
                    h, w = out.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(args.video, fourcc, min(fps_smooth, 30), (w, h))
                video_writer.write(out)

            # MJPEG stream
            if args.stream:
                update_stream_frame(out)

            if args.headless or args.video:
                if args.headless:
                    path = outdir / f"frame_{frame_count:05d}.jpg"
                    cv2.imwrite(str(path), out)
                print(f"\r[{frame_count}] {fps_smooth:.1f} FPS  {dt*1000:.0f}ms", end="", flush=True)
            elif not args.stream:
                cv2.imshow("YOLOv8 — TinyGrad NV=1 (q to quit)", out)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                # stream-only mode, just print progress
                print(f"\r[{frame_count}] {fps_smooth:.1f} FPS  {dt*1000:.0f}ms", end="", flush=True)

            if args.frames > 0 and frame_count >= args.frames:
                break

    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()
        if not args.headless and not args.stream and not args.video:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    print(f"\nDone — {frame_count} frames, avg {fps_smooth:.1f} FPS")
    if args.headless:
        print(f"Frames saved to: {outdir}/")
    if args.video:
        sz = os.path.getsize(args.video) / (1024 * 1024)
        print(f"Video saved: {args.video} ({sz:.1f} MB)")


def main():
    args = parse_args()
    if args.image:
        run_on_image(args)
    else:
        run_camera(args)


if __name__ == "__main__":
    main()
