#!/usr/bin/env python3
"""YOLOv8 live camera demo — PyTorch CUDA Graphs on Jetson AGX Orin.

Uses a pure-PyTorch YOLOv8-n port with CUDA Graphs for zero-overhead
GPU replay. Loads the same tinygrad safetensor weights.

Benchmark results (YOLOv8-n @ 320x320, Orin AGX 64GB MAXN):
  - CUDA Graphs:   6.3 ms / 158 FPS
  - torch.compile: 12.6 ms / 79 FPS
  - Eager:         23.2 ms / 43 FPS

Usage:
  cd ~/agx-orin-dev-kit/examples/yolo-camera && nix develop
  python3 demo_yolov8_pytorch.py --stream

SSH tunnel (view in browser on laptop):
  ssh -L 9999:localhost:8090 Orin-AGX-NixOS
  open http://localhost:9999/
"""
import argparse, sys, time, os, threading
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler

import cv2
import numpy as np
import torch
import torch.nn as nn

from bench_yolov8_pytorch import YOLOv8PyTorch, load_tinygrad_weights_into_pytorch


# ═══════════════════════════════════════════════════════════════════════════════
# MJPEG HTTP streaming server (reused from tinygrad demo)
# ═══════════════════════════════════════════════════════════════════════════════

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
                time.sleep(0.03)
        except (BrokenPipeError, ConnectionResetError):
            pass
    def log_message(self, fmt, *a): pass

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


# ═══════════════════════════════════════════════════════════════════════════════
# COCO labels
# ═══════════════════════════════════════════════════════════════════════════════

COCO_LABELS = None
def get_labels():
    global COCO_LABELS
    if COCO_LABELS is None:
        sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), '../../external/tinygrad')))
        from tinygrad.helpers import fetch
        COCO_LABELS = fetch(
            "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
        ).read_text().split("\n")
    return COCO_LABELS


# ═══════════════════════════════════════════════════════════════════════════════
# Preprocessing / Postprocessing
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess_frame(frame, imgsz=320):
    """Resize + pad (letterbox) + normalize → (1,3,H,W) float32 tensor."""
    h, w = frame.shape[:2]
    scale = min(imgsz / h, imgsz / w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    top, left = (imgsz - nh) // 2, (imgsz - nw) // 2
    canvas[top:top+nh, left:left+nw] = resized
    blob = canvas.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[np.newaxis]  # (1,3,H,W)
    return blob, scale, top, left


def postprocess_pytorch(output, conf_thresh=0.25, iou_thresh=0.45):
    """Decode YOLOv8 output → (N, 6) boxes [x1,y1,x2,y2,conf,cls]."""
    # output: (1, 4+nc, num_anchors)
    out = output[0].T  # (num_anchors, 4+nc)
    boxes_xywh = out[:, :4]
    scores = out[:, 4:]
    max_scores, class_ids = scores.max(dim=1)

    mask = max_scores > conf_thresh
    boxes_xywh = boxes_xywh[mask]
    max_scores = max_scores[mask]
    class_ids = class_ids[mask]

    if len(boxes_xywh) == 0:
        return np.zeros((0, 6), dtype=np.float32)

    # xywh → xyxy
    x1 = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    y1 = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    x2 = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    y2 = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2
    boxes_xyxy = torch.stack((x1, y1, x2, y2), dim=1)

    # NMS (pure-torch fallback — avoids torchvision source build on Jetson)
    try:
        from torchvision.ops import nms
    except ImportError:
        def nms(boxes, scores, iou_threshold):
            order = scores.argsort(descending=True)
            keep = []
            while order.numel() > 0:
                i = order[0].item()
                keep.append(i)
                if order.numel() == 1:
                    break
                rest = order[1:]
                xx1 = torch.clamp(boxes[rest, 0], min=boxes[i, 0].item())
                yy1 = torch.clamp(boxes[rest, 1], min=boxes[i, 1].item())
                xx2 = torch.clamp(boxes[rest, 2], max=boxes[i, 2].item())
                yy2 = torch.clamp(boxes[rest, 3], max=boxes[i, 3].item())
                inter = torch.clamp(xx2 - xx1, min=0) * torch.clamp(yy2 - yy1, min=0)
                areas_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
                areas_rest = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
                iou = inter / (areas_i + areas_rest - inter + 1e-6)
                order = rest[iou <= iou_threshold]
            return torch.tensor(keep, dtype=torch.long, device=boxes.device)
    keep = nms(boxes_xyxy, max_scores, iou_thresh)
    keep = keep[:300]

    result = torch.cat((
        boxes_xyxy[keep],
        max_scores[keep].unsqueeze(1),
        class_ids[keep].float().unsqueeze(1),
    ), dim=1)
    return result.cpu().numpy()


def scale_boxes_to_frame(preds, imgsz, scale, top, left, frame_shape):
    """Scale detection boxes back to original frame coordinates."""
    if len(preds) == 0:
        return preds
    preds = preds.copy()
    preds[:, 0] = (preds[:, 0] - left) / scale
    preds[:, 1] = (preds[:, 1] - top) / scale
    preds[:, 2] = (preds[:, 2] - left) / scale
    preds[:, 3] = (preds[:, 3] - top) / scale
    # Clip
    h, w = frame_shape[:2]
    preds[:, 0] = np.clip(preds[:, 0], 0, w)
    preds[:, 1] = np.clip(preds[:, 1], 0, h)
    preds[:, 2] = np.clip(preds[:, 2], 0, w)
    preds[:, 3] = np.clip(preds[:, 3], 0, h)
    return preds


def annotate(frame, preds, dt, backend="PyTorch CUDA Graphs"):
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
    cv2.putText(frame, f"{backend}  FPS: {fps:.1f}  inf: {dt*1000:.0f}ms",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return frame


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 camera demo (PyTorch CUDA Graphs)")
    parser.add_argument("--device", type=str, default="/dev/video0", help="Camera device")
    parser.add_argument("--variant", type=str, default="n")
    parser.add_argument("--size", type=int, default=320)
    parser.add_argument("--width", type=int, default=2560)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--stereo", type=str, default="left", choices=["left", "right", "none"])
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--stream-port", type=int, default=8090)
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--bench", action="store_true", help="Benchmark mode (no camera)")
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    # Load model
    print(f"Loading YOLOv8-{args.variant} (PyTorch)...")
    w_map = {'n': (0.33, 0.25, 2.0), 's': (0.33, 0.50, 2.0), 'm': (0.67, 0.75, 1.5)}
    d, w, r = w_map[args.variant]
    model = YOLOv8PyTorch(w=w, r=r, d=d, nc=80).cuda().eval()
    model = load_tinygrad_weights_into_pytorch(model, args.variant)

    # Build CUDA Graph
    print("  Capturing CUDA Graph...")
    x_static = torch.zeros(1, 3, args.size, args.size, device='cuda', dtype=torch.float32)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.no_grad(), torch.cuda.stream(s):
        for _ in range(args.warmup):
            _ = model(x_static)
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.cuda.graph(g):
        out_static = model(x_static)
    print("  CUDA Graph captured.")

    # Benchmark mode
    if args.bench:
        print(f"\nBenchmark: 200 iterations @ {args.size}x{args.size}...")
        times = []
        for _ in range(200):
            torch.cuda.synchronize()
            t0 = time.perf_counter_ns()
            g.replay()
            torch.cuda.synchronize()
            times.append(time.perf_counter_ns() - t0)
        a = np.array(times) / 1000.0
        print(f"  median: {np.median(a):.0f} µs ({1e6/np.median(a):.1f} FPS)")
        return

    # Camera setup
    dev = args.device
    if dev.isdigit():
        dev = int(dev)
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"Error: cannot open {args.device}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print(f"Camera: {args.device}  {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

    if args.stream:
        stream_server = start_stream_server(args.stream_port)
        print(f"MJPEG stream: http://0.0.0.0:{args.stream_port}/")

    frame_count = 0
    fps_smooth = 0.0

    try:
        while True:
            _ = cap.grab()
            ret, frame = cap.read()
            if not ret:
                break

            if args.stereo != "none":
                h, w = frame.shape[:2]
                half = w // 2
                frame = frame[:, :half] if args.stereo == "left" else frame[:, half:]

            t0 = time.perf_counter()

            # Preprocess on CPU
            blob, scale, top, left = preprocess_frame(frame, args.size)

            # Copy into static input, replay graph
            x_static.copy_(torch.from_numpy(blob).cuda())
            g.replay()
            torch.cuda.synchronize()

            # Postprocess on CPU
            preds = postprocess_pytorch(out_static.detach(), conf_thresh=args.conf)
            preds = scale_boxes_to_frame(preds, args.size, scale, top, left, frame.shape)

            dt = time.perf_counter() - t0
            fps_smooth = 0.9 * fps_smooth + 0.1 * (1.0 / max(dt, 1e-6)) if fps_smooth > 0 else 1.0 / max(dt, 1e-6)

            out = annotate(frame.copy(), preds, dt)
            frame_count += 1

            if args.stream:
                update_stream_frame(out)

            print(f"\r[{frame_count}] {fps_smooth:.1f} FPS  {dt*1000:.0f}ms", end="", flush=True)

            if args.frames > 0 and frame_count >= args.frames:
                break

    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        cap.release()

    print(f"\nDone — {frame_count} frames, avg {fps_smooth:.1f} FPS")


if __name__ == "__main__":
    main()
