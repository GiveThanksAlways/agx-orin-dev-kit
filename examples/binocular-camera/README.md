# Binocular Camera (Waveshare Dual IMX219) — Stereo Vision on Jetson Orin AGX

Stereo vision projects using the **Waveshare Binocular Camera (Dual IMX219 8MP)**
on the **Jetson Orin AGX 64GB**, powered by **tinygrad** with the NV backend.

## Hardware

| Component | Spec |
|---|---|
| Camera | Waveshare Binocular Camera, 2× IMX219 8MP (3280×2464 max, 1280×720@30fps) |
| Baseline | ~60mm between lenses |
| Interface | 2× CSI-2 (2-lane each), via dual 15-pin FPC cables |
| Board | Jetson Orin AGX 64GB (ga10b iGPU, SM 8.7, 64GB LPDDR5) |

## Setup

### 1. Connect the cameras

Plug both 15-pin FPC cables into the Jetson's **CAM0** and **CAM1** connectors.
After boot, you should see two video devices:

```bash
ls /dev/video*
# /dev/video0  /dev/video1
```

### 2. Enter the Nix dev shell

```bash
cd examples/binocular-camera
nix develop
```

This gives you: v4l-utils, GStreamer, ffmpeg, CUDA toolkit, Python with OpenCV + tinygrad.

### 3. Verify cameras work

```bash
v4l2-ctl --device=/dev/video0 --all
v4l2-ctl --device=/dev/video1 --all

# Quick test capture
ffmpeg -f v4l2 -video_size 1280x720 -i /dev/video0 -frames:v 1 test_left.png
ffmpeg -f v4l2 -video_size 1280x720 -i /dev/video1 -frames:v 1 test_right.png
```

## Scripts

### Capture & Calibration

| Script | Description |
|---|---|
| `scripts/capture_stereo.py` | Capture synchronized stereo pairs (V4L2 or GStreamer) |
| `scripts/calibrate_stereo.py` | Checkerboard calibration → intrinsics + extrinsics + rectification |

### Stereo Vision Applications

| Script | Description |
|---|---|
| `scripts/depth_map.py` | GPU-accelerated depth estimation (tinygrad cost volume vs OpenCV SGBM) |
| `scripts/stereo_object_detect.py` | 3D object detection: CNN (tinygrad) + stereo depth → 3D bounding boxes |
| `scripts/obstacle_avoidance.py` | Real-time obstacle detection with clearance grid + heading suggestion |
| `scripts/hand_tracking_3d.py` | Hand keypoint detection + 3D tracking via stereo depth |

### Benchmarks

| Script | Description |
|---|---|
| `scripts/bench_stereo_pipeline.py` | Full pipeline throughput: depth + CNN + batch scaling, NV vs CUDA |

## Workflow

### Step 1: Capture calibration images

Print a checkerboard pattern (9×6 inner corners, 25mm squares recommended).
Hold it at various angles and distances in front of both cameras:

```bash
python3 scripts/capture_stereo.py --save calibration/images --count 30
```

### Step 2: Calibrate

```bash
python3 scripts/calibrate_stereo.py calibration/images \
    --output calibration/stereo_calib.npz \
    --board-cols 9 --board-rows 6 --square-size 25

# Verify the rectification visually:
python3 scripts/calibrate_stereo.py calibration/images --verify
```

### Step 3: Depth estimation

```bash
# Single pair
NV=1 python3 scripts/depth_map.py \
    --calib calibration/stereo_calib.npz \
    --left captures/left_0000.png --right captures/right_0000.png

# Live depth
NV=1 python3 scripts/depth_map.py --calib calibration/stereo_calib.npz --live

# Benchmark tinygrad vs OpenCV
NV=1 python3 scripts/depth_map.py --calib calibration/stereo_calib.npz --bench
```

### Step 4: 3D applications

```bash
# 3D object detection
NV=1 python3 scripts/stereo_object_detect.py --calib calibration/stereo_calib.npz --live

# Obstacle avoidance
NV=1 python3 scripts/obstacle_avoidance.py --calib calibration/stereo_calib.npz --live

# 3D hand tracking
NV=1 python3 scripts/hand_tracking_3d.py --calib calibration/stereo_calib.npz --live

# Headless mode (SSH — no display)
NV=1 python3 scripts/obstacle_avoidance.py --calib calibration/stereo_calib.npz --live --headless
```

### Step 5: Benchmark

```bash
# NV backend
NV=1 python3 scripts/bench_stereo_pipeline.py

# CUDA backend
CUDA=1 python3 scripts/bench_stereo_pipeline.py

# With BEAM search optimization
NV=1 JITBEAM=2 python3 scripts/bench_stereo_pipeline.py
```

## Architecture

### Depth Estimation (tinygrad GPU)

The depth pipeline avoids OpenCV's CPU-bound SGBM and instead runs entirely
on the GPU using tinygrad's tensor operations:

1. **Upload** left/right rectified images as `Tensor`
2. **Build cost volume**: For each disparity `d ∈ [0, max_disp)`, shift the right
   image by `d` pixels and compute SAD (Sum of Absolute Differences)
3. **Aggregate**: Box filter (uniform kernel conv2d) for local cost smoothing
4. **WTA**: `argmin` over the disparity axis → dense disparity map
5. **Reproject**: Q matrix from calibration → metric depth (meters)

This is the core of all classical stereo methods, and tinygrad's lazy evaluation
+ JIT compilation means the entire cost volume construction fuses into efficient
GPU kernels — especially with BEAM search.

### CNN Models (tinygrad)

All models use tinygrad's `nn.Conv2d` and `nn.Linear` — no external frameworks:

- **TinyDetector**: MobileNet-v1-like backbone with depthwise separable convs,
  SSD-style detection head (class + bbox + objectness per grid cell)
- **HandKeypointNet**: 4 conv layers + 2 FC layers → 21 keypoints × (x, y)
- Both run on the NV backend's HCQ (Hardware Command Queue) for minimal latency

### 3D Fusion

2D detections + stereo depth → 3D positions using the Q reprojection matrix
from stereo calibration. This gives metric (X, Y, Z) coordinates in meters
for every detected object, keypoint, or obstacle zone.

## Directories

```
binocular-camera/
├── flake.nix                  # Nix dev shell (v4l-utils, GStreamer, OpenCV, tinygrad)
├── README.md                  # This file
├── scripts/
│   ├── capture_stereo.py      # Stereo capture (V4L2 / GStreamer)
│   ├── calibrate_stereo.py    # Stereo calibration (checkerboard)
│   ├── depth_map.py           # GPU depth estimation (tinygrad + OpenCV baseline)
│   ├── stereo_object_detect.py# 3D object detection
│   ├── obstacle_avoidance.py  # Clearance grid + heading
│   ├── hand_tracking_3d.py    # 3D hand keypoints + gestures
│   └── bench_stereo_pipeline.py # Pipeline throughput benchmark
├── calibration/               # Calibration data (.npz) and images
└── models/                    # Trained model weights (when available)
```
