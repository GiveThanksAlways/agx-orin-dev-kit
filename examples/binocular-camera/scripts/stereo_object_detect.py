#!/usr/bin/env python3
"""
3D Object Detection with Stereo Depth — tinygrad on Jetson Orin AGX.

Combines a lightweight CNN detector (tinygrad) with stereo depth to produce
3D bounding boxes. The pipeline:

  1. Capture stereo pair → rectify
  2. Run 2D object detection (tinygrad CNN)
  3. Compute depth map (tinygrad cost volume)
  4. For each 2D detection, estimate 3D position from median depth in bbox

This demonstrates real-time 3D perception using only tinygrad — no PyTorch,
no TensorRT, just raw GPU compute via the NV backend.

Usage:
    NV=1 python3 stereo_object_detect.py --calib calibration/stereo_calib.npz --live
    NV=1 python3 stereo_object_detect.py --calib calibration/stereo_calib.npz \\
         --left test_left.png --right test_right.png
"""
import argparse, os, sys, time
import numpy as np

def build_tiny_detector(num_classes=5, input_size=224):
    """
    Build a lightweight object detector backbone in tinygrad.

    Architecture: MobileNet-v1-like with depthwise separable convolutions
    followed by a simple detection head (classification + bbox regression).

    This is a simplified single-shot detector (SSD-like) designed for
    real-time inference on the Jetson's iGPU.
    """
    from tinygrad import Tensor
    from tinygrad.nn import Conv2d, Linear

    class DepthwiseSeparableConv:
        def __init__(self, in_ch, out_ch, stride=1):
            self.dw = Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False)
            self.pw = Conv2d(in_ch, out_ch, 1, bias=False)

        def __call__(self, x):
            return self.pw(self.dw(x).relu()).relu()

    class TinyDetector:
        """Tiny single-shot detector for stereo depth experiments."""
        def __init__(self):
            # Backbone: progressively downsample with depthwise separable convs
            self.conv0 = Conv2d(3, 16, 3, stride=2, padding=1)    # 224→112
            self.block1 = DepthwiseSeparableConv(16, 32, stride=2)  # 112→56
            self.block2 = DepthwiseSeparableConv(32, 64, stride=2)  # 56→28
            self.block3 = DepthwiseSeparableConv(64, 128, stride=2) # 28→14
            self.block4 = DepthwiseSeparableConv(128, 256, stride=2) # 14→7

            # Detection head (7x7 grid, each cell predicts 1 box)
            self.cls_head = Conv2d(256, num_classes, 1)  # Class scores
            self.reg_head = Conv2d(256, 4, 1)  # Bbox: (cx, cy, w, h)
            self.obj_head = Conv2d(256, 1, 1)  # Objectness score

        def __call__(self, x):
            """Forward pass. x: [B, 3, 224, 224] → detections."""
            x = self.conv0(x).relu()
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            feat = self.block4(x)  # [B, 256, 7, 7]

            cls_logits = self.cls_head(feat)  # [B, num_classes, 7, 7]
            bbox_reg = self.reg_head(feat).sigmoid()  # [B, 4, 7, 7] normalized
            objectness = self.obj_head(feat).sigmoid()  # [B, 1, 7, 7]

            return cls_logits, bbox_reg, objectness

        def detect(self, x, conf_thresh=0.3):
            """Run detection and return decoded bounding boxes."""
            cls_logits, bbox_reg, objectness = self(x)

            # Decode on CPU
            cls_np = cls_logits.softmax(axis=1).numpy()
            bbox_np = bbox_reg.numpy()
            obj_np = objectness.numpy()

            detections = []
            B, C, grid_h, grid_w = cls_np.shape

            for b in range(B):
                for gy in range(grid_h):
                    for gx in range(grid_w):
                        conf = obj_np[b, 0, gy, gx]
                        if conf < conf_thresh:
                            continue
                        cls_id = np.argmax(cls_np[b, :, gy, gx])
                        cls_conf = cls_np[b, cls_id, gy, gx]
                        cx, cy, w, h = bbox_np[b, :, gy, gx]
                        # Convert grid-relative to image coordinates
                        cx = (gx + cx) / grid_w
                        cy = (gy + cy) / grid_h
                        detections.append({
                            "class": cls_id,
                            "confidence": float(conf * cls_conf),
                            "bbox": (cx - w/2, cy - h/2, cx + w/2, cy + h/2),  # x1,y1,x2,y2 normalized
                        })

            return detections

    return TinyDetector()

CLASS_NAMES = ["person", "chair", "bottle", "cup", "book"]

def estimate_3d_position(detection, depth_map, img_size, Q):
    """
    Estimate 3D position of a detected object using stereo depth.

    Takes the 2D bounding box, extracts the median depth within that region,
    and back-projects the center pixel to 3D using the Q matrix.
    """
    x1, y1, x2, y2 = detection["bbox"]
    H, W = depth_map.shape
    # Convert normalized coords to pixel coords
    px1, py1 = int(x1 * W), int(y1 * H)
    px2, py2 = int(x2 * W), int(y2 * H)
    px1, py1 = max(0, px1), max(0, py1)
    px2, py2 = min(W, px2), min(H, py2)

    if px2 <= px1 or py2 <= py1:
        return None

    roi_depth = depth_map[py1:py2, px1:px2]
    valid = roi_depth[roi_depth > 0]
    if len(valid) == 0:
        return None

    median_depth = np.median(valid)
    cx = (px1 + px2) / 2
    cy = (py1 + py2) / 2

    # Back-project to 3D: X = (cx - cx0) * Z / f, Y = (cy - cy0) * Z / f
    fx = Q[2, 3]  # -focal_length * baseline (negative)
    cx0 = -Q[0, 3]
    cy0 = -Q[1, 3]

    X = (cx - cx0) * median_depth / abs(fx) if fx != 0 else 0
    Y = (cy - cy0) * median_depth / abs(fx) if fx != 0 else 0
    Z = median_depth

    return {"x": float(X), "y": float(Y), "z": float(Z),
            "depth_median": float(median_depth),
            "pixel_bbox": (px1, py1, px2, py2)}

def preprocess_for_detection(frame, size=224):
    """Resize and normalize image for the detector."""
    import cv2
    from tinygrad import Tensor
    resized = cv2.resize(frame, (size, size))
    # BGR→RGB, HWC→CHW, normalize to [0,1]
    rgb = resized[:, :, ::-1].astype(np.float32) / 255.0
    chw = np.transpose(rgb, (2, 0, 1))
    return Tensor(chw).reshape(1, 3, size, size)

def draw_3d_detections(frame, detections_3d):
    """Draw 2D bboxes with 3D distance annotations."""
    import cv2
    for det in detections_3d:
        if det["position"] is None:
            continue
        pos = det["position"]
        x1, y1, x2, y2 = pos["pixel_bbox"]
        cls_name = CLASS_NAMES[det["class"]]
        conf = det["confidence"]
        depth = pos["z"]

        color = [(0,255,0), (255,0,0), (0,0,255), (255,255,0), (0,255,255)][det["class"] % 5]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{cls_name} {conf:.2f} @ {depth:.2f}m"
        cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        # 3D position annotation
        pos_str = f"({pos['x']:.2f}, {pos['y']:.2f}, {pos['z']:.2f})m"
        cv2.putText(frame, pos_str, (x1, y2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    return frame

def main():
    parser = argparse.ArgumentParser(description="3D Object Detection with Stereo")
    parser.add_argument("--calib", required=True, help="Calibration .npz file")
    parser.add_argument("--left", help="Left image path")
    parser.add_argument("--right", help="Right image path")
    parser.add_argument("--live", action="store_true", help="Live detection from cameras")
    parser.add_argument("--conf-thresh", type=float, default=0.3, help="Detection threshold")
    parser.add_argument("--max-disp", type=int, default=64, help="Max disparity for depth")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    import cv2
    from depth_map import stereo_cost_volume_tinygrad, disparity_to_depth, colorize_depth

    # Load calibration
    calib = np.load(args.calib)
    mtx_l, dist_l = calib["mtx_l"], calib["dist_l"]
    R1, R2, P1, P2, Q = calib["R1"], calib["R2"], calib["P1"], calib["P2"], calib["Q"]
    img_size = tuple(calib["img_size"])
    map1_l, map2_l = cv2.initUndistortRectifyMap(mtx_l, dist_l, R1, P1, img_size, cv2.CV_16SC2)
    mtx_r, dist_r = calib["mtx_r"], calib["dist_r"]
    map1_r, map2_r = cv2.initUndistortRectifyMap(mtx_r, dist_r, R2, P2, img_size, cv2.CV_16SC2)

    print("Building detector...")
    detector = build_tiny_detector(num_classes=len(CLASS_NAMES))
    print("Detector ready (random weights — for demo/benchmark purposes)")
    print("Note: For real detections, train the model or load pretrained weights\n")

    if args.live:
        from capture_stereo import get_v4l2_capture, capture_stereo_pair
        cap_l = get_v4l2_capture(0, img_size[0], img_size[1])
        cap_r = get_v4l2_capture(1, img_size[0], img_size[1])
        print("Live 3D detection (press 'q' to quit)...")

        while True:
            t0 = time.time()

            frame_l, frame_r = capture_stereo_pair(cap_l, cap_r)
            gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
            rect_l = cv2.remap(gray_l, map1_l, map2_l, cv2.INTER_LINEAR)
            rect_r = cv2.remap(gray_r, map1_r, map2_r, cv2.INTER_LINEAR)

            # Depth map (tinygrad GPU)
            disp = stereo_cost_volume_tinygrad(rect_l, rect_r, args.max_disp)
            depth = disparity_to_depth(disp, Q)

            # 2D Detection (tinygrad GPU)
            input_tensor = preprocess_for_detection(frame_l)
            detections_2d = detector.detect(input_tensor, args.conf_thresh)

            # Fuse: 2D + depth → 3D
            detections_3d = []
            for det in detections_2d:
                pos = estimate_3d_position(det, depth, img_size, Q)
                detections_3d.append({**det, "position": pos})

            fps = 1.0 / (time.time() - t0)

            if not args.headless:
                vis = frame_l.copy()
                vis = draw_3d_detections(vis, detections_3d)
                depth_vis = colorize_depth(depth)
                combined = np.hstack([vis, depth_vis])
                cv2.putText(combined, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("3D Detection | Depth", combined)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap_l.release()
        cap_r.release()
        cv2.destroyAllWindows()
    else:
        if not args.left or not args.right:
            print("Provide --left/--right or --live")
            sys.exit(1)

        img_l = cv2.imread(args.left)
        img_r = cv2.imread(args.right)
        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
        rect_l = cv2.remap(gray_l, map1_l, map2_l, cv2.INTER_LINEAR)
        rect_r = cv2.remap(gray_r, map1_r, map2_r, cv2.INTER_LINEAR)

        # Depth
        disp = stereo_cost_volume_tinygrad(rect_l, rect_r, args.max_disp)
        depth = disparity_to_depth(disp, Q)

        # Detection
        input_tensor = preprocess_for_detection(img_l)
        detections_2d = detector.detect(input_tensor, args.conf_thresh)

        # 3D fusion
        for det in detections_2d:
            pos = estimate_3d_position(det, depth, img_size, Q)
            cls = CLASS_NAMES[det["class"]]
            if pos:
                print(f"  {cls} ({det['confidence']:.2f}): "
                      f"X={pos['x']:.2f}m Y={pos['y']:.2f}m Z={pos['z']:.2f}m")
            else:
                print(f"  {cls} ({det['confidence']:.2f}): no depth")

        # Save visualization
        vis = draw_3d_detections(img_l.copy(), [
            {**d, "position": estimate_3d_position(d, depth, img_size, Q)}
            for d in detections_2d
        ])
        cv2.imwrite("detection_3d.png", vis)
        cv2.imwrite("depth_map.png", colorize_depth(depth))
        print("\nSaved detection_3d.png and depth_map.png")

if __name__ == "__main__":
    main()
