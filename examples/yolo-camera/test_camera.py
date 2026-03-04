#!/usr/bin/env python3
"""Quick sanity check: grab one frame from the stereo camera and save it."""
import cv2, sys

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
if not cap.isOpened():
    print("ERROR: cannot open /dev/video0")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 60)

ret, frame = cap.read()
cap.release()

if not ret:
    print("ERROR: failed to grab frame")
    sys.exit(1)

h, w = frame.shape[:2]
print(f"Stereo frame: {w}x{h}")

left = frame[:, :w//2]
right = frame[:, w//2:]
cv2.imwrite("/tmp/stereo_full.jpg", frame)
cv2.imwrite("/tmp/stereo_left.jpg", left)
cv2.imwrite("/tmp/stereo_right.jpg", right)
print(f"Left eye:  {left.shape[1]}x{left.shape[0]}")
print(f"Saved: /tmp/stereo_full.jpg, /tmp/stereo_left.jpg, /tmp/stereo_right.jpg")
