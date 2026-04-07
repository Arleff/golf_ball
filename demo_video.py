"""
Golf Ball Detection and Tracking from Swing Videos

Hybrid approach:
  1. CNN-based: torchvision Faster R-CNN (COCO pre-trained, "sports ball")
  2. CV-based:  Hough circle + color filter for small white golf balls
Combined with a Kalman filter for temporal tracking — reproducing the approach
described in "Efficient Golf Ball Detection and Tracking Based on CNNs and
Kalman Filter" (Zhang et al., IEEE SMC 2020).
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights


BALL_RELATED_COCO_IDS = {37}


class KalmanBallTracker:
    """
    Linear Kalman filter for 2D ball tracking.
    State: [x, y, vx, vy]  (position + velocity)
    Measurement: [x, y]

    Mirrors the Kalman filter used in test_net.py with the same transition
    and observation matrices from the original paper implementation.
    """

    def __init__(self):
        self.A = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)
        self.H = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
        ], dtype=np.float64)
        self.Q = 0.1 * np.eye(4)
        self.R = 0.0001 * np.eye(2)

        self.x = np.zeros(4)
        self.P = np.eye(4)
        self.initialized = False
        self.frames_since_seen = 0
        self.max_coast = 30

    def predict(self):
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x[:2].copy()

    def update(self, z):
        """z: measurement [x_center, y_center]"""
        z = np.array(z, dtype=np.float64)
        if not self.initialized:
            self.x[:2] = z
            self.initialized = True
            self.frames_since_seen = 0
            return

        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        innovation = z - self.H @ self.x
        self.x = self.x + K @ innovation
        self.P = (np.eye(4) - K @ self.H) @ self.P
        self.frames_since_seen = 0

    def mark_missed(self):
        self.frames_since_seen += 1

    @property
    def is_lost(self):
        return self.frames_since_seen > self.max_coast

    @property
    def position(self):
        return int(round(self.x[0])), int(round(self.x[1]))


def load_detector(score_threshold=0.3):
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=score_threshold)
    model.eval()
    preprocess = weights.transforms()
    return model, preprocess


def detect_balls_cnn(model, frame_rgb, device="cpu"):
    """CNN-based detection using Faster R-CNN (COCO "sports ball")."""
    img_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        preds = model([img_tensor])
    pred = preds[0]

    results = []
    for label, score, box in zip(
        pred["labels"].cpu().numpy(),
        pred["scores"].cpu().numpy(),
        pred["boxes"].cpu().numpy(),
    ):
        if int(label) in BALL_RELATED_COCO_IDS:
            results.append((*box, float(score)))
    return results


def detect_balls_cv(frame_bgr, min_radius=4, max_radius=40):
    """
    Traditional CV fallback: detect small white/bright circular objects
    using Hough circles and color filtering. Golf balls are typically
    small, white, and round.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=30,
        param1=80,
        param2=25,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    results = []

    if circles is not None:
        circles = np.round(circles[0]).astype(int)
        for (cx, cy, r) in circles:
            y1 = max(cy - r, 0)
            y2 = min(cy + r, frame_bgr.shape[0])
            x1 = max(cx - r, 0)
            x2 = min(cx + r, frame_bgr.shape[1])

            if y2 <= y1 or x2 <= x1:
                continue

            roi_hsv = hsv[y1:y2, x1:x2]
            roi_gray = gray[y1:y2, x1:x2]

            mean_v = np.mean(roi_hsv[:, :, 2])
            mean_s = np.mean(roi_hsv[:, :, 1])
            mean_brightness = np.mean(roi_gray)

            is_white = mean_brightness > 150 and mean_s < 80
            is_bright = mean_v > 160

            if is_white or is_bright:
                score = 0.5 + 0.3 * (mean_brightness / 255.0) + 0.2 * (1.0 - mean_s / 255.0)
                score = min(score, 0.95)
                results.append((float(x1), float(y1), float(x2), float(y2), score))

    return results


def merge_detections(cnn_dets, cv_dets, iou_threshold=0.3):
    """
    Merge CNN and CV detections. CNN detections take priority; CV detections
    are only added when they don't overlap with any CNN result.
    """
    if not cnn_dets and not cv_dets:
        return []

    if not cnn_dets:
        return cv_dets
    if not cv_dets:
        return cnn_dets

    merged = list(cnn_dets)

    for cv_det in cv_dets:
        cv_box = np.array(cv_det[:4])
        overlaps = False
        for cnn_det in cnn_dets:
            cnn_box = np.array(cnn_det[:4])
            iou = _compute_iou(cv_box, cnn_box)
            if iou > iou_threshold:
                overlaps = True
                break
        if not overlaps:
            merged.append(cv_det)

    return merged


def _compute_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / max(union, 1e-6)


def draw_detections(frame, detections, tracker, frame_idx, trajectory):
    """Draw bounding boxes, tracker state, and trajectory on the frame."""
    overlay = frame.copy()

    for det in detections:
        x1, y1, x2, y2, score = det
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"ball {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(overlay, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 255, 0), -1)
        cv2.putText(overlay, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

    if tracker.initialized and not tracker.is_lost:
        tx, ty = tracker.position
        cv2.circle(overlay, (tx, ty), 8, (0, 0, 255), -1)
        cv2.circle(overlay, (tx, ty), 12, (0, 0, 255), 2)

    for i in range(1, len(trajectory)):
        alpha = i / len(trajectory)
        color = (int(255 * (1 - alpha)), 0, int(255 * alpha))
        thickness = max(1, int(3 * alpha))
        cv2.line(overlay, trajectory[i - 1], trajectory[i], color, thickness, cv2.LINE_AA)

    info_y = 30
    cv2.putText(overlay, f"Frame {frame_idx}", (10, info_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    info_y += 30
    cv2.putText(overlay, f"Detections: {len(detections)}", (10, info_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
    if tracker.initialized:
        info_y += 25
        state = "Tracking" if not tracker.is_lost else "Lost"
        cv2.putText(overlay, f"Tracker: {state}", (10, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

    return overlay


def process_video(input_path, output_path=None, score_threshold=0.3, max_frames=0,
                  use_cv_fallback=True):
    """
    Process a video file: detect golf balls frame-by-frame and track with Kalman filter.

    Args:
        input_path:      path to input video
        output_path:     path to output video (None = auto-generate)
        score_threshold: minimum confidence for CNN detections
        max_frames:      0 = process all frames
        use_cv_fallback: also run Hough-circle white-ball detector
    Returns:
        output_path, stats dict
    """
    if output_path is None:
        base, _ = os.path.splitext(input_path)
        output_path = base + "_detected.mp4"

    print("Loading Faster R-CNN model (COCO pre-trained) ...")
    model, preprocess = load_detector(score_threshold)
    device = "cpu"
    model = model.to(device)
    print("Model loaded.")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    tracker = KalmanBallTracker()
    trajectory = []
    stats = {
        "total_frames": 0,
        "frames_with_detection": 0,
        "total_detections": 0,
        "cnn_detections": 0,
        "cv_detections": 0,
        "avg_detect_time": 0,
    }
    total_detect_time = 0.0
    frame_idx = 0

    print(f"Processing video: {width}x{height} @ {fps:.1f} fps, {total_frames} frames")
    print(f"Output: {output_path}")
    print(f"CV fallback: {'ON' if use_cv_fallback else 'OFF'}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if 0 < max_frames <= frame_idx:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        t0 = time.time()

        cnn_dets = detect_balls_cnn(model, frame_rgb, device)
        stats["cnn_detections"] += len(cnn_dets)

        cv_dets = []
        if use_cv_fallback:
            cv_dets = detect_balls_cv(frame)
            stats["cv_detections"] += len(cv_dets)

        detections = merge_detections(cnn_dets, cv_dets)

        detect_time = time.time() - t0
        total_detect_time += detect_time

        tracker.predict()

        if detections:
            best = max(detections, key=lambda d: d[4])
            cx = (best[0] + best[2]) / 2.0
            cy = (best[1] + best[3]) / 2.0
            tracker.update([cx, cy])
            stats["frames_with_detection"] += 1
            stats["total_detections"] += len(detections)
        else:
            tracker.mark_missed()

        if tracker.initialized and not tracker.is_lost:
            trajectory.append(tracker.position)
            if len(trajectory) > 200:
                trajectory = trajectory[-200:]

        annotated = draw_detections(frame, detections, tracker, frame_idx, trajectory)
        writer.write(annotated)

        frame_idx += 1
        if frame_idx % 10 == 0 or frame_idx == 1:
            sys.stdout.write(
                f"\r  Frame {frame_idx}/{total_frames} "
                f"| Det: {len(detections)} (CNN:{len(cnn_dets)} CV:{len(cv_dets)}) "
                f"| {detect_time:.3f}s"
            )
            sys.stdout.flush()

    cap.release()
    writer.release()
    print(f"\nDone! Processed {frame_idx} frames.")

    stats["total_frames"] = frame_idx
    stats["avg_detect_time"] = total_detect_time / max(frame_idx, 1)
    return output_path, stats


def find_default_video():
    """Auto-discover a video file in the current directory when --input is omitted."""
    video_exts = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm")
    candidates = [
        f for f in os.listdir(".")
        if os.path.isfile(f)
        and f.lower().endswith(video_exts)
        and "_detected" not in f
    ]
    if candidates:
        candidates.sort(key=lambda f: os.path.getmtime(f), reverse=True)
        return candidates[0]
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Golf Ball Detection & Tracking from Swing Videos"
    )
    parser.add_argument("--input", "-i", default=None,
                        help="Path to input video file (omit to auto-detect or generate a demo)")
    parser.add_argument("--output", "-o", default=None,
                        help="Path to output video (default: <input>_detected.mp4)")
    parser.add_argument("--threshold", "-t", type=float, default=0.3,
                        help="Detection confidence threshold (default: 0.3)")
    parser.add_argument("--max_frames", type=int, default=0,
                        help="Max frames to process (0 = all)")
    parser.add_argument("--no_cv_fallback", action="store_true",
                        help="Disable Hough-circle CV fallback detector")

    args = parser.parse_args()

    input_path = args.input
    if input_path is None:
        input_path = find_default_video()
        if input_path:
            print(f"未指定 --input，自动发现视频: {input_path}")
        else:
            print("未指定 --input，当前目录无视频文件，自动生成测试视频 ...")
            from create_test_video import create_test_video
            input_path = create_test_video()

    output_path, stats = process_video(
        input_path,
        args.output,
        args.threshold,
        args.max_frames,
        use_cv_fallback=not args.no_cv_fallback,
    )

    print("\n--- Statistics ---")
    print(f"  Total frames:           {stats['total_frames']}")
    print(f"  Frames with detection:  {stats['frames_with_detection']}")
    print(f"  Total detections:       {stats['total_detections']}")
    print(f"    CNN detections:       {stats['cnn_detections']}")
    print(f"    CV  detections:       {stats['cv_detections']}")
    print(f"  Avg detection time:     {stats['avg_detect_time']:.3f}s/frame")
    print(f"  Output saved to:        {output_path}")


if __name__ == "__main__":
    main()
