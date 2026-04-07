"""
Golf Ball Detection and Tracking from Swing Videos  (v2)

Two-pass pipeline:
  Pass 1 — detect ball candidates per-frame, collect raw observations
  Pass 2 — fit parabolic trajectory via RANSAC, render smooth curve

Detection: Faster R-CNN (COCO "sports ball") + frame-difference motion detector
Tracking:  Kalman filter with gravity model, Mahalanobis gating
User aids: optional impact-frame index and seed-point annotations
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import cv2
import torch
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)

BALL_COCO_IDS = {37}


# ---------------------------------------------------------------------------
#  Kalman filter with constant-acceleration (gravity) model
# ---------------------------------------------------------------------------

class BallKalmanTracker:
    """
    State  x = [px, py, vx, vy]
    Physics: px' = px + vx,  py' = py + vy + 0.5*g   (g in px/frame^2)
             vx' = vx,       vy' = vy + g
    Measurement z = [px, py]
    """

    def __init__(self, gravity=0.5, q_pos=2.0, q_vel=1.0, r_obs=4.0,
                 gate_sigma=4.0, max_coast=15):
        self.g = gravity
        self.gate_sigma = gate_sigma
        self.max_coast = max_coast

        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)
        self.B = np.array([0, 0.5 * self.g, 0, self.g], dtype=np.float64)

        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)

        self.Q = np.diag([q_pos, q_pos, q_vel, q_vel]) ** 2
        self.R = np.eye(2) * r_obs ** 2

        self.x = np.zeros(4, dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64) * 1e4

        self.initialized = False
        self.age = 0
        self.hits = 0
        self.frames_since_seen = 0

    def predict(self):
        self.x = self.F @ self.x + self.B
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        return self.x[:2].copy()

    def gating_distance(self, z):
        """Squared Mahalanobis distance between predicted measurement and z."""
        z = np.asarray(z, dtype=np.float64)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        d2 = y @ np.linalg.inv(S) @ y
        return d2

    def update(self, z):
        z = np.asarray(z, dtype=np.float64)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        self.hits += 1
        self.frames_since_seen = 0

    def init_state(self, pos, vel=None):
        self.x[:2] = np.asarray(pos, dtype=np.float64)
        if vel is not None:
            self.x[2:] = np.asarray(vel, dtype=np.float64)
        self.P = np.diag([10, 10, 50, 50]).astype(np.float64)
        self.initialized = True
        self.hits = 1
        self.frames_since_seen = 0

    def mark_missed(self):
        self.frames_since_seen += 1

    @property
    def is_lost(self):
        return self.frames_since_seen > self.max_coast

    @property
    def position(self):
        return self.x[0], self.x[1]

    @property
    def velocity(self):
        return self.x[2], self.x[3]


# ---------------------------------------------------------------------------
#  Detection helpers
# ---------------------------------------------------------------------------

def load_detector(score_threshold=0.25):
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(
        weights=weights, box_score_thresh=score_threshold
    )
    model.eval()
    return model


def detect_balls_cnn(model, frame_rgb, device="cpu"):
    """Returns list of (x1, y1, x2, y2, score)."""
    img = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
    with torch.no_grad():
        preds = model([img.to(device)])
    out = []
    for lab, sc, box in zip(
        preds[0]["labels"].cpu().numpy(),
        preds[0]["scores"].cpu().numpy(),
        preds[0]["boxes"].cpu().numpy(),
    ):
        if int(lab) in BALL_COCO_IDS:
            out.append((*box, float(sc)))
    return out


class MotionDetector:
    """Frame-difference based small-object motion detector."""

    def __init__(self, history=3, min_area=8, max_area=2500, learn_rate=0.05):
        self.history = history
        self.min_area = min_area
        self.max_area = max_area
        self.learn_rate = learn_rate
        self.bg = None

    def detect(self, gray):
        gray_f = gray.astype(np.float32)
        if self.bg is None:
            self.bg = gray_f.copy()
            return []

        diff = cv2.absdiff(gray_f, self.bg)
        self.bg = self.bg * (1 - self.learn_rate) + gray_f * self.learn_rate

        _, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        mask = mask.astype(np.uint8)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        results = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area or area > self.max_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            aspect = max(w, h) / max(min(w, h), 1)
            if aspect > 4:
                continue
            cx, cy = x + w / 2, y + h / 2
            results.append((float(x), float(y), float(x + w), float(y + h), cx, cy, area))
        return results


def pick_best_candidate(cnn_dets, motion_dets, tracker, frame_gray):
    """
    From CNN + motion candidates, pick the single best ball detection.
    Uses Mahalanobis gating when tracker is active.
    Returns (cx, cy, x1, y1, x2, y2, score, source) or None.
    """
    candidates = []

    for (x1, y1, x2, y2, sc) in cnn_dets:
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        if w > 150 or h > 150:
            continue
        candidates.append((cx, cy, x1, y1, x2, y2, float(sc) * 1.5, "cnn"))

    for (x1, y1, x2, y2, cx, cy, area) in motion_dets:
        roi = frame_gray[int(y1):int(y2), int(x1):int(x2)]
        if roi.size == 0:
            continue
        brightness = float(np.mean(roi))
        sc = 0.2 + 0.3 * (brightness / 255.0)
        candidates.append((cx, cy, x1, y1, x2, y2, sc, "motion"))

    if not candidates:
        return None

    if tracker.initialized and not tracker.is_lost:
        gate = tracker.gate_sigma ** 2
        gated = []
        for c in candidates:
            d2 = tracker.gating_distance([c[0], c[1]])
            if d2 < gate * 16:
                proximity_bonus = max(0, 1.0 - d2 / (gate * 16))
                gated.append((*c[:7], c[7], c[6] + proximity_bonus))
            else:
                gated.append((*c[:7], c[7], c[6] * 0.1))
        gated.sort(key=lambda c: c[8], reverse=True)
        best = gated[0]
        return best[:8]
    else:
        candidates.sort(key=lambda c: c[6], reverse=True)
        return candidates[0]


# ---------------------------------------------------------------------------
#  Impact frame detection
# ---------------------------------------------------------------------------

def detect_impact_frame(frames_gray, fps, search_window=None):
    """
    Detect the frame where the club hits the ball by looking for a spike
    in local motion energy. Returns frame index or None.
    """
    if len(frames_gray) < 3:
        return None

    energies = []
    for i in range(1, len(frames_gray)):
        diff = cv2.absdiff(frames_gray[i], frames_gray[i - 1])
        energies.append(float(np.sum(diff > 30)))

    if not energies:
        return None

    energies = np.array(energies, dtype=np.float64)
    if search_window:
        lo, hi = search_window
    else:
        lo, hi = 0, len(energies)

    window = energies[lo:hi]
    if len(window) == 0:
        return None

    median_e = np.median(window)
    std_e = np.std(window) + 1e-6

    best_idx = None
    best_score = 0
    for i in range(len(window)):
        score = (window[i] - median_e) / std_e
        if score > best_score and score > 2.0:
            best_score = score
            best_idx = i

    if best_idx is not None:
        return lo + best_idx + 1
    return None


# ---------------------------------------------------------------------------
#  Parabola fitting (RANSAC)
# ---------------------------------------------------------------------------

def fit_parabola_ransac(observations, min_samples=5, n_iter=200, inlier_thresh=15.0):
    """
    Fit y = a*x^2 + b*x + c  (treating frame-index t as implicit via x(t)).
    Actually we fit px(t) = a1*t + b1  and  py(t) = a2*t^2 + b2*t + c2
    since x is roughly linear and y is parabolic.

    observations: list of (frame_idx, cx, cy)
    Returns: (x_coeffs, y_coeffs, inlier_mask) or None
    """
    if len(observations) < min_samples:
        return None

    obs = np.array(observations, dtype=np.float64)
    t = obs[:, 0]
    px = obs[:, 1]
    py = obs[:, 2]
    n = len(obs)

    best_inliers = None
    best_count = 0

    for _ in range(n_iter):
        idx = np.random.choice(n, size=min(3, n), replace=False)
        ts = t[idx]

        A_x = np.column_stack([ts, np.ones_like(ts)])
        A_y = np.column_stack([ts ** 2, ts, np.ones_like(ts)])

        try:
            cx_fit = np.linalg.lstsq(A_x, px[idx], rcond=None)[0]
            cy_fit = np.linalg.lstsq(A_y, py[idx], rcond=None)[0]
        except np.linalg.LinAlgError:
            continue

        A_x_all = np.column_stack([t, np.ones(n)])
        A_y_all = np.column_stack([t ** 2, t, np.ones(n)])
        pred_x = A_x_all @ cx_fit
        pred_y = A_y_all @ cy_fit

        err = np.sqrt((px - pred_x) ** 2 + (py - pred_y) ** 2)
        inliers = err < inlier_thresh
        count = np.sum(inliers)

        if count > best_count:
            best_count = count
            best_inliers = inliers

    if best_count < min_samples:
        return None

    ti = t[best_inliers]
    A_x = np.column_stack([ti, np.ones_like(ti)])
    A_y = np.column_stack([ti ** 2, ti, np.ones_like(ti)])
    cx_fit = np.linalg.lstsq(A_x, px[best_inliers], rcond=None)[0]
    cy_fit = np.linalg.lstsq(A_y, py[best_inliers], rcond=None)[0]

    return cx_fit, cy_fit, best_inliers


def sample_parabola(x_coeffs, y_coeffs, t_start, t_end, num_points=200):
    """Generate smooth parabola points from fitted coefficients."""
    ts = np.linspace(t_start, t_end, num_points)
    xs = x_coeffs[0] * ts + x_coeffs[1]
    ys = y_coeffs[0] * ts ** 2 + y_coeffs[1] * ts + y_coeffs[2]
    return list(zip(xs, ys, ts))


# ---------------------------------------------------------------------------
#  Drawing / rendering
# ---------------------------------------------------------------------------

def draw_frame(frame, detections, best_candidate, tracker, frame_idx,
               phase, parabola_pts, raw_obs, impact_frame, annotations):
    """Render all overlays on a single frame."""
    overlay = frame.copy()
    h, w = frame.shape[:2]

    # --- parabola trajectory (smooth curve) ---
    if parabola_pts and len(parabola_pts) >= 2:
        visible = [(int(round(x)), int(round(y)))
                    for x, y, t in parabola_pts if t <= frame_idx]
        for i in range(1, len(visible)):
            alpha = i / len(visible)
            b = int(255 * (1 - alpha))
            r = int(255 * alpha)
            thick = max(1, int(2 + 2 * alpha))
            pt1 = visible[i - 1]
            pt2 = visible[i]
            if 0 <= pt1[0] < w and 0 <= pt1[1] < h and \
               0 <= pt2[0] < w and 0 <= pt2[1] < h:
                cv2.line(overlay, pt1, pt2, (b, 80, r), thick, cv2.LINE_AA)

    # --- raw observation dots (small, semi-transparent) ---
    for (t, cx, cy) in raw_obs:
        if t <= frame_idx:
            pt = (int(round(cx)), int(round(cy)))
            if 0 <= pt[0] < w and 0 <= pt[1] < h:
                cv2.circle(overlay, pt, 3, (200, 200, 0), -1, cv2.LINE_AA)

    # --- detection boxes ---
    if best_candidate:
        cx, cy, x1, y1, x2, y2, sc, src = best_candidate
        x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
        color = (0, 255, 0) if src == "cnn" else (0, 200, 255)
        cv2.rectangle(overlay, (x1i, y1i), (x2i, y2i), color, 2)
        label = f"ball {sc:.2f} [{src}]"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(overlay, (x1i, y1i - th - 6), (x1i + tw + 4, y1i), color, -1)
        cv2.putText(overlay, label, (x1i + 2, y1i - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # --- tracker crosshair ---
    if tracker.initialized and not tracker.is_lost:
        tx, ty = int(round(tracker.position[0])), int(round(tracker.position[1]))
        if 0 <= tx < w and 0 <= ty < h:
            cv2.drawMarker(overlay, (tx, ty), (0, 0, 255),
                           cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)
            cv2.circle(overlay, (tx, ty), 6, (0, 0, 255), -1, cv2.LINE_AA)

    # --- user annotation dots ---
    if annotations:
        for ann in annotations:
            af = ann.get("frame")
            if af is not None and af == frame_idx:
                ax, ay = int(ann["x"]), int(ann["y"])
                cv2.drawMarker(overlay, (ax, ay), (255, 0, 255),
                               cv2.MARKER_DIAMOND, 16, 2, cv2.LINE_AA)
                cv2.putText(overlay, "USER", (ax + 10, ay - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

    # --- impact frame marker ---
    if impact_frame is not None and frame_idx == impact_frame:
        cv2.putText(overlay, ">> IMPACT <<", (w // 2 - 80, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

    # --- HUD ---
    hud_y = 25
    phase_colors = {
        "pre_impact": (180, 180, 180),
        "flight": (0, 255, 100),
        "post_flight": (100, 100, 255),
    }
    phase_names = {
        "pre_impact": "Pre-Impact",
        "flight": "In Flight",
        "post_flight": "Landed",
    }
    pc = phase_colors.get(phase, (200, 200, 200))
    pn = phase_names.get(phase, phase)

    cv2.putText(overlay, f"Frame {frame_idx}", (10, hud_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(overlay, pn, (10, hud_y + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, pc, 1, cv2.LINE_AA)

    if tracker.initialized and not tracker.is_lost:
        vx, vy = tracker.velocity
        spd = np.sqrt(vx ** 2 + vy ** 2)
        cv2.putText(overlay, f"Speed: {spd:.1f} px/f", (10, hud_y + 53),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    return overlay


# ---------------------------------------------------------------------------
#  Main two-pass pipeline
# ---------------------------------------------------------------------------

def process_video(input_path, output_path=None, score_threshold=0.25,
                  max_frames=0, impact_frame_hint=None, annotations=None,
                  gravity=0.5, use_cv_fallback=True):
    """
    Two-pass processing:
      Pass 1 — per-frame detection + Kalman tracking → raw observations
      Pass 2 — RANSAC parabola fit → re-render with smooth trajectory

    annotations: list of {"frame": int, "x": float, "y": float}
    """
    if output_path is None:
        base, _ = os.path.splitext(input_path)
        output_path = base + "_detected.mp4"

    print("Loading Faster R-CNN ...")
    model = load_detector(score_threshold)
    device = "cpu"
    model.to(device)
    print("Model loaded.")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if annotations is None:
        annotations = []

    # ---- read all frames into memory (needed for two-pass) ----
    print(f"Video: {width}x{height} @ {fps:.1f} fps, {total_frames} frames")
    all_frames = []
    all_grays = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if 0 < max_frames <= len(all_frames):
            break
        all_frames.append(frame)
        all_grays.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()
    n_frames = len(all_frames)
    print(f"Loaded {n_frames} frames into memory.")

    # ---- detect impact frame ----
    impact_frame = impact_frame_hint
    if impact_frame is None:
        print("Auto-detecting impact frame ...")
        impact_frame = detect_impact_frame(all_grays, fps)
        if impact_frame is not None:
            print(f"  Impact frame detected: {impact_frame}")
        else:
            print("  Could not auto-detect impact frame; using frame 0.")
            impact_frame = 0

    # ---- inject user annotations as seed points ----
    ann_by_frame = {}
    for ann in annotations:
        f = ann.get("frame")
        if f is not None:
            ann_by_frame.setdefault(int(f), []).append(ann)

    # ================================================================
    #  PASS 1 — detect + track → collect raw observations
    # ================================================================
    print("Pass 1: Detection + Tracking ...")
    tracker = BallKalmanTracker(gravity=gravity)
    motion_det = MotionDetector()
    raw_observations = []
    per_frame_best = [None] * n_frames
    per_frame_cnn = [[] for _ in range(n_frames)]
    per_frame_phase = ["pre_impact"] * n_frames

    stats = {
        "total_frames": n_frames,
        "frames_with_detection": 0,
        "total_detections": 0,
        "cnn_detections": 0,
        "motion_detections": 0,
        "impact_frame": impact_frame,
        "parabola_fitted": False,
        "parabola_inliers": 0,
        "avg_detect_time": 0,
    }
    total_detect_time = 0.0

    for i in range(n_frames):
        frame = all_frames[i]
        gray = all_grays[i]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # determine phase
        if i < impact_frame:
            phase = "pre_impact"
        else:
            phase = "flight"
        per_frame_phase[i] = phase

        t0 = time.time()

        # CNN detection
        cnn_dets = detect_balls_cnn(model, frame_rgb, device)
        stats["cnn_detections"] += len(cnn_dets)
        per_frame_cnn[i] = cnn_dets

        # motion detection (only after impact)
        motion_dets = []
        if i >= impact_frame and use_cv_fallback:
            motion_dets = motion_det.detect(gray)
            stats["motion_detections"] += len(motion_dets)
        else:
            motion_det.detect(gray)

        detect_time = time.time() - t0
        total_detect_time += detect_time

        # user annotation for this frame
        if i in ann_by_frame:
            for ann in ann_by_frame[i]:
                ax, ay = float(ann["x"]), float(ann["y"])
                if not tracker.initialized:
                    tracker.init_state([ax, ay])
                else:
                    tracker.predict()
                    tracker.update([ax, ay])
                raw_observations.append((i, ax, ay))
                per_frame_best[i] = (ax, ay, ax - 10, ay - 10, ax + 10, ay + 10, 1.0, "user")
            stats["frames_with_detection"] += 1
            continue

        # predict
        if tracker.initialized:
            tracker.predict()

        # pick best candidate
        best = pick_best_candidate(cnn_dets, motion_dets, tracker, gray)

        if best is not None:
            cx, cy = best[0], best[1]

            if not tracker.initialized:
                if i >= impact_frame:
                    tracker.init_state([cx, cy])
            else:
                d2 = tracker.gating_distance([cx, cy])
                if d2 < (tracker.gate_sigma * 3) ** 2:
                    tracker.update([cx, cy])
                else:
                    tracker.mark_missed()
                    best = None

        if best is not None and tracker.initialized:
            raw_observations.append((i, best[0], best[1]))
            per_frame_best[i] = best
            stats["frames_with_detection"] += 1
            stats["total_detections"] += 1
        elif tracker.initialized:
            tracker.mark_missed()

        # transition to post_flight if tracker lost
        if tracker.initialized and tracker.is_lost and phase == "flight":
            per_frame_phase[i] = "post_flight"

        if (i + 1) % 10 == 0 or i == 0:
            sys.stdout.write(
                f"\r  Frame {i + 1}/{n_frames} | Obs: {len(raw_observations)} "
                f"| {detect_time:.3f}s"
            )
            sys.stdout.flush()

    stats["avg_detect_time"] = total_detect_time / max(n_frames, 1)
    print(f"\n  Pass 1 done. Raw observations: {len(raw_observations)}")

    # ================================================================
    #  PASS 1.5 — fit parabolic trajectory
    # ================================================================
    parabola_pts = []
    if len(raw_observations) >= 5:
        print("Fitting parabolic trajectory (RANSAC) ...")
        result = fit_parabola_ransac(
            raw_observations, min_samples=4, n_iter=300, inlier_thresh=20.0
        )
        if result is not None:
            x_coeffs, y_coeffs, inlier_mask = result
            n_inliers = int(np.sum(inlier_mask))
            stats["parabola_fitted"] = True
            stats["parabola_inliers"] = n_inliers
            print(f"  Parabola fitted with {n_inliers}/{len(raw_observations)} inliers.")

            obs_arr = np.array(raw_observations)
            t_min = obs_arr[inlier_mask, 0].min()
            t_max = obs_arr[inlier_mask, 0].max()
            parabola_pts = sample_parabola(x_coeffs, y_coeffs, t_min, t_max,
                                           num_points=max(200, int(t_max - t_min) * 3))
        else:
            print("  Parabola fitting failed (not enough inliers).")
    else:
        print(f"  Too few observations ({len(raw_observations)}) to fit parabola.")

    # ================================================================
    #  PASS 2 — render output video
    # ================================================================
    print("Pass 2: Rendering output ...")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for i in range(n_frames):
        annotated = draw_frame(
            all_frames[i],
            per_frame_cnn[i],
            per_frame_best[i],
            tracker,
            i,
            per_frame_phase[i],
            parabola_pts,
            raw_observations,
            impact_frame,
            annotations,
        )
        writer.write(annotated)

    writer.release()
    print(f"Done! Output: {output_path}")
    return output_path, stats


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def find_default_video():
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
        description="Golf Ball Detection & Tracking from Swing Videos (v2)"
    )
    parser.add_argument("--input", "-i", default=None,
                        help="Input video (omit to auto-detect or generate demo)")
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--threshold", "-t", type=float, default=0.25)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--impact_frame", type=int, default=None,
                        help="Manually specify the impact frame index")
    parser.add_argument("--annotations", type=str, default=None,
                        help='JSON: [{"frame":10,"x":500,"y":300}, ...]')
    parser.add_argument("--gravity", type=float, default=0.5,
                        help="Gravity constant in px/frame^2 (default 0.5)")
    parser.add_argument("--no_cv_fallback", action="store_true")

    args = parser.parse_args()

    input_path = args.input
    if input_path is None:
        input_path = find_default_video()
        if input_path:
            print(f"Auto-discovered: {input_path}")
        else:
            print("No video found, generating test video ...")
            from create_test_video import create_test_video
            input_path = create_test_video()

    annotations = []
    if args.annotations:
        annotations = json.loads(args.annotations)

    output_path, stats = process_video(
        input_path,
        args.output,
        args.threshold,
        args.max_frames,
        impact_frame_hint=args.impact_frame,
        annotations=annotations,
        gravity=args.gravity,
        use_cv_fallback=not args.no_cv_fallback,
    )

    print("\n--- Statistics ---")
    for k, v in stats.items():
        print(f"  {k:25s}: {v}")


if __name__ == "__main__":
    main()
