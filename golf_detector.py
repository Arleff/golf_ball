"""
高尔夫球检测与追踪模块
基于 Hough 圆形检测 + 卡尔曼滤波，复现论文
"Efficient Golf Ball Detection and Tracking Based on CNN + Kalman Filter"
的追踪逻辑（以 CV 方法替代 Faster R-CNN 推理，便于无 GPU 环境演示）
"""
import cv2
import numpy as np
from filterpy.kalman import KalmanFilter


class GolfBallKalmanTracker:
    """常速度卡尔曼滤波器，与论文 test_net.py 中结构一致"""

    def __init__(self):
        # 状态: [x, y, vx, vy]  观测: [x, y]
        self.kf = KalmanFilter(dim_x=4, dim_z=2)

        dt = 1.0
        self.kf.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=float)

        self.kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=float)

        self.kf.R *= 0.0001       # 观测噪声（极小，置信测量）
        self.kf.Q *= 0.1          # 过程噪声
        self.kf.P *= 100.0        # 初始协方差

        self.initialized = False

    def init(self, x: float, y: float):
        self.kf.x = np.array([[x], [y], [0.0], [0.0]], dtype=float)
        self.initialized = True

    def predict(self):
        self.kf.predict()
        return float(self.kf.x[0, 0]), float(self.kf.x[1, 0])

    def update(self, x: float, y: float):
        self.kf.update(np.array([[x], [y]]))
        return float(self.kf.x[0, 0]), float(self.kf.x[1, 0])


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """增强对比度，便于检测白色高尔夫球"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)
    return gray


def detect_golf_ball_hough(
    gray_crop: np.ndarray,
    min_radius: int = 5,
    max_radius: int = 60,
) -> tuple[int, int, int] | None:
    """
    用 HoughCircles 在裁剪灰度图中检测高尔夫球。
    返回 (x, y, r) 或 None。
    """
    circles = cv2.HoughCircles(
        gray_crop,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=20,
        param1=50,
        param2=20,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is None:
        return None
    circles = np.uint16(np.around(circles[0]))
    # 取 accumulator 最大的（第一个）
    x, y, r = circles[0]
    return int(x), int(y), int(r)


def detect_golf_ball_color(
    frame_crop: np.ndarray,
) -> tuple[int, int, int] | None:
    """
    颜色 + 形状备用检测：高尔夫球通常为白色圆形。
    """
    hsv = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2HSV)
    # 白色范围
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 40, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_score = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity > 0.6 and area > best_score:
            best_score = area
            best = cnt
    if best is None:
        return None
    (x, y), r = cv2.minEnclosingCircle(best)
    return int(x), int(y), max(int(r), 5)


ROI_HALF = 150   # 与论文 test_net.py 中 ±150 px 一致


def process_video(
    video_path: str,
    output_path: str,
    progress_callback=None,
) -> dict:
    """
    主处理函数：逐帧检测高尔夫球并用卡尔曼滤波平滑轨迹，输出带标注的视频。

    返回统计信息 dict。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    tracker = GolfBallKalmanTracker()
    trajectory: list[tuple[int, int]] = []

    detected_count = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_full = preprocess_frame(frame)

        # ---------- 确定 ROI ----------
        if not tracker.initialized:
            # 首帧：全图搜索
            cx, cy = width // 2, height // 2
        else:
            cx, cy = tracker.predict()
            cx, cy = int(cx), int(cy)

        left = max(cx - ROI_HALF, 0)
        upper = max(cy - ROI_HALF, 0)
        right = min(cx + ROI_HALF, width)
        lower = min(cy + ROI_HALF, height)

        gray_crop = gray_full[upper:lower, left:right]
        color_crop = frame[upper:lower, left:right]

        # ---------- 检测 ----------
        det = detect_golf_ball_hough(gray_crop)
        if det is None:
            det = detect_golf_ball_color(color_crop)

        if det is not None:
            lx, ly, lr = det
            # 转回全图坐标
            gx = lx + left
            gy = ly + upper

            if not tracker.initialized:
                tracker.init(float(gx), float(gy))

            sx, sy = tracker.update(float(gx), float(gy))
            detected_count += 1
        else:
            if tracker.initialized:
                sx, sy = tracker.predict()
            else:
                sx, sy = float(cx), float(cy)
            lr = 15  # 默认半径

        sx, sy = int(sx), int(sy)
        trajectory.append((sx, sy))

        # ---------- 绘制 ----------
        annotated = frame.copy()

        # ROI 框（半透明）
        roi_overlay = annotated.copy()
        cv2.rectangle(roi_overlay, (left, upper), (right, lower), (255, 200, 0), -1)
        cv2.addWeighted(roi_overlay, 0.05, annotated, 0.95, 0, annotated)
        cv2.rectangle(annotated, (left, upper), (right, lower), (255, 200, 0), 1)

        # 轨迹
        for k in range(1, len(trajectory)):
            alpha = k / len(trajectory)
            color = (
                int(0 * alpha + 0 * (1 - alpha)),
                int(255 * alpha + 128 * (1 - alpha)),
                int(255 * alpha + 0 * (1 - alpha)),
            )
            cv2.line(annotated, trajectory[k - 1], trajectory[k], color, 2, cv2.LINE_AA)

        # 检测圆
        if det is not None:
            cv2.circle(annotated, (sx, sy), lr + 4, (0, 255, 0), 2)
            cv2.circle(annotated, (sx, sy), 3, (0, 255, 0), -1)
        else:
            # 仅预测位置（虚线圆）
            cv2.circle(annotated, (sx, sy), lr + 4, (0, 165, 255), 2)
            cv2.circle(annotated, (sx, sy), 3, (0, 165, 255), -1)

        # 文字标签
        label = f"Golf Ball ({sx},{sy})"
        cv2.putText(annotated, label, (sx + lr + 6, sy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(annotated, f"Frame {frame_idx + 1}/{total_frames}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(annotated, f"Detected: {detected_count}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 120), 2, cv2.LINE_AA)

        out.write(annotated)
        frame_idx += 1

        if progress_callback and frame_idx % 10 == 0:
            progress_callback(frame_idx, total_frames)

    cap.release()
    out.release()

    return {
        "total_frames": total_frames,
        "detected_frames": detected_count,
        "detection_rate": round(detected_count / max(total_frames, 1) * 100, 1),
        "trajectory_points": len(trajectory),
        "fps": round(fps, 1),
        "resolution": f"{width}x{height}",
    }
