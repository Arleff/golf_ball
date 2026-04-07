"""
Generate a synthetic golf swing test video with three phases:
  1. Pre-impact:  ball sits still on the ground, club approaches
  2. Impact:      sudden motion burst (bright flash frame)
  3. Flight:      ball follows parabolic trajectory across the sky

Used for pipeline validation.
"""

import cv2
import numpy as np


def create_test_video(output_path="test_swing.mp4", num_frames=150, fps=30):
    width, height = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    impact_frame = 30
    ball_radius = 10
    ball_x_start, ball_y_start = 300, height - 100

    vx0, vy0 = 14, -20
    gravity = 0.45

    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        sky = np.linspace([140, 180, 230], [90, 140, 200], height // 2).astype(np.uint8)
        for row in range(height // 2):
            frame[row, :] = sky[row]

        grass_base = np.array([40, 120, 50], dtype=np.uint8)
        for row in range(height // 2, height):
            shade = 1.0 - 0.15 * (row - height // 2) / (height // 2)
            frame[row, :] = (grass_base * shade).astype(np.uint8)

        if i < impact_frame:
            bx, by = ball_x_start, ball_y_start
            cv2.circle(frame, (bx, by), ball_radius, (255, 255, 255), -1)
            cv2.circle(frame, (bx - 3, by - 3), 3, (240, 240, 230), -1)

            club_progress = i / impact_frame
            club_angle = -90 + 120 * club_progress
            rad = np.radians(club_angle)
            pivot_x, pivot_y = ball_x_start - 30, ball_y_start - 120
            club_len = 130
            tip_x = int(pivot_x + club_len * np.cos(rad))
            tip_y = int(pivot_y + club_len * np.sin(rad))
            cv2.line(frame, (pivot_x, pivot_y), (tip_x, tip_y),
                     (80, 80, 80), 4, cv2.LINE_AA)
            cv2.circle(frame, (tip_x, tip_y), 8, (100, 100, 100), -1)

        elif i == impact_frame:
            bx, by = ball_x_start, ball_y_start
            cv2.circle(frame, (bx, by), ball_radius + 5, (255, 255, 200), -1)
            cv2.circle(frame, (bx, by), ball_radius + 15, (255, 255, 100), 2)

        else:
            t = i - impact_frame
            bx = ball_x_start + vx0 * t
            by = ball_y_start + vy0 * t + 0.5 * gravity * t * t
            bx_i = int(np.clip(bx, ball_radius, width - ball_radius))
            by_i = int(np.clip(by, ball_radius, height - ball_radius))

            if 0 < bx_i < width and 0 < by_i < height:
                cv2.circle(frame, (bx_i, by_i), ball_radius,
                           (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, (bx_i - 3, by_i - 3), 3,
                           (240, 240, 230), -1, cv2.LINE_AA)

        phase_txt = "Pre-Impact" if i < impact_frame else (
            "IMPACT!" if i == impact_frame else "Flight")
        cv2.putText(frame, f"Frame {i}  [{phase_txt}]", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        writer.write(frame)

    writer.release()
    print(f"Test video: {output_path} ({num_frames} frames, impact@{impact_frame})")
    return output_path


if __name__ == "__main__":
    create_test_video()
