"""
Generate a synthetic golf swing test video with a moving white ball
on a green background for testing the detection pipeline.
"""

import cv2
import numpy as np
import os


def create_test_video(output_path="test_swing.mp4", num_frames=90, fps=30):
    width, height = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    ball_radius = 12
    x0, y0 = 200, 500
    vx0, vy0 = 12, -18
    gravity = 0.5

    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (34, 120, 50)

        cv2.rectangle(frame, (0, height - 60), (width, height), (50, 80, 30), -1)

        t = i
        bx = int(x0 + vx0 * t)
        by = int(y0 + vy0 * t + 0.5 * gravity * t * t)
        bx = np.clip(bx, ball_radius, width - ball_radius)
        by = np.clip(by, ball_radius, height - ball_radius)

        cv2.circle(frame, (bx, by), ball_radius, (255, 255, 255), -1)
        cv2.circle(frame, (bx, by), ball_radius, (200, 200, 200), 1)

        highlight_x = bx - ball_radius // 3
        highlight_y = by - ball_radius // 3
        cv2.circle(frame, (highlight_x, highlight_y), ball_radius // 4, (255, 255, 240), -1)

        cv2.putText(frame, "Golf Swing Test", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        writer.write(frame)

    writer.release()
    print(f"Test video created: {output_path} ({num_frames} frames, {width}x{height})")
    return output_path


if __name__ == "__main__":
    create_test_video()
