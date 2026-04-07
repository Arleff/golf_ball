"""
Gradio Web App: Golf Ball Detection & Tracking from Swing Videos

Upload a golf swing video and the system will:
1. Run frame-by-frame Faster R-CNN detection (COCO pre-trained, "sports ball")
2. Apply Kalman filter tracking for smooth ball trajectory
3. Output an annotated video with bounding boxes and trajectory overlay
"""

import os
import tempfile
import gradio as gr
from demo_video import process_video


def run_detection(video_file, confidence_threshold, max_frames):
    """Gradio handler: process uploaded video and return annotated output."""
    if video_file is None:
        raise gr.Error("请上传一个视频文件！")

    max_frames_int = int(max_frames) if max_frames else 0

    with tempfile.NamedTemporaryFile(suffix="_detected.mp4", delete=False) as tmp:
        output_path = tmp.name

    try:
        output_path, stats = process_video(
            input_path=video_file,
            output_path=output_path,
            score_threshold=confidence_threshold,
            max_frames=max_frames_int,
        )
    except Exception as e:
        raise gr.Error(f"处理视频时出错: {str(e)}")

    det_rate = stats['frames_with_detection'] / max(stats['total_frames'], 1) * 100
    summary = (
        f"### 检测结果统计\n\n"
        f"| 指标 | 数值 |\n"
        f"|------|------|\n"
        f"| 总帧数 | {stats['total_frames']} |\n"
        f"| 检测到球的帧数 | {stats['frames_with_detection']} |\n"
        f"| 检测率 | {det_rate:.1f}% |\n"
        f"| 总检测数 | {stats['total_detections']} |\n"
        f"| CNN 检测数 | {stats.get('cnn_detections', 'N/A')} |\n"
        f"| CV 检测数 | {stats.get('cv_detections', 'N/A')} |\n"
        f"| 平均检测耗时 | {stats['avg_detect_time']:.3f} 秒/帧 |\n"
    )

    return output_path, summary


with gr.Blocks(
    title="高尔夫球检测与跟踪",
) as demo:

    gr.Markdown(
        """
        # ⛳ 高尔夫球检测与跟踪系统

        基于 **Faster R-CNN + 卡尔曼滤波** 的高尔夫球实时检测与跟踪。

        > 复现论文: *Efficient Golf Ball Detection and Tracking Based on
        > Convolutional Neural Networks and Kalman Filter* (Zhang et al., IEEE SMC 2020)

        ### 使用方法
        1. 上传一段高尔夫挥杆视频
        2. 调整置信度阈值（默认 0.3）
        3. 点击 **开始检测** 按钮
        4. 查看带有检测框和轨迹的输出视频

        ---
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_video = gr.Video(label="📹 上传挥杆视频")
            with gr.Row():
                confidence = gr.Slider(
                    minimum=0.1, maximum=0.9, value=0.3, step=0.05,
                    label="置信度阈值",
                    info="越低检测越多（可能有误检），越高越严格",
                )
                max_frames_input = gr.Number(
                    value=0, label="最大处理帧数",
                    info="0 = 处理全部帧",
                    precision=0,
                )
            run_btn = gr.Button("🚀 开始检测", variant="primary", size="lg")

        with gr.Column(scale=1):
            output_video = gr.Video(label="🎯 检测结果视频")
            stats_md = gr.Markdown(label="📊 统计信息")

    run_btn.click(
        fn=run_detection,
        inputs=[input_video, confidence, max_frames_input],
        outputs=[output_video, stats_md],
    )

    gr.Markdown(
        """
        ---
        ### 技术说明

        - **检测模型**: torchvision Faster R-CNN V2 (ResNet-50 FPN, COCO 预训练)
        - **检测目标**: COCO 类别 #37 "sports ball" (涵盖高尔夫球)
        - **跟踪算法**: 线性卡尔曼滤波 (状态: [x, y, vx, vy])
        - **可视化**: 绿色框 = 检测结果，红色圆点 = 跟踪器位置，彩色线 = 运动轨迹
        """
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
