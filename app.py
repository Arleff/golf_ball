"""
Gradio Web App: Golf Ball Detection & Tracking from Swing Videos (v2)

Upload a golf swing video, optionally provide auxiliary hints
(impact frame, ball position annotations), and the system will:
1. Auto-detect the impact frame (or use your hint)
2. Run two-pass detection + parabolic trajectory fitting
3. Output annotated video with smooth trajectory overlay
"""

import os
import json
import tempfile
import gradio as gr
from demo_video import process_video


def run_detection(video_file, confidence_threshold, max_frames,
                  impact_frame, annotations_text, gravity):
    if video_file is None:
        raise gr.Error("请上传一个视频文件！")

    max_frames_int = int(max_frames) if max_frames else 0
    impact_hint = int(impact_frame) if impact_frame and int(impact_frame) >= 0 else None
    grav = float(gravity) if gravity else 0.5

    annotations = []
    if annotations_text and annotations_text.strip():
        try:
            annotations = json.loads(annotations_text)
            if not isinstance(annotations, list):
                raise ValueError
        except (json.JSONDecodeError, ValueError):
            raise gr.Error(
                '标注格式错误，应为 JSON 数组，如:\n'
                '[{"frame":10, "x":500, "y":300}]'
            )

    with tempfile.NamedTemporaryFile(suffix="_detected.mp4", delete=False) as tmp:
        output_path = tmp.name

    try:
        output_path, stats = process_video(
            input_path=video_file,
            output_path=output_path,
            score_threshold=confidence_threshold,
            max_frames=max_frames_int,
            impact_frame_hint=impact_hint,
            annotations=annotations,
            gravity=grav,
        )
    except Exception as e:
        raise gr.Error(f"处理视频时出错: {str(e)}")

    det_rate = stats['frames_with_detection'] / max(stats['total_frames'], 1) * 100
    para_status = (f"已拟合 ({stats['parabola_inliers']} 内点)"
                   if stats['parabola_fitted'] else "未拟合（观测点不足）")

    summary = (
        f"### 检测结果统计\n\n"
        f"| 指标 | 数值 |\n"
        f"|------|------|\n"
        f"| 总帧数 | {stats['total_frames']} |\n"
        f"| 击球帧 | {stats.get('impact_frame', 'N/A')} |\n"
        f"| 检测到球的帧数 | {stats['frames_with_detection']} |\n"
        f"| 检测率 | {det_rate:.1f}% |\n"
        f"| CNN 检测数 | {stats.get('cnn_detections', 0)} |\n"
        f"| 运动检测数 | {stats.get('motion_detections', 0)} |\n"
        f"| 抛物线轨迹 | {para_status} |\n"
        f"| 平均检测耗时 | {stats['avg_detect_time']:.3f} 秒/帧 |\n"
    )

    return output_path, summary


with gr.Blocks(title="高尔夫球检测与跟踪 v2") as demo:
    gr.Markdown(
        """
        # Golf Ball Detection & Tracking v2

        基于 **Faster R-CNN + 卡尔曼滤波(重力模型) + 抛物线拟合** 的高尔夫球检测与跟踪。

        ### 使用方法
        1. 上传挥杆视频
        2. (可选) 填写 **击球帧** 编号——不填则自动检测
        3. (可选) 填写 **辅助标注**——在某些帧上标出球的位置帮助系统锁定球
        4. 点击 **开始检测**

        ---
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_video = gr.Video(label="上传挥杆视频")

            with gr.Accordion("基本参数", open=True):
                with gr.Row():
                    confidence = gr.Slider(
                        minimum=0.1, maximum=0.9, value=0.25, step=0.05,
                        label="CNN 置信度阈值",
                    )
                    max_frames_input = gr.Number(
                        value=0, label="最大帧数", info="0=全部",
                        precision=0,
                    )

            with gr.Accordion("辅助标记（提升识别准确率）", open=True):
                impact_frame_input = gr.Number(
                    value=-1, label="击球帧编号",
                    info="-1 = 自动检测。如果你知道第几帧击球，填入可大幅提升准确率",
                    precision=0,
                )
                gravity_input = gr.Slider(
                    minimum=0.0, maximum=3.0, value=0.5, step=0.1,
                    label="重力系数 (px/frame²)",
                    info="球飞行的下坠速度，默认 0.5。远距离挥杆可适当增大",
                )
                annotations_input = gr.Textbox(
                    label="球位置标注 (JSON)",
                    placeholder='[{"frame":15, "x":620, "y":380}, {"frame":25, "x":750, "y":320}]',
                    info="在你能看见球的帧上标出球心坐标，系统会以此为锚点增强跟踪",
                    lines=3,
                )

            run_btn = gr.Button("开始检测", variant="primary", size="lg")

        with gr.Column(scale=1):
            output_video = gr.Video(label="检测结果")
            stats_md = gr.Markdown(label="统计信息")

    run_btn.click(
        fn=run_detection,
        inputs=[input_video, confidence, max_frames_input,
                impact_frame_input, annotations_input, gravity_input],
        outputs=[output_video, stats_md],
    )

    gr.Markdown(
        """
        ---
        ### 技术说明

        | 模块 | 方法 |
        |------|------|
        | CNN 检测 | Faster R-CNN V2 (COCO "sports ball") |
        | 运动检测 | 帧差法 + 轮廓分析（仅击球后激活） |
        | 跟踪 | 卡尔曼滤波（重力加速度模型 + Mahalanobis 门控） |
        | 轨迹 | RANSAC 抛物线拟合（x 线性 + y 二次） |
        | 可视化 | 绿框=CNN检测 / 橙框=运动检测 / 红十字=跟踪器 / 曲线=拟合轨迹 |
        """
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
