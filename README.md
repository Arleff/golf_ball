# Efficient Golf Ball Detection and Tracking Based on Convolutional Neural Networks and Kalman Filter 

We borrow the codes and implementations from [jwyang-faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch/tree/pytorch-1.0), and the Pytorch version is 1.0. Please refer to [jwyang-faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch/tree/pytorch-1.0) for more details of setups and implementations.

## Approach
Below is the framework of proposed method:

<div style="color:#0000FF" align="center">
<img src="images/Process.png" width="430"/>
</div>

## Dataset
The dataset link is [golf_ball](https://drive.google.com/file/d/10pzr6mDQPlrylIHg8CdXzHkF4WBMZxfn/view?usp=sharing).

## Results
Some tracking results are shown below:
<div style="color:#0000FF" align="center">
<img src="images/Golf_10.png" width="430"/><img src="images/Golf_16.png" width="430"/>
</div>

## Video Detection (Swing Video Golf Ball Detection)

We provide `detect_video.py` for detecting and tracking golf balls from swing video files. It reads a video frame by frame, runs Faster R-CNN detection on each frame, and optionally applies a Kalman filter for temporal tracking. The output is an annotated video with bounding boxes and trajectories.

### Quick Start

```bash
# 1. Install dependencies
pip install torch torchvision opencv-python easydict pyyaml scipy numpy matplotlib

# 2. Place your trained model checkpoint at:
#    models/res101/pascal_voc/faster_rcnn_<session>_<epoch>_<checkpoint>.pth

# 3. Run detection on a swing video
python detect_video.py --video your_swing_video.mp4 --net res101 \
    --load_dir models --checksession 1 --checkepoch 20 --checkpoint 2504

# With Kalman filter tracking enabled
python detect_video.py --video your_swing_video.mp4 --net res101 \
    --load_dir models --checksession 1 --checkepoch 20 --checkpoint 2504 \
    --kalman

# Save individual annotated frames
python detect_video.py --video your_swing_video.mp4 --net res101 \
    --load_dir models --checksession 1 --checkepoch 20 --checkpoint 2504 \
    --kalman --save_frames --frame_dir output_frames

# With CUDA acceleration (if GPU available)
python detect_video.py --video your_swing_video.mp4 --net res101 \
    --load_dir models --checksession 1 --checkepoch 20 --checkpoint 2504 \
    --cuda --kalman
```

### Arguments

| Argument | Description | Default |
|----------|------------|---------|
| `--video` | Path to input video file | (required) |
| `--output` | Path to output annotated video | `<input>_detected.mp4` |
| `--net` | Backbone network (vgg16, res50, res101, res152) | `res101` |
| `--load_dir` | Directory containing model checkpoints | `models` |
| `--cuda` | Enable CUDA GPU acceleration | off |
| `--kalman` | Enable Kalman filter for ball tracking | off |
| `--thresh` | Detection confidence threshold | `0.5` |
| `--save_frames` | Save each annotated frame as an image | off |
| `--frame_dir` | Directory for saved frames | `output_frames` |
| `--max_frames` | Max frames to process (0 = all) | `0` |
| `--cag` | Class-agnostic bbox regression | off |

### Compatibility Note

The ROI layers (`roi_align`, `roi_pool`, `nms`) have been updated to use **torchvision.ops** instead of custom C++/CUDA extensions, making the project compatible with modern PyTorch (1.x / 2.x) without requiring a C++ build step.

## Citation

    @article{zhang2020efficient,
      title={Efficient Golf Ball Detection and Tracking Based on Convolutional Neural Networks and Kalman Filter},
      author={Zhang, Tianxiao and Zhang, Xiaohan and Yang, Yiju and Wang, Zongbo and Wang, Guanghui},
      journal={arXiv preprint arXiv:2012.09393},
      year={2020}
    }
    
    
    @inproceedings{zhang2020real,
      title={Real-time golf ball detection and tracking based on convolutional neural networks},
      author={Zhang, Xiaohan and Zhang, Tianxiao and Yang, Yiju and Wang, Zongbo and Wang, Guanghui},
      booktitle={2020 IEEE International Conference on Systems, Man, and Cybernetics (SMC)},
      pages={2808--2813},
      year={2020},
      organization={IEEE}
    }
