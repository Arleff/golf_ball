#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Golf Ball Detection from Swing Videos

Reads a golf swing video file, runs Faster R-CNN frame-by-frame to detect
golf balls, and optionally applies a Kalman filter for temporal tracking.
Produces an annotated output video and/or image sequence.

Usage:
    python detect_video.py --video input.mp4 --net res101 \
        --load_dir models --checksession 1 --checkepoch 20 --checkpoint 2504 \
        [--cuda] [--kalman] [--output output.mp4] [--save_frames]
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import time
import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable

import torchvision.transforms as transforms
from model.utils.config import cfg, cfg_from_file, cfg_from_list
from model.rpn.bbox_transform import clip_boxes, bbox_transform_inv
from model.roi_layers import nms
from model.utils.net_utils import vis_detections
from model.utils.blob import im_list_to_blob
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet

try:
    xrange
except NameError:
    xrange = range


def parse_args():
    parser = argparse.ArgumentParser(
        description='Golf ball detection and tracking from swing videos')

    parser.add_argument('--video', dest='video_path',
                        help='path to input video file',
                        required=True, type=str)
    parser.add_argument('--output', dest='output_path',
                        help='path to output video file (default: <input>_detected.mp4)',
                        default=None, type=str)
    parser.add_argument('--net', dest='net',
                        help='backbone network: vgg16, res50, res101, res152',
                        default='res101', type=str)
    parser.add_argument('--dataset', dest='dataset',
                        help='dataset the model was trained on',
                        default='pascal_voc', type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--load_dir', dest='load_dir',
                        help='directory to load models',
                        default='models', type=str)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether to use CUDA',
                        action='store_true')
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load network',
                        default=20, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load network',
                        default=2504, type=int)
    parser.add_argument('--thresh', dest='thresh',
                        help='detection confidence threshold for visualization',
                        default=0.5, type=float)
    parser.add_argument('--kalman', dest='use_kalman',
                        help='enable Kalman filter tracking',
                        action='store_true')
    parser.add_argument('--save_frames', dest='save_frames',
                        help='save individual annotated frames as images',
                        action='store_true')
    parser.add_argument('--frame_dir', dest='frame_dir',
                        help='directory to save annotated frames',
                        default='output_frames', type=str)
    parser.add_argument('--max_frames', dest='max_frames',
                        help='maximum number of frames to process (0 = all)',
                        default=0, type=int)

    args = parser.parse_args()
    return args


def _get_image_blob(im):
    """Converts an image into a network input."""
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im_resized = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                                interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im_resized)

    blob = im_list_to_blob(processed_ims)
    return blob, np.array(im_scale_factors)


def detect_single_frame(fasterRCNN, im, im_data, im_info, gt_boxes, num_boxes,
                        classes, args, thresh_vis=0.5):
    """Run detection on a single BGR image. Returns annotated image and detections."""
    blobs, im_scales = _get_image_blob(im)
    assert len(im_scales) == 1

    im_blob = blobs
    im_info_np = np.array(
        [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
        dtype=np.float32)

    im_data_pt = torch.from_numpy(im_blob).permute(0, 3, 1, 2)
    im_info_pt = torch.from_numpy(im_info_np)

    with torch.no_grad():
        im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
        im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
        gt_boxes.resize_(1, 1, 5).zero_()
        num_boxes.resize_(1).zero_()

    rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

    scores = cls_prob.data
    boxes = rois.data[:, :, 1:5]

    if cfg.TEST.BBOX_REG:
        box_deltas = bbox_pred.data
        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            if args.class_agnostic:
                if args.cuda:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(
                        cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                        + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(
                        cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                        + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                box_deltas = box_deltas.view(1, -1, 4)
            else:
                if args.cuda:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(
                        cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                        + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(
                        cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                        + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                box_deltas = box_deltas.view(1, -1, 4 * len(classes))

        pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
        pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
    else:
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    pred_boxes /= im_scales[0]
    scores = scores.squeeze()
    pred_boxes = pred_boxes.squeeze()

    im2show = np.copy(im)
    all_dets = []

    for j in xrange(1, len(classes)):
        inds = torch.nonzero(scores[:, j] > 0.05).view(-1)
        if inds.numel() > 0:
            cls_scores = scores[:, j][inds]
            _, order = torch.sort(cls_scores, 0, True)
            if args.class_agnostic:
                cls_boxes = pred_boxes[inds, :]
            else:
                cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

            cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
            cls_dets = cls_dets[order]
            keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
            cls_dets = cls_dets[keep.view(-1).long()]

            dets_np = cls_dets.cpu().numpy()
            im2show = vis_detections(im2show, classes[j], dets_np, thresh_vis)

            for d in dets_np:
                if d[-1] > thresh_vis:
                    all_dets.append({
                        'class': classes[j],
                        'bbox': d[:4].tolist(),
                        'score': float(d[-1])
                    })

    return im2show, all_dets


class KalmanTracker:
    """Simple 2D Kalman filter for ball center tracking."""

    def __init__(self):
        self.A = np.array([[1, 0, 1, 0],
                           [0, 1, 0, 1],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float64)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=np.float64)
        self.Q = 0.1 * np.eye(4)
        self.R = 0.0001 * np.eye(2)
        self.x = np.zeros(4)
        self.P = np.eye(4)
        self.initialized = False
        self.trajectory = []

    def init_state(self, cx, cy):
        self.x = np.array([cx, cy, 0, 0], dtype=np.float64)
        self.P = np.eye(4)
        self.initialized = True
        self.trajectory.append((cx, cy))

    def predict(self):
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x[0], self.x[1]

    def update(self, cx, cy):
        z = np.array([cx, cy])
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(4) - K @ self.H) @ self.P
        self.trajectory.append((self.x[0], self.x[1]))

    @property
    def predicted_center(self):
        return int(self.x[0]), int(self.x[1])


def draw_trajectory(im, trajectory, color=(0, 255, 255), max_points=50):
    """Draw the ball trajectory on the image."""
    pts = trajectory[-max_points:]
    for k in range(1, len(pts)):
        thickness = max(1, int(2 * k / len(pts)) + 1)
        p1 = (int(pts[k - 1][0]), int(pts[k - 1][1]))
        p2 = (int(pts[k][0]), int(pts[k][1]))
        cv2.line(im, p1, p2, color, thickness)
    return im


def draw_info_overlay(im, frame_idx, total_frames, fps, det_count, kalman_on):
    """Draw a semi-transparent info bar at the top."""
    h, w = im.shape[:2]
    overlay = im.copy()
    cv2.rectangle(overlay, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, im, 0.4, 0, im)

    info = "Frame: {}/{} | FPS: {:.1f} | Detections: {}".format(
        frame_idx, total_frames, fps, det_count)
    if kalman_on:
        info += " | Kalman: ON"
    cv2.putText(im, info, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2)
    return im


def main():
    args = parse_args()

    if not os.path.isfile(args.video_path):
        print("ERROR: Video file not found: {}".format(args.video_path))
        sys.exit(1)

    if args.cfg_file is None:
        args.cfg_file = "cfgs/{}.yml".format(args.net)
    if os.path.isfile(args.cfg_file):
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    cfg.USE_GPU_NMS = args.cuda

    np.random.seed(cfg.RNG_SEED)

    golf_classes = np.asarray(['__background__', 'golfball'])

    input_dir = os.path.join(args.load_dir, args.net, args.dataset)
    load_name = os.path.join(
        input_dir,
        'faster_rcnn_{}_{}_{}.pth'.format(
            args.checksession, args.checkepoch, args.checkpoint))

    if not os.path.isfile(load_name):
        print("=" * 60)
        print("WARNING: Model checkpoint not found at:")
        print("  {}".format(load_name))
        print()
        print("To run detection you need a trained golf ball model.")
        print("Expected path structure:")
        print("  {}/faster_rcnn_<session>_<epoch>_<checkpoint>.pth".format(input_dir))
        print()
        print("Download the pretrained model and place it in the correct path.")
        print("See README.md for the dataset link:")
        print("  https://drive.google.com/file/d/10pzr6mDQPlrylIHg8CdXzHkF4WBMZxfn")
        print("=" * 60)
        sys.exit(1)

    if args.net == 'vgg16':
        fasterRCNN = vgg16(golf_classes, pretrained=False,
                           class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
        fasterRCNN = resnet(golf_classes, 101, pretrained=False,
                            class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
        fasterRCNN = resnet(golf_classes, 50, pretrained=False,
                            class_agnostic=args.class_agnostic)
    elif args.net == 'res152':
        fasterRCNN = resnet(golf_classes, 152, pretrained=False,
                            class_agnostic=args.class_agnostic)
    else:
        print("ERROR: network '{}' is not defined".format(args.net))
        sys.exit(1)

    fasterRCNN.create_architecture()

    print("Loading checkpoint: {}".format(load_name))
    if args.cuda:
        checkpoint = torch.load(load_name)
    else:
        checkpoint = torch.load(load_name,
                                map_location=lambda storage, loc: storage)
    fasterRCNN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint:
        cfg.POOLING_MODE = checkpoint['pooling_mode']
    print("Model loaded successfully!")

    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()
        cfg.CUDA = True
        fasterRCNN.cuda()

    fasterRCNN.eval()

    # --- Open video ---
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print("ERROR: Cannot open video: {}".format(args.video_path))
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("Video: {} | {}x{} | {:.1f} fps | {} frames".format(
        args.video_path, width, height, fps, total_frames))

    if args.output_path is None:
        base, ext = os.path.splitext(args.video_path)
        args.output_path = base + '_detected.mp4'

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(args.output_path, fourcc, fps, (width, height))

    if args.save_frames:
        os.makedirs(args.frame_dir, exist_ok=True)

    tracker = KalmanTracker() if args.use_kalman else None

    frame_idx = 0
    total_detections = 0
    total_time = 0.0

    print("\nProcessing frames...")
    print("-" * 60)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if args.max_frames > 0 and frame_idx > args.max_frames:
            break

        tic = time.time()

        im2show, dets = detect_single_frame(
            fasterRCNN, frame, im_data, im_info, gt_boxes, num_boxes,
            golf_classes, args, thresh_vis=args.thresh)

        det_time = time.time() - tic
        total_time += det_time
        current_fps = 1.0 / det_time if det_time > 0 else 0

        if tracker is not None and len(dets) > 0:
            best = max(dets, key=lambda d: d['score'])
            cx = (best['bbox'][0] + best['bbox'][2]) / 2
            cy = (best['bbox'][1] + best['bbox'][3]) / 2

            if not tracker.initialized:
                tracker.init_state(cx, cy)
            else:
                tracker.predict()
                tracker.update(cx, cy)

            im2show = draw_trajectory(im2show, tracker.trajectory)

            pcx, pcy = tracker.predicted_center
            cv2.circle(im2show, (pcx, pcy), 6, (0, 255, 255), 2)
        elif tracker is not None and tracker.initialized:
            tracker.predict()
            pcx, pcy = tracker.predicted_center
            cv2.circle(im2show, (pcx, pcy), 6, (0, 0, 255), 2)

        im2show = draw_info_overlay(
            im2show, frame_idx, total_frames, current_fps,
            len(dets), args.use_kalman)

        out_writer.write(im2show)
        total_detections += len(dets)

        if args.save_frames:
            frame_path = os.path.join(
                args.frame_dir, 'frame_{:06d}.jpg'.format(frame_idx))
            cv2.imwrite(frame_path, im2show)

        sys.stdout.write(
            '\rFrame {}/{} | {:.1f} fps | Detections in frame: {}'.format(
                frame_idx, total_frames, current_fps, len(dets)))
        sys.stdout.flush()

    cap.release()
    out_writer.release()

    avg_fps = frame_idx / total_time if total_time > 0 else 0

    print("\n" + "=" * 60)
    print("Detection complete!")
    print("  Frames processed : {}".format(frame_idx))
    print("  Total detections : {}".format(total_detections))
    print("  Average FPS      : {:.2f}".format(avg_fps))
    print("  Output video     : {}".format(args.output_path))
    if args.save_frames:
        print("  Frames saved to  : {}/".format(args.frame_dir))
    print("=" * 60)


if __name__ == '__main__':
    main()
