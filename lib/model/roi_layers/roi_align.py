# Use torchvision's ROIAlign instead of custom C extension
import torch
from torch import nn
import torchvision


def roi_align(input, rois, output_size, spatial_scale, sampling_ratio):
    return torchvision.ops.roi_align(
        input, rois, output_size,
        spatial_scale=spatial_scale,
        sampling_ratio=sampling_ratio
    )


class ROIAlign(nn.Module):
    def __init__(self, output_size, spatial_scale, sampling_ratio):
        super(ROIAlign, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio

    def forward(self, input, rois):
        return torchvision.ops.roi_align(
            input, rois, self.output_size,
            spatial_scale=self.spatial_scale,
            sampling_ratio=self.sampling_ratio
        )

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "output_size=" + str(self.output_size)
        tmpstr += ", spatial_scale=" + str(self.spatial_scale)
        tmpstr += ", sampling_ratio=" + str(self.sampling_ratio)
        tmpstr += ")"
        return tmpstr
