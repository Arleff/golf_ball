# Use torchvision's ROIPool instead of custom C extension
import torch
from torch import nn
import torchvision


def roi_pool(input, rois, output_size, spatial_scale):
    return torchvision.ops.roi_pool(
        input, rois, output_size,
        spatial_scale=spatial_scale
    )


class ROIPool(nn.Module):
    def __init__(self, output_size, spatial_scale):
        super(ROIPool, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale

    def forward(self, input, rois):
        return torchvision.ops.roi_pool(
            input, rois, self.output_size,
            spatial_scale=self.spatial_scale
        )

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "output_size=" + str(self.output_size)
        tmpstr += ", spatial_scale=" + str(self.spatial_scale)
        tmpstr += ")"
        return tmpstr
