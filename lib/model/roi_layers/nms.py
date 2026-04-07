# Use torchvision's NMS instead of custom C extension
import torchvision

nms = torchvision.ops.nms
