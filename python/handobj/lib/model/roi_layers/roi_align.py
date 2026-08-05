# Modern torchvision ROIAlign replacement for old custom _C ROIAlign
import torch
from torch import nn
from torch.nn.modules.utils import _pair
from torchvision.ops import roi_align as torchvision_roi_align


def roi_align(input, rois, output_size, spatial_scale, sampling_ratio):
    output_size = _pair(output_size)
    rois = rois.to(dtype=input.dtype, device=input.device)

    return torchvision_roi_align(
        input,
        rois,
        output_size=output_size,
        spatial_scale=spatial_scale,
        sampling_ratio=sampling_ratio,
        aligned=False,
    )


class ROIAlign(nn.Module):
    def __init__(self, output_size, spatial_scale, sampling_ratio):
        super(ROIAlign, self).__init__()
        self.output_size = _pair(output_size)
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio

    def forward(self, input, rois):
        return roi_align(
            input,
            rois,
            self.output_size,
            self.spatial_scale,
            self.sampling_ratio,
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"output_size={self.output_size}, "
            f"spatial_scale={self.spatial_scale}, "
            f"sampling_ratio={self.sampling_ratio})"
        )