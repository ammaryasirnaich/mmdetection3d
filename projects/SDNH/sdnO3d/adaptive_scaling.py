# mm3d/models/necks/adaptive_scaling.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveResolutionScaling(nn.Module):
    '''
    Initialization:

    Creates multiple scales, each consisting of a convolution and activation.

    Forward Method:
    For each scale:
        Downsamples the input features by a factor of 2^idx.
        Applies convolution and activation.
    Returns a list of scaled feature maps.
    '''
    
    def __init__(self, num_scales, in_channels, out_channels):
        super(AdaptiveResolutionScaling, self).__init__()
        self.scales = nn.ModuleList()
        for _ in range(num_scales):
            self.scales.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True)
                )
            )

    def forward(self, features):
        scaled_feats = []
        for idx, scale in enumerate(self.scales):
            scaled_feat = F.interpolate(features, scale_factor=1/(2**idx), mode='bilinear', align_corners=False)
            scaled_feat = scale(scaled_feat)
            scaled_feats.append(scaled_feat)
        return scaled_feats
