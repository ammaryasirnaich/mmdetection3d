# mm3d/models/necks/hierarchical_fusion.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalFeatureFusion(nn.Module):
    '''
    Initialization:

    conv_img and conv_pts: 1x1 convolutions to align the channel dimensions of image and 
                           point cloud features.
    fusion_conv: 3x3 convolution to fuse the features.
    relu: Activation function.
    
    Forward Method:
    Iterates over feature maps from both modalities.
    Resizes point cloud features to match the spatial dimensions of image features.
    Applies convolution and fuses features via addition followed by activation.
    Returns a list of fused feature maps
    
    '''
    def __init__(self, in_channels_img, in_channels_pts, fusion_channels):
        super(HierarchicalFeatureFusion, self).__init__()
        self.conv_img = nn.Conv2d(in_channels_img, fusion_channels, kernel_size=1)
        self.conv_pts = nn.Conv2d(in_channels_pts, fusion_channels, kernel_size=1)
        self.fusion_conv = nn.Conv2d(fusion_channels, fusion_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, img_feats, pts_feats):
        # Assume img_feats and pts_feats are lists of feature maps at different scales
        fused_feats = []
        for img_feat, pts_feat in zip(img_feats, pts_feats):
            # Resize pts_feat to match img_feat size
            pts_feat_resized = F.interpolate(pts_feat, size=img_feat.shape[2:], mode='bilinear', align_corners=False)
            img_conv = self.conv_img(img_feat)
            pts_conv = self.conv_pts(pts_feat_resized)
            fusion = self.relu(img_conv + pts_conv)
            fusion = self.fusion_conv(fusion)
            fused_feats.append(fusion)
        return fused_feats
