# mm3d/models/heads/occupancy_head.py


import torch
import torch.nn as nn
import torch.nn.functional as F

class OccupancyPredictionHead(nn.Module):
    def __init__(self, in_channels, grid_size):
        super(OccupancyPredictionHead, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, in_channels//2, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(in_channels//2, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.grid_size = grid_size

    def forward(self, fused_feats):
        # Assume fused_feats is a list of 2D feature maps
        # First, we need to project them into 3D space
        # For simplicity, we'll stack and reshape them
        batch_size = fused_feats[0].shape[0]
        x = torch.stack(fused_feats, dim=1)  # Shape: (B, N, C, H, W)
        x = x.view(batch_size, -1, self.grid_size[0], self.grid_size[1], self.grid_size[2])
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        occupancy = self.sigmoid(x)
        return occupancy
