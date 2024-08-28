# src/refinement.py
import torch.nn as nn
import torch.nn.functional as F

class FeatureRefinement(nn.Module):
    def __init__(self, input_dim):
        super(FeatureRefinement, self).__init__()
        self.conv1 = nn.Conv3d(input_dim, input_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(input_dim, input_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(input_dim)
        self.bn2 = nn.BatchNorm3d(input_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x
