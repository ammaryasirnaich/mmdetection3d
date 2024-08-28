# src/segmentation.py
import torch.nn as nn
import torch.nn.functional as F

class SegmentationHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SegmentationHead, self).__init__()
        self.conv1 = nn.Conv3d(input_dim, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(128, num_classes, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return x
