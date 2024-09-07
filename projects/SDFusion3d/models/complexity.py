# src/complexity.py
import torch
import torch.nn as nn

class ComplexityModule(nn.Module):
    def __init__(self, input_dim):
        super(ComplexityModule, self).__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        complexity = torch.sigmoid(self.fc(x))
        return complexity

def adjust_resolution(fused_feature, complexity_score, low_res_feature, threshold=0.5):
    # If complexity is high, keep the high-res feature; otherwise, use the low-res feature
    adaptive_feature = torch.where(complexity_score > threshold, fused_feature, low_res_feature)
    return adaptive_feature
