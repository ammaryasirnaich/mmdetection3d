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

def adjust_resolution(fused_feature, fine_feature, intermediate_feature, complexity_score, threshold=0.5):
    adaptive_feature = torch.where(complexity_score > threshold, fine_feature, intermediate_feature)
    return adaptive_feature
