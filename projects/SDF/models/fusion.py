# src/fusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveWeight(nn.Module):
    def __init__(self, sparse_dim, dense_dim):
        super(AdaptiveWeight, self).__init__()
        self.U_s = nn.Linear(sparse_dim, 1)
        self.V_s = nn.Linear(dense_dim, 1)
        self.U_d = nn.Linear(dense_dim, 1)
        self.V_d = nn.Linear(sparse_dim, 1)
        self.b_s = nn.Parameter(torch.zeros(1))
        self.b_d = nn.Parameter(torch.zeros(1))

    def forward(self, sparse_feature, dense_feature):
        sparse_weight = torch.sigmoid(self.U_s(sparse_feature) + self.V_s(dense_feature) + self.b_s)
        dense_weight = torch.sigmoid(self.U_d(dense_feature) + self.V_d(sparse_feature) + self.b_d)
        normalized_weights = torch.cat([sparse_weight, dense_weight], dim=-1)
        normalized_weights = F.softmax(normalized_weights, dim=-1)
        return normalized_weights[:, 0], normalized_weights[:, 1]  # sparse and dense weights

def fuse_features(sparse_feature, dense_feature, sparse_weight, dense_weight):
    fused_feature = sparse_weight * sparse_feature + dense_weight * dense_feature
    return fused_feature
