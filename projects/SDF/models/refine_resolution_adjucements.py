import torch.nn as nn
import torch.nn.functional as F
import torch

from .fusion import AdaptiveWeight, fuse_features
from .refinement import FeatureRefinement
from .complexity import ComplexityModule, adjust_resolution



class Refine_Resolution_Adjacement(nn.Module):
    def __init__(self):
        super().__init__()
        self.adaptive_weight = AdaptiveWeight(sparse_dim=256, dense_dim=256)
        self.refinement = FeatureRefinement(input_dim=256)
        self.complexity_module = ComplexityModule(input_dim=256)
        

    def forward(self,sparse_features,dense_feature):
                
        # Global pooling for dense features to match sparse feature dimensions
        dense_pooled = [torch.mean(f, dim=[2, 3, 4], keepdim=True) for f in dense_feature]
        dense_pooled = torch.stack(dense_pooled, dim=0).sum(dim=0)  # Aggregate multi-view features


        # Adaptive Fusion
        sparse_weight, dense_weight = self.adaptive_weight(sparse_features, dense_pooled)
        fused_feature = fuse_features(sparse_features, dense_pooled, sparse_weight, dense_weight)

        # Refinement and Resolution Adjustment
        refined_feature = self.refinement(fused_feature)
        complexity_score = self.complexity_module(refined_feature)
        adaptive_feature = adjust_resolution(refined_feature, refined_feature, refined_feature, complexity_score)

        # Final segmentation
        # predictions = self.segmentation_head(adaptive_feature)
        # return predictions
        
        
        return adaptive_feature