import torch.nn as nn
import torch.nn.functional as F
import torch

from .fusion import AdaptiveWeight, fuse_features
from .refinement import FeatureRefinement
from .complexity import ComplexityModule, adjust_resolution
from .window_attention import WindowAttention
from .multiviewAdapFusion import Multiview_AdaptiveWeightedFusion
from einops import rearrange



class MultiViewAttentionAggregation(nn.Module):
    def __init__(self, dense_dim, num_heads=4):
        super(MultiViewAttentionAggregation, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=dense_dim, num_heads=num_heads, batch_first=True)
        
    def forward(self, dense_features):
        # Assuming dense_features is of shape [B, N, C, H, W]
        B, N, C, H, W = dense_features.shape
        
        # Reshape to [B, N, C*H*W]
        dense_features = dense_features.view(B, N, -1)
        
        # Apply multi-head attention (keys, queries, values are the same in self-attention)
        aggregated_features, _ = self.attention(dense_features, dense_features, dense_features)
        
        # Reshape back to [B, C, H, W] after summing over the view dimension
        aggregated_features = aggregated_features.sum(dim=1).view(B, C, H, W)
        
        return aggregated_features


class Refine_Resolution_Adjacement(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.multiview_adap_fusion_model = Multiview_AdaptiveWeightedFusion(num_views=6)
        self.adaptive_weight = AdaptiveWeight(voxel_dim=512, image_dim=256,upscale_size=(200, 176)).cuda()
        self.refinement = FeatureRefinement(input_dim=512)
        self.complexity_module = ComplexityModule(input_dim=512)
        # self.multi_view_attn_agg = MultiViewAttentionAggregation(dense_dim=256)
        # self.winAttention = WindowAttention(embed_dim=256,num_heads=8,window_size =(8,8))
        # self.sparseMscaa =  SparseMSCAA(embed_dim=256, num_heads=8, sparsity_threshold=0.1)
   
    def forward(self,sparse_features,dense_feature):
        # sparse_features : # [B, N, 256, H, W]
        # dense_feature   : #  [B, C, H, W]
                      
         # Apply multi-view attention aggregation
        # dense_pooled = self.multi_view_attn_agg(dense_feature)  # Outputs [B, 256, H, W]
        # dense_sparse_feature = self.winAttention(dense_feature)
        # dense_sparse_feature = self.sparseMscaa(dense_feature)
        # image_agg_feature =self.multiview_adap_fusion_model(dense_feature)  # fusing the multi-view
        # dense_feature = rearrange(dense_feature,'b n c h w d f ->(b n h w d) (cf) ')
        
        
        # dense_feature = rearrange(dense_feature,'b c h w ->(b n h w d) (cf) ')

        # Adaptive Fusion between lidar and image features
        sparse_weight, dense_weight, upscaled_image_feature = self.adaptive_weight(sparse_features[0], dense_feature)
        
        fused_feature = fuse_features(sparse_features[0], upscaled_image_feature, sparse_weight, dense_weight)

        # Refinement 
        refined_feature = self.refinement(fused_feature) # refinement outputs [B, 512, ...]
        
        # Compute complexity score for adaptive resolution
        complexity_score = self.complexity_module(refined_feature)  
             
        # Simulate low-res feature for the purpose of this example
        low_res_feature = torch.mean(refined_feature, dim=[2, 3], keepdim=True)  # Downscaling the refined feature

        # Adaptive Resolution Scaling based on complexity
        adaptive_feature = adjust_resolution(refined_feature, complexity_score, low_res_feature)

        # Final segmentation
        # predictions = self.segmentation_head(adaptive_feature)
        # return predictions
        
        
        return adaptive_feature
        

if __name__=="__main__":
    batch_size = 4
    num_views = 6
    num_classes = 10
    height, width = 64, 176
    channel = 256
    voxel_channel=512
    bev_hight=200
    bev_width=176
    dense_image_feature = torch.rand(batch_size, num_views, channel, height, width)
    sparse_feature      = torch.rand(batch_size, voxel_channel, bev_hight, bev_width)
    
    
    model = Refine_Resolution_Adjacement()
    out = model(sparse_feature,dense_image_feature)
    print(out.shape)
    print("pass")
    