import torch.nn as nn
import torch.nn.functional as F
import torch

from .fusion import AdaptiveWeight
# from .refinement import FeatureRefinement
from .complexity import ComplexityModule, adjust_resolution
from .window_attention import WindowAttention
# from .multiviewAdapFusion import Multiview_AdaptiveWeightedFusion
from einops import rearrange


class MultiScaleConvolution(nn.Module):
    def __init__(self, in_channels):
        super(MultiScaleConvolution, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv3x3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2)

    def forward(self, x):
        F1 = self.conv1x1(x)
        F3 = self.conv3x3(x)
        F5 = self.conv5x5(x)
        F_multi_scale = F1 + F3 + F5
        return F_multi_scale


class DeformableAttention(nn.Module):
    def __init__(self, in_channels, n_ref_points=4):
        super(DeformableAttention, self).__init__()
        self.n_ref_points = n_ref_points
        
        # Learnable query, key, value projection layers
        self.query_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # Offset prediction for reference points
        self.offset_proj = nn.Conv2d(in_channels, 2 * n_ref_points, kernel_size=1)  # 2 coords (x,y) for each ref point

    def forward(self, x):
        B, C, H, W = x.shape

        # Project to queries, keys, and values
        Q = self.query_proj(x)  # [B, C, H, W]
        K = self.key_proj(x)    # [B, C, H, W]
        V = self.value_proj(x)  # [B, C, H, W]

        # Predict offsets for the reference points
        offsets = self.offset_proj(Q)  # [B, 2 * n_ref_points, H, W]
        offsets = offsets.view(B, self.n_ref_points, 2, H, W)  # Reshape to [B, n_ref_points, 2 (x,y), H, W]

        # Initialize output tensor
        output = torch.zeros_like(Q).to(x.device)  # Output tensor to accumulate the results

        for b in range(B):
            for h in range(H):
                for w in range(W):
                    # Get query for the current position (h, w)
                    query = Q[b, :, h, w]

                    # Collect reference points using offsets
                    for i in range(self.n_ref_points):
                        offset_x, offset_y = offsets[b, i, :, h, w]
                        ref_x = int(min(max(w + offset_x, 0), W - 1))
                        ref_y = int(min(max(h + offset_y, 0), H - 1))

                        # Get corresponding key and value
                        key = K[b, :, ref_y, ref_x]
                        value = V[b, :, ref_y, ref_x]

                        # Compute attention score
                        attention_weight = torch.dot(query, key) / torch.sqrt(torch.tensor(C, dtype=torch.float32))
                        attention_weight = torch.sigmoid(attention_weight)  # Use sigmoid to limit the range

                        # Accumulate the weighted value
                        output[b, :, h, w] += attention_weight * value

        return output


class ComplexityScoreMap(nn.Module):
    def __init__(self):
        super(ComplexityScoreMap, self).__init__()

    def forward(self, x):
        # Compute the mean across the channel dimension
        complexity_map = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        return complexity_map



class AdaptiveResidualFeatureRefinement(nn.Module):
    def __init__(self, in_channels):
        super(AdaptiveResidualFeatureRefinement, self).__init__()
        
        # Fine network (dilated residual blocks)
        self.dilated_conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=2, dilation=2)
        self.dilated_conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=4, dilation=4)
        
        # Intermediate network (depthwise separable convolutions)
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x, complexity_map, threshold=0.5):
        B, C, H, W = x.shape
        
        # Initialize output tensor
        output = torch.zeros_like(x)
        
        for b in range(B):
            for h in range(H):
                for w in range(W):
                    if complexity_map[b, 0, h, w] > threshold:
                        # Fine-level processing for high complexity regions
                        out_fine = F.relu(x[b, :, h, w] + self.dilated_conv1(x[b, :, h, w]))
                        out_fine = F.relu(out_fine + self.dilated_conv2(out_fine))
                        output[b, :, h, w] = out_fine
                    else:
                        # Intermediate-level processing for low complexity regions
                        out_intermediate = F.relu(self.depthwise_conv(x[b, :, h, w]))
                        out_intermediate = F.relu(self.pointwise_conv(out_intermediate))
                        output[b, :, h, w] = out_intermediate
                        
        return output






class Refine_Resolution_Adjacement(nn.Module):
    def __init__(self):
        super().__init__()
         
        self.adaptive_weight = AdaptiveWeight(voxel_dim=512, image_dim=64,upscale_size=(200, 176)).cuda()
        self.complexity_module = ComplexityModule(input_dim=512)
        
    def forward(self,sparse_features,dense_feature):
        # sparse_features : # [B, N, 256, H, W]
        # dense_feature   : #  [B, C, H, W]
                      
        # Adaptive Fusion between lidar and image features
        fused_feature = self.adaptive_weight(sparse_features[0], dense_feature)
        
        # fused_feature = fuse_features(sparse_features[0], upscaled_image_feature, sparse_weight, dense_weight)

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
    