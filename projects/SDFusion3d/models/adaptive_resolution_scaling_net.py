import torch
import torch.nn as nn
import torch.nn.functional as F
from .deformable_attention_norm import DeformableAttention
from mmdet3d.registry import MODELS


@MODELS.register_module()
class MultiScaleConvolution(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv3x3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2)

        # Optional: BatchNorm
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn3 = nn.BatchNorm2d(in_channels)
        self.bn5 = nn.BatchNorm2d(in_channels)

        # Optional: Use a final conv layer to restore the channel count if concatenating
        self.final_conv = nn.Conv2d(in_channels * 3, in_channels, kernel_size=1)

    def forward(self, x):
        # Convolution with activation and optional normalization
        F1 = F.relu(self.bn1(self.conv1x1(x)))
        F3 = F.relu(self.bn3(self.conv3x3(x)))
        F5 = F.relu(self.bn5(self.conv5x5(x)))
        
        # Optional: Concatenate or sum the multi-scale features
        F_multi_scale = torch.cat([F1, F3, F5], dim=1)  # Concatenate along the channel dimension

        # Optional: Use a final 1x1 conv to merge channels back to original dimension
        F_multi_scale = self.final_conv(F_multi_scale)

        return F_multi_scale



class ComplexityScoreMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Compute the mean across the channel dimension
        complexity_map = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        return complexity_map


@MODELS.register_module()
class AdaptiveResidualFeatureRefinement(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        # Fine network (dilated residual blocks)
        self.dilated_conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=2, dilation=2)
        self.dilated_conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=4, dilation=4)
        self.bn_dilated1 = nn.BatchNorm2d(in_channels)
        self.bn_dilated2 = nn.BatchNorm2d(in_channels)

        # Intermediate network (depthwise separable convolutions)
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.bn_depthwise = nn.BatchNorm2d(in_channels)
        self.bn_pointwise = nn.BatchNorm2d(in_channels)

    def forward(self, x, complexity_map, threshold=0.5):
        B, C, H, W = x.shape

        # Compute the mask for complexity (broadcasted across channels)
        mask_fine = complexity_map > threshold  # Shape: [B, 1, H, W]
        mask_fine = mask_fine.expand(-1, C, -1, -1)  # Broadcast across the channels, shape: [B, C, H, W]
        mask_intermediate = ~mask_fine  # Inverse of mask_fine, shape: [B, C, H, W]

        # Fine-level processing for high complexity regions (with residual connection and batchnorm)
        out_fine = F.relu(x + self.bn_dilated1(self.dilated_conv1(x)))
        out_fine = F.relu(out_fine + self.bn_dilated2(self.dilated_conv2(out_fine)))

        # Intermediate-level processing for low complexity regions (with residual connection and batchnorm)
        out_intermediate = F.relu(x + self.bn_depthwise(self.depthwise_conv(x)))
        out_intermediate = F.relu(out_intermediate + self.bn_pointwise(self.pointwise_conv(out_intermediate)))

        # Softly combine outputs based on the complexity map
        output = mask_fine * out_fine + mask_intermediate * out_intermediate

        return output.contiguous()

@MODELS.register_module()
class AdaptiveResolutionScalingNetwork(nn.Module):
    def __init__(self, in_channels=512, n_ref_points=4):
        super().__init__()
        
        # Multi-scale convolution block
        self.multi_scale_conv = MultiScaleConvolution(in_channels)
          
        # Deformable attention block
        self.deformable_attention = DeformableAttention(in_channels, n_ref_points)
        
        # Complexity score map generation
        self.complexity_map = ComplexityScoreMap()
        
        # Adaptive residual feature refinement
        self.arfr = AdaptiveResidualFeatureRefinement(in_channels)

    def forward(self, x):
        # Step 1: Multi-scale feature extraction
        x_multi_scale = self.multi_scale_conv(x)
        
        # print(f'x_multi_scale : {x_multi_scale.shape}')
        
        
        # Step 2: Deformable attention
        x_att = self.deformable_attention(x_multi_scale)
        
        # print(f'Deformable attention feature shape : {x_att.shape}')
        
        # Step 3: Complexity score map
        complexity_map = self.complexity_map(x_att)
        
        # print(f'complexity_map : {complexity_map.shape}')
        
        # Step 4: Adaptive feature refinement
        output = self.arfr(x_att, complexity_map)
        
        # print(f'output : {output.shape}')
        
        return output


if __name__=="__main__":
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define the input feature map with size (B=4, C=512, H=200, W=176)
    input_feature = torch.randn(4, 512, 180, 180).to(device)

    # Initialize the model
    model = AdaptiveResolutionScalingNetwork(in_channels=512, n_ref_points=4).to(device)

    # Forward pass
    output, complexity_map = model(input_feature)

    # Print the output shapes
    print(f"Output shape: {output.shape}")         # Output shape: (4, 512, 200, 176)
    print(f"Complexity Map shape: {complexity_map.shape}")  # Complexity Map shape: (4, 1, 200, 176)
    
    
    
