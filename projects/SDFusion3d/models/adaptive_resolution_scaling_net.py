import torch
import torch.nn as nn
import torch.nn.functional as F
from .deformable_attention import DeformableAttention
from mmdet3d.registry import MODELS
from .complexity_score import Complexity_Score_Feature

@MODELS.register_module()
class MultiScaleConvolution(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv3x3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2)

        self.final_bn = nn.BatchNorm2d(in_channels * 3)

        # Optional: Use a final conv layer to restore the channel count if concatenating
        self.final_conv = nn.Conv2d(in_channels * 3, in_channels, kernel_size=1)
        
        # Initialize weights
        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')  # He initialization
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        

    def forward(self, x):
      # Convolution with ReLU activation
        F1 = F.relu(self.conv1x1(x))
        F3 = F.relu(self.conv3x3(x))
        F5 = F.relu(self.conv5x5(x))
        
        # Concatenate multi-scale features
        F_multi_scale = torch.cat([F1, F3, F5], dim=1)

        # Apply BatchNorm after concatenation (optional)
        F_multi_scale = self.final_bn(F_multi_scale)

        # Final conv to reduce back to original dimension
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

        # Intermediate network (depthwise separable convolutions)
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')  # He initialization
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                

    def forward(self, x, complexity_map, threshold=0.5):
        B, C, H, W = x.shape

        # Compute the mask for complexity
        mask_fine = complexity_map > threshold
        mask_fine = mask_fine.expand(-1, C, -1, -1)
        mask_intermediate = ~mask_fine

        # Fine-level processing for high complexity regions
        out_fine = F.relu(x + self.dilated_conv1(x))
        out_fine = F.relu(out_fine + self.dilated_conv2(out_fine))

        # Intermediate-level processing for low complexity regions
        out_intermediate = F.relu(x + self.depthwise_conv(x))
        out_intermediate = F.relu(out_intermediate + self.pointwise_conv(out_intermediate))

        # Softly combine outputs based on the complexity map
        output = mask_fine * out_fine + mask_intermediate * out_intermediate

        return output.contiguous()

@MODELS.register_module()
class AdaptiveResolutionScalingNetwork(nn.Module):
    def __init__(self, in_channels=512, n_ref_points=4, num_attention_blocks=6):
        super().__init__()
        
        # Multi-scale convolution block
        self.multi_scale_conv = MultiScaleConvolution(in_channels)
          
        
        self.complexity_score_feature = Complexity_Score_Feature()
        
        # Deformable attention block list
        self.deformable_attention_block = nn.ModuleList([DeformableAttention(in_channels, n_ref_points) for _ in range (num_attention_blocks)])
        
        # Complexity score map generation
        self.complexity_map = ComplexityScoreMap()
        
        # Adaptive residual feature refinement
        self.arfr = AdaptiveResidualFeatureRefinement(in_channels)

    def forward(self, x):
        # Step 1: Multi-scale feature extraction
        x_multi_scale = self.multi_scale_conv(x)
        
        # print(f'x_multi_scale : {x_multi_scale.shape}')
        
        x_multi_scale = self.complexity_score_feature(x_multi_scale)

      
        # Step 2: Pass through multiple deformable attention blocks
        for deformable_attention in self.deformable_attention_block:
            x_att = deformable_attention(x_multi_scale)
        
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
    model = AdaptiveResolutionScalingNetwork(in_channels=512, n_ref_points=4, num_attention_blocks=6).to(device)

    # Forward pass
    output, complexity_map = model(input_feature)

    # Print the output shapes
    print(f"Output shape: {output.shape}")         # Output shape: (4, 512, 200, 176)
    print(f"Complexity Map shape: {complexity_map.shape}")  # Complexity Map shape: (4, 1, 200, 176)
    
    
    
