import torch
import torch.nn as nn
import torch.nn.functional as F

from deformable_attention import DeformableAttention




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




class AdaptiveResolutionScalingNetwork(nn.Module):
    def __init__(self, in_channels=512, n_ref_points=4):
        super(AdaptiveResolutionScalingNetwork, self).__init__()
        
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
        
        print(f'x_multi_scale : {x_multi_scale.shape}')
        
        
        # Step 2: Deformable attention
        x_att = self.deformable_attention(x_multi_scale)
        
        # Step 3: Complexity score map
        complexity_map = self.complexity_map(x_att)
        
        # Step 4: Adaptive feature refinement
        output = self.arfr(x_att, complexity_map)
        
        return output, complexity_map





if __name__=="__main__":
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define the input feature map with size (B=4, C=512, H=200, W=176)
    input_feature = torch.randn(4, 512, 200, 176).to(device)

    # Initialize the model
    model = AdaptiveResolutionScalingNetwork(in_channels=512, n_ref_points=4).to(device)

    # Forward pass
    output, complexity_map = model(input_feature)

    # Print the output shapes
    print(f"Output shape: {output.shape}")         # Output shape: (4, 512, 200, 176)
    print(f"Complexity Map shape: {complexity_map.shape}")  # Complexity Map shape: (4, 1, 200, 176)
    
    
    
