import torch
import torch.nn as nn
import torch.nn.functional as F

class DeformableAttention(nn.Module):
    def __init__(self, in_channels, num_heads=8, num_points=4):
        super(DeformableAttention, self).__init__()
        self.num_heads = num_heads
        self.num_points = num_points
        self.in_channels = in_channels
        
        # Learnable offset weights (for sampling locations)
        self.offset_conv = nn.Conv2d(in_channels, num_heads * num_points * 2, kernel_size=3, padding=1)
        
        # Learnable offset weights (for sampling locations)
        self.offset_conv = nn.Conv2d(in_channels, num_heads * num_points * 2, kernel_size=3, padding=1)
        self.offset_bn = nn.BatchNorm2d(num_heads * num_points * 2)  # Apply BatchNorm after offset convolution
        
        # Attention weights for each sampled point
        self.attention_weights_conv = nn.Conv2d(in_channels, num_heads * num_points, kernel_size=3, padding=1)
        self.attention_weights_bn = nn.BatchNorm2d(num_heads * num_points)  # Apply BatchNorm after attention weights conv
        
        # Output projection layer to mix attended features
        self.output_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # Layer Normalization after attention
        self.layer_norm = nn.LayerNorm(in_channels)
        
        # Residual Connection
        self.residual = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.residual_bn = nn.BatchNorm2d(in_channels)  # BatchNorm for residual connection
    

        # Cache for meshgrid, only recompute when necessary
        self.cached_grid_size = None
        self.cached_meshgrid = None

    def generate_sampling_grids(self, offsets, H, W):
        """
        Generate sampling grids based on deformable offsets.
        Args:
            offsets: Deformable offsets with shape [B, num_heads * num_points * 2, H, W]
            H, W: Spatial height and width of the feature map
        Returns:
            Sampling grids with shape [B, num_heads, num_points, H, W, 2]
        """
        B, _, h, w = offsets.shape
        num_heads = self.num_heads
        num_points = self.num_points

        # Cache the base grid and reuse if the grid size hasn't changed
        if (H, W) != self.cached_grid_size:
            base_grid_y, base_grid_x = torch.meshgrid(torch.arange(H, device=offsets.device), torch.arange(W, device=offsets.device), indexing='ij')
            base_grid = torch.stack([base_grid_x, base_grid_y], dim=-1).float()  # Shape [H, W, 2]

            # Expand base grid to match the dimensions of offsets
            base_grid = base_grid.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # Shape [1, 1, 1, H, W, 2]
            self.cached_meshgrid = base_grid
            self.cached_grid_size = (H, W)

        # Retrieve the cached grid
        base_grid = self.cached_meshgrid

        # Reshape offsets to match grid shape [B, num_heads, num_points, H, W, 2]
        offsets = offsets.view(B, num_heads, num_points, 2, H, W).permute(0, 1, 2, 4, 5, 3)

        # Add offsets to the base grid
        sampling_grid = base_grid + offsets

        return sampling_grid
    
    def sample_features(self, x, sampling_grids):
        """
        Sample features from the input feature map using sampling grids.
        Args:
            x: Input feature map with shape [B, C, H, W]
            sampling_grids: Sampling grids with shape [B, num_heads, num_points, H, W, 2]
        Returns:
            Sampled features with shape [B, num_heads, num_points, C, H, W]
        """
        B, C, H, W = x.shape
        num_heads = self.num_heads
        num_points = self.num_points
        
        # Normalize sampling grids to the range [-1, 1]
        sampling_grids = 2.0 * sampling_grids / torch.tensor([W, H], device=x.device) - 1.0

        # Sample features using bilinear interpolation
        x_repeated = x.unsqueeze(2).repeat(1, 1, num_points * num_heads, 1, 1)  # Shape [B, C, num_points * num_heads, H, W]
        x_repeated = x_repeated.view(B * num_heads * num_points, C, H, W)  # Reshape for sampling
        sampling_grids = sampling_grids.view(B * num_heads * num_points, H, W, 2)

        sampled_features = F.grid_sample(x_repeated, sampling_grids, mode='bilinear', align_corners=True)  # [B * num_heads * num_points, C, H, W]
        sampled_features = sampled_features.view(B, num_heads, num_points, C, H, W)
        
        return sampled_features

    def forward(self, x):
        """
        Args:
            x: Input feature map with shape [B, C, H, W]
        Returns:
            Output feature map with the same shape [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # Generate deformable offsets [B, num_heads * num_points * 2, H, W]
        offsets = self.offset_conv(x)
        offsets = self.offset_bn(offsets)  # Apply BatchNorm after offset convolution
        
        # Generate attention weights [B, num_heads * num_points, H, W]
        attention_weights = self.attention_weights_conv(x)
        attention_weights = self.attention_weights_bn(attention_weights)  # Apply BatchNorm after attention weights conv
        attention_weights = attention_weights.view(B, self.num_heads, self.num_points, H, W)
        attention_weights = torch.softmax(attention_weights, dim=2)
        
        
        # Create sampling grid based on the offsets
        sampling_grids = self.generate_sampling_grids(offsets, H, W)
        
        # Sample from the input feature map using the sampling grids
        sampled_features = self.sample_features(x, sampling_grids)
        
        # Apply attention weights to the sampled features
        attention_weights = attention_weights.unsqueeze(3)  # Shape becomes [B, num_heads, num_points, 1, H, W]
        weighted_features = (attention_weights * sampled_features).sum(dim=2)  # Shape [B, num_heads, C, H, W]
        
        # Collapse heads by summing them
        output = weighted_features.sum(dim=1)  # Shape [B, C, H, W]
        
        # Output projection
        output = self.output_proj(output)
        
        # Apply residual connection and layer normalization
        output = self.layer_norm(output.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        output = self.residual_bn(output + self.residual(x)) 
        
        
        return output


if __name__=="__main__":
    
    B, C, H, W = 4, 512, 180, 180  # Input shape
     
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    
    input_tensor = torch.randn(B, C, H, W).to(device)  # Random input tensor

    # Initialize DeformableAttention layer
    deformable_attention = DeformableAttention(in_channels=C, num_heads=8, num_points=4).to(device)

    # Forward pass with the deformable attention module
    output_tensor = deformable_attention(input_tensor)

    # Print output shape
    print("Output shape:", output_tensor.shape)
        
    
    # Set device
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # # Define the input feature map with size (B=4, C=512, H=200, W=176)
    # x = torch.randn(4, 512, 180, 180).to(device)

    # # Initialize the model
    # model = DeformableAttention(in_channels=512, n_ref_points=4).to(device)
    
    # output = model(x)
    # assert output.shape == x.shape, "Output shape mismatch."
  
    # # Test with varying input sizes
    # x = torch.randn(2, 256, 200, 200).to(device)
    # output = model(x)
    # assert output.shape == x.shape, "Output shape mismatch for different input size."

    # # Test with large offsets (should be clamped)
    # model.offset_proj.weight.data.fill_(10.0)
    # model.offset_proj.bias.data.fill_(0.0)
    # output = model(x)
    # assert (output == 0).sum() == 0, "Output should not be all zeros."

        
        

    # Print the output shapes
    # print(f"Output shape: {output.shape}")         # Output shape: (4, 512, 200, 176)
   
    