import torch
import torch.nn as nn
import torch.nn.functional as F

class DeformableAttention(nn.Module):
    def __init__(self, in_channels, n_ref_points=4):
        super(DeformableAttention, self).__init__()
        self.n_ref_points = n_ref_points

        # Projection layers for query, key, and value
        self.query_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # Offset projection for reference points (2 coordinates per ref point)
        self.offset_proj = nn.Conv2d(in_channels, 2 * n_ref_points, kernel_size=1)

        # Initialize meshgrid cache
        self.register_buffer('grid_h', None, persistent=False)
        self.register_buffer('grid_w', None, persistent=False)

    def forward(self, x):
        """
        Forward pass for Deformable Attention.

        Args:
            x (Tensor): Input tensor of shape [B, C, H, W].

        Returns:
            Tensor: Output tensor after deformable attention of shape [B, C, H, W].
        """
        B, C, H, W = x.shape

        # Project input to query, key, and value
        Q = self.query_proj(x)  # [B, C, H, W]
        K = self.key_proj(x)    # [B, C, H, W]
        V = self.value_proj(x)  # [B, C, H, W]

        # Predict offsets for reference points
        offsets = self.offset_proj(Q)  # [B, 2 * n_ref_points, H, W]
        offsets = offsets.view(B, self.n_ref_points, 2, H, W)  # [B, n_ref_points, 2, H, W]

        # Generate meshgrid if not cached or if input size has changed
        if (self.grid_h is None or self.grid_w is None or
            self.grid_h.shape[-2:] != (H, W)):
            grid_h, grid_w = torch.meshgrid(
                torch.arange(H, device=x.device),
                torch.arange(W, device=x.device),
                indexing='ij'
            )  # Each of shape [H, W]

            # Expand to [B, n_ref_points, H, W]
            grid_h = grid_h.unsqueeze(0).unsqueeze(1).expand(B, self.n_ref_points, H, W)
            grid_w = grid_w.unsqueeze(0).unsqueeze(1).expand(B, self.n_ref_points, H, W)

            self.grid_h = grid_h  # [B, n_ref_points, H, W]
            self.grid_w = grid_w  # [B, n_ref_points, H, W]
        else:
            grid_h = self.grid_h  # [B, n_ref_points, H, W]
            grid_w = self.grid_w  # [B, n_ref_points, H, W]

        # Compute reference points by adding offsets to the grid
        ref_x = (grid_w + offsets[:, :, 0]).round().long().clamp(0, W - 1)  # [B, n_ref_points, H, W]
        ref_y = (grid_h + offsets[:, :, 1]).round().long().clamp(0, H - 1)  # [B, n_ref_points, H, W]

        # Compute flattened reference indices
        ref_indices = ref_y * W + ref_x  # [B, n_ref_points, H, W]

        # Ensure that all ref_indices are within [0, H*W - 1]
        if not torch.isfinite(ref_indices).all():
            raise ValueError("Reference indices contain NaN or inf values.")

        if torch.any(ref_indices < 0) or torch.any(ref_indices >= H * W):
            raise ValueError("Reference indices are out of bounds.")

        # Flatten K and V for gathering
        K_flat = K.view(B, C, -1)  # [B, C, H*W]
        V_flat = V.view(B, C, -1)  # [B, C, H*W]

        # Reshape ref_indices for gathering
        # Original shape: [B, n_ref_points, H, W]
        ref_indices = ref_indices.view(B, self.n_ref_points, -1)  # [B, n_ref_points, H*W]
        ref_indices = ref_indices.unsqueeze(1).expand(B, C, self.n_ref_points, H * W)  # [B, C, n_ref_points, H*W]
        ref_indices = ref_indices.contiguous().view(B, C, -1)  # [B, C, n_ref_points * H * W]

        # Gather keys and values at reference points
        K_gathered = torch.gather(K_flat, 2, ref_indices)  # [B, C, n_ref_points * H * W]
        V_gathered = torch.gather(V_flat, 2, ref_indices)  # [B, C, n_ref_points * H * W]

        # Reshape for attention computation
        K_gathered = K_gathered.view(B, C, self.n_ref_points, H * W)  # [B, C, n_ref_points, H*W]
        V_gathered = V_gathered.view(B, C, self.n_ref_points, H * W)  # [B, C, n_ref_points, H*W]

        Q_flat = Q.view(B, C, -1)  # [B, C, H*W]

        # Compute attention weights using dot product between Q and K_gathered
        # [B, C, H*W] x [B, C, n_ref_points, H*W] -> [B, n_ref_points, H*W]
        attention_weights = torch.einsum('bch,bcnh->bnh', Q_flat, K_gathered)  # [B, n_ref_points, H*W]

        # Apply softmax to attention weights along the reference points dimension
        attention_weights = F.softmax(attention_weights, dim=1)  # [B, n_ref_points, H*W]

        # Apply attention weights to V_gathered
        # [B, n_ref_points, H*W] x [B, C, n_ref_points, H*W] -> [B, C, H*W]
        output = torch.einsum('bnh,bcnh->bch', attention_weights, V_gathered)  # [B, C, H*W]

        # Reshape output back to [B, C, H, W]
        output = output.view(B, C, H, W)  # [B, C, H, W]

        return output

    
if __name__=="__main__":
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define the input feature map with size (B=4, C=512, H=200, W=176)
    input_feature = torch.randn(4, 512, 180, 180).to(device)

    # Initialize the model
    model = DeformableAttention(in_channels=512, n_ref_points=4).to(device)

    # Forward pass
    for i in range(2):
        output = model(input_feature)

    # Print the output shapes
    print(f"Output shape: {output.shape}")         # Output shape: (4, 512, 200, 176)
   
    
    
    