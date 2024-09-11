import torch
import torch.nn as nn
import torch.nn.functional as F

class DeformableAttention(nn.Module):
    def __init__(self, in_channels, n_ref_points=4):
        super(DeformableAttention, self).__init__()
        self.n_ref_points = n_ref_points

        # Learnable query, key, value projection layers
        self.query_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # Offset prediction for reference points
        self.offset_proj = nn.Conv2d(in_channels, 2 * n_ref_points, kernel_size=1)  # 2 coords (x, y) for each ref point

    def forward(self, x):
        B, C, H, W = x.shape

        # Project to queries, keys, and values
        Q = self.query_proj(x)  # [B, C, H, W]
        K = self.key_proj(x)    # [B, C, H, W]
        V = self.value_proj(x)  # [B, C, H, W]

        # Predict offsets for the reference points
        offsets = self.offset_proj(Q)  # [B, 2 * n_ref_points, H, W]
        offsets = offsets.view(B, self.n_ref_points, 2, H, W)  # Reshape to [B, n_ref_points, 2 (x, y), H, W]

        # Generate meshgrid of the original positions (h, w)
        grid_h, grid_w = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        grid_w = grid_w.to(x.device)
        grid_h = grid_h.to(x.device)

        # Repeat the grid to match the batch size and number of reference points
        grid_w = grid_w.unsqueeze(0).unsqueeze(0).expand(B, self.n_ref_points, H, W)
        grid_h = grid_h.unsqueeze(0).unsqueeze(0).expand(B, self.n_ref_points, H, W)

        # Compute the reference points by adding the offsets
        ref_w = torch.clamp(grid_w + offsets[:, :, 0, :, :], min=0, max=W-1)
        ref_h = torch.clamp(grid_h + offsets[:, :, 1, :, :], min=0, max=H-1)

        # Flatten the height and width for batched index selection
        ref_w = ref_w.long().view(B, self.n_ref_points, -1)  # Shape: [B, n_ref_points, H*W]
        ref_h = ref_h.long().view(B, self.n_ref_points, -1)  # Shape: [B, n_ref_points, H*W]

        # Compute 1D indices for the reference points from 2D indices
        ref_indices = ref_h * W + ref_w  # Convert 2D indices to 1D, shape: [B, n_ref_points, H*W]

        # Reshape ref_indices to [B, n_ref_points * H * W] for broadcasting
        ref_indices = ref_indices.view(B, -1)  # Shape: [B, n_ref_points * H * W]

        # Flatten K and V for gathering
        K_flat = K.view(B, C, -1)  # Shape: [B, C, H*W]
        V_flat = V.view(B, C, -1)  # Shape: [B, C, H*W]

        # Expand ref_indices to match the shape of K_flat along batch and channel dimensions
        ref_indices = ref_indices.unsqueeze(1).expand(B, C, ref_indices.size(1))  # [B, C, n_ref_points * H * W]

        # Gather the keys and values at the reference points
        K_gathered = torch.gather(K_flat, 2, ref_indices)  # [B, C, n_ref_points * H * W]
        V_gathered = torch.gather(V_flat, 2, ref_indices)  # [B, C, n_ref_points * H * W]

        # Reshape Q_flat and K_gathered for the einsum operation
        Q_flat = Q.view(B, C, -1)  # Shape: [B, C, H*W]
        K_gathered = K_gathered.view(B, C, self.n_ref_points, -1)  # Shape: [B, C, n_ref_points, H*W]
        V_gathered = V_gathered.view(B, C, self.n_ref_points, -1)  # Shape: [B, C, n_ref_points, H*W]

        # Compute the attention weights using einsum (correct equation)
        attention_weights = torch.einsum('bch,bcnh->bnh', Q_flat, K_gathered)  # [B, n_ref_points, H*W]

        # Apply the attention weights to the gathered values using einsum
        output = torch.einsum('bnh,bcnh->bch', attention_weights, V_gathered).contiguous()  # [B, C, H*W]
        
        # Reshape output back to [B, C, H, W]
        output = output.view(B, C, H, W)

        return output

if __name__=="__main__":
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define the input feature map with size (B=4, C=512, H=200, W=176)
    input_feature = torch.randn(4, 512, 200, 176).to(device)

    # Initialize the model
    model = DeformableAttention(in_channels=512, n_ref_points=4).to(device)

    # Forward pass
    output = model(input_feature)

    # Print the output shapes
    print(f"Output shape: {output.shape}")         # Output shape: (4, 512, 200, 176)
   
    
    
    