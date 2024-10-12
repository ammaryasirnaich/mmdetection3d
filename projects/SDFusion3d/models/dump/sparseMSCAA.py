import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .window_attention import WindowAttention

class SparseMSCAA(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, sparsity_threshold=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.sparsity_threshold = sparsity_threshold
      

        # Cross-view attention layers
        # self.cross_view_attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.winAttention = WindowAttention(embed_dim=256,num_heads=8,window_size =(8,8))
        
        # Attention weights for view aggregation
        self.view_attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, img_features):
        B, N, C, H, W = img_features.shape  # Input shape: [B, N, C, H, W]
        
        # Step 1: Reshape and apply Cross-View Attention
        # img_features = rearrange(img_features, 'b n c h w -> (h w) (b n) c')
        # attn_output, _ = self.cross_view_attention(img_features, img_features, img_features)
        self.cross_view_attention = self.winAttention(img_features)
        
        # Step 2: Compute attention weights for each view
        view_weights = self.view_attention(attn_output).squeeze(-1)  # Shape: [H*W, B*N]
        view_weights = view_weights.view(B, N, H, W)
        view_weights = F.softmax(view_weights, dim=1)  # Normalize weights across views
        
        # Step 3: Apply the attention weights to aggregate features
        attn_output = rearrange(attn_output, '(h w) (b n) c -> b n c h w', b=B, n=N, h=H, w=W)
        weighted_features = attn_output * view_weights.unsqueeze(2)  # Apply weights
        
        aggregated_features = weighted_features.sum(dim=1)  # Aggregate across views
        
        # Step 4: Sparsification
        sparse_output = torch.where(aggregated_features > self.sparsity_threshold, aggregated_features, torch.tensor(0.0).to(aggregated_features.device))

        return sparse_output

# # Initialize the model
# model = SparseMSCAA(embed_dim=256, num_heads=8, sparsity_threshold=0.1)

# # Dummy input tensor (already processed by FPN)
# img_features = torch.rand(4, 6, 256, 64, 176)  # Example input

# # Forward pass
# output = model(img_features)
# print(output.shape)
# print(output)
