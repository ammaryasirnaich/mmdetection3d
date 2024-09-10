import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func

class ComplexityModule(nn.Module):
    def __init__(self, input_channels, embed_dim, num_heads):
        super(ComplexityModule, self).__init__()
        
        # Multi-scale convolutions for local and global context extraction
        self.conv1x1 = nn.Conv2d(input_channels, input_channels, kernel_size=1)
        self.conv3x3 = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(input_channels, input_channels, kernel_size=5, padding=2)
        
        # Transformer self-attention parameters
        self.query_proj = nn.Linear(input_channels, embed_dim)
        self.key_proj = nn.Linear(input_channels, embed_dim)
        self.value_proj = nn.Linear(input_channels, embed_dim)   
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        
    def forward(self, x):
        # x: Tensor of shape (B, C, H, W)
        B, C, H, W = x.shape
        
        # Multi-scale convolutions
        F1 = self.conv1x1(x)   # (B, C, H, W)
        F3 = self.conv3x3(x)   # (B, C, H, W)
        F5 = self.conv5x5(x)   # (B, C, H, W)
        
        # Combine multi-scale features
        F_multi_scale = F1 + F3 + F5  # (B, C, H, W)
        
        # Reshape to (B, C, H*W) for attention
        F_flat = F_multi_scale.view(B, C, -1)  # (B, C, H*W)
        
        # Compute query, key, and value matrices
        Q = self.query_proj(F_flat.permute(2, 0, 1))  # (H*W, B, embed_dim)
        K = self.key_proj(F_flat.permute(2, 0, 1))    # (H*W, B, embed_dim)
        V = self.value_proj(F_flat.permute(2, 0, 1))  # (H*W, B, embed_dim)
        
        # Apply transformer-style multihead attention
        attn_output = flash_attn_func(Q, K, V, dropout_p=0.0, softmax_scale=None, causal=False)
        # attn_output, _ = self.attention(Q, K, V)  # (H*W, B, embed_dim)
    
        
        # Reshape back to (B, C, H, W)
        F_att = attn_output.permute(1, 2, 0).view(B, C, H, W)
        
        # Compute complexity score by averaging across channels
        complexity_score = torch.mean(F_att, dim=1, keepdim=True)  # (B, 1, H, W)
        
        return F_att, complexity_score

# Example usage

if __name__=="__main__":
    # Check if a GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")



    B, C, H, W = 4, 256, 200, 176 # Example dimensions
    fused_features = torch.randn(B, C, H, W).to(device)

    # Initialize the Multi-Scale Transformer Attention Module
    mstam = ComplexityModule(input_channels=C, embed_dim=128, num_heads=8).to(device)

    # Perform forward pass
    attention_features, complexity_score = mstam(fused_features)

    print("Attention Features Shape:", attention_features.shape)
    print("Complexity Score Shape:", complexity_score.shape)








# # src/complexity.py
# import torch
# import torch.nn as nn

# class ComplexityModule(nn.Module):
#     def __init__(self, input_dim):
#         super(ComplexityModule, self).__init__()
#         self.fc = nn.Linear(input_dim, 1)

#     def forward(self, x):
#         complexity = torch.sigmoid(self.fc(x))
#         return complexity

# def adjust_resolution(fused_feature, complexity_score, low_res_feature, threshold=0.5):
#     # If complexity is high, keep the high-res feature; otherwise, use the low-res feature
#     adaptive_feature = torch.where(complexity_score > threshold, fused_feature, low_res_feature)
#     return adaptive_feature
