import torch
import torch.nn as nn
from einops import rearrange

class WindowAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size):
        super().__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, img_features):
        B, N, C, H, W = img_features.shape
        
        # Unpack window size into separate dimensions
        ws1, ws2 = self.window_size
        
        # Make sure the height and width are divisible by the respective window sizes
        assert H % ws1 == 0 and W % ws2 == 0, "Height and Width must be divisible by the respective window sizes."
        
        # Correctly rearrange the dimensions
        img_features = rearrange(img_features, 'b n c (h ws1) (w ws2) -> b n (h w) (ws1 ws2) c', 
                                 ws1=ws1, ws2=ws2, h=H // ws1, w=W // ws2)
        
        # Reshape for attention
        windowed_features = img_features.reshape(-1, ws1 * ws2, C)
        
        attn_output, _ = self.attention(windowed_features, windowed_features, windowed_features)
        
        # Reshape back to original
        return attn_output.view(B, N, H, W, -1)