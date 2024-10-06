import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the function to compute entropy-based complexity
def compute_entropy_complexity(x):
    B, C, H, W = x.shape
    
    # Normalize the features across the channels to obtain probabilities
    F_normalized = F.softmax(dim=1)  # Normalize along the channel dimension (C)
    
    # Compute entropy across the channel dimension (C)
    entropy = -torch.sum(F_normalized * torch.log(F_normalized + 1e-6), dim=1)  # Shape [B, H, W]
    # print(f'entropy {entropy.shape}')
    
    return entropy

# Define the SoftmaxAttentionModule that uses entropy complexity
class Complexity_Score_Feature(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        # Step 1: Compute the entropy-based complexity score map
        complexity_score_map = compute_entropy_complexity(x)  # Shape [B, H, W]
        
        # Step 2: Apply softmax to the complexity score map to get attention weights
        attention_weights = torch.softmax(complexity_score_map.view(x.size(0), -1), dim=-1)
        attention_weights = attention_weights.view_as(complexity_score_map)  # Reshape back to [B, H, W]
        
        # Step 3: Multiply the attention weights with the feature map (expand dimensions to match feature map)
        F_attention = F * attention_weights.unsqueeze(1)  # Unsqueeze to add channel dimension for multiplication
        
        return F_attention


if __name__=="__main__":
    # Example usage
    F = torch.randn(8, 256, 64, 64)  # Example feature map [B, C, H, W]

    # Initialize the attention module
    attention_module = Complexity_Score_Weight()
    # Apply the attention module to the feature map
    F_attention = attention_module(F)
    print(F_attention.shape)

