import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the importance score network
class ImportanceScoreNet(nn.Module):
    def __init__(self, in_channels=256):
        super(ImportanceScoreNet, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, 1)
    
    def forward(self, x):
        x = self.global_pool(x)  # Global Average Pooling
        x = torch.flatten(x, 1)  # Flatten to (batch_size, channels)
        x = torch.sigmoid(self.fc(x))  # Importance score (0 to 1)
        return x

# Define the task-specific prediction network (e.g., segmentation)
# class PredictionHead(nn.Module):
#     def __init__(self, in_channels=256, num_classes=10):
#         super(PredictionHead, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
#         self.bn1 = nn.BatchNorm2d(in_channels)
#         self.conv2 = nn.Conv2d(in_channels, num_classes, kernel_size=1, stride=1)
    
#     def forward(self, x):
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = self.conv2(x)  # Output logits for each class
#         return x

# Define the main model for Adaptive Weighted Feature Fusion
class Multiview_AdaptiveWeightedFusion(nn.Module):
    def __init__(self, num_views=4, num_classes=10):
        super(Multiview_AdaptiveWeightedFusion, self).__init__()
        self.num_views = num_views
        self.importance_score_net = ImportanceScoreNet(in_channels=256)
        # self.prediction_head = PredictionHead(in_channels=256, num_classes=num_classes)
    
    def forward(self, x):
        # x has shape (B, N, 256, H, W)
        batch_size, num_views, channels, height, width = x.shape
        assert num_views == self.num_views, "Number of views must match the model's num_views"
        
        features = []
        scores = []
        
        for i in range(self.num_views):
            # Extract features for each view (no additional feature extraction is needed as input is already a feature map)
            feat = x[:, i, :, :, :]  # Shape: (B, 256, H, W)
            features.append(feat)
            
            # Calculate importance score for each view
            score = self.importance_score_net(feat)
            scores.append(score)
        
        # Convert list to tensors
        features = torch.stack(features, dim=0)  # Shape: (N, B, 256, H, W)
        scores = torch.stack(scores, dim=0)      # Shape: (N, B, 1)
        
        # Normalize scores across views (softmax normalization)
        scores = F.softmax(scores, dim=0)  # Normalize across views
        
        # Weighted feature fusion
        fused_features = torch.sum(scores.unsqueeze(-1).unsqueeze(-1) * features, dim=0)  # Shape: (B, 256, H, W)
        
        # # Task-specific prediction
        # output = self.prediction_head(fused_features)
        
        return fused_features

# Example usage
if __name__ == "__main__":
    # Example input with shape (4, 6, 256, 64, 176)
    batch_size = 4
    num_views = 6
    num_classes = 10
    height, width = 64, 176
    channel = 256
    x = torch.rand(batch_size, num_views, channel, height, width)
    
    # Instantiate the model
    model = Multiview_AdaptiveWeightedFusion(num_views=num_views, num_classes=num_classes)
    
    # Forward pass
    result = model(x)
    
    print(result.shape)  # Expected output shape: (batch_size, num_classes, H, W)
