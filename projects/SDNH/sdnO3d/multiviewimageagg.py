import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# Load pre-trained ResNet50 model
model = models.resnet50(pretrained=True)
model = nn.Sequential(*list(model.children())[:-2])  # Remove the fully connected layers
model.eval()  # Set the model to evaluation mode

# Define a transformation to preprocess the input images
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(img_path):
    img = Image.open(img_path).convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        features = model(img_tensor)
    return features

def aggregate_features(features_list):
    # Concatenate features along the channel dimension
    aggregated_features = torch.cat(features_list, dim=1)
    return aggregated_features

def project_to_3d(aggregated_features, transformation_matrix):
    # Assuming transformation_matrix is a homography matrix
    h, w = aggregated_features.shape[2], aggregated_features.shape[3]
    points_2d = torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w)), dim=0).reshape(2, -1).float()
    points_2d = torch.cat((points_2d, torch.ones(1, points_2d.shape[1])), dim=0)
    
    # Convert transformation matrix to a torch tensor
    transformation_matrix = torch.tensor(transformation_matrix, dtype=torch.float32)
    
    # Apply the transformation matrix to project into 3D
    points_3d = torch.matmul(transformation_matrix, points_2d)
    points_3d /= points_3d[2, :]  # Normalize by the third row
    return points_3d

# Example usage
features_view1 = extract_features('view1.jpg')
features_view2 = extract_features('view2.jpg')
aggregated_features = aggregate_features([features_view1, features_view2])

# Example transformation matrix (identity matrix for simplicity)
transformation_matrix = np.eye(3)
points_3d = project_to_3d(aggregated_features, transformation_matrix)
