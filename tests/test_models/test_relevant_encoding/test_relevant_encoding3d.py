import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path
from matplotlib.transforms import Affine2D

class RelPositionalEncoding(nn.Module):
    def __init__(self, input_dim, max_points):
        super(RelPositionalEncoding, self).__init__()
        self.input_dim = input_dim
        self.max_points = max_points
        
        self.position_encoding = nn.Embedding(self.max_points, self.input_dim)

    def forward(self, points):
        '''
        points: 3D point cloud (B,N,D)
        return: relevant position encoding cooridnates(3) with pairwise eucliden distance(1) (B,N,N,4) 
        
        '''
        batch_size, num_points, _ = points.size()
        
        # Compute relative coordinates
        relative_coords = points[:, :, None, :] - points[:, None, :, :]
        
        # Compute pairwise distances
        distances = torch.sqrt(torch.sum(relative_coords ** 2, dim=-1))  # Euclidean distance
        
        # Compute position encoding
        position_indices = torch.arange(num_points, device=points.device).unsqueeze(0).expand(batch_size, -1)
        position_encodings = self.position_encoding(position_indices)
        
        # Expand position encodings to match the shape of distances
        position_encodings = position_encodings.unsqueeze(2).expand(-1, -1, num_points, -1)
        
        # Concatenate position encodings with distances
        encodings = torch.cat([position_encodings, distances.unsqueeze(-1)], dim=-1)
        
        return encodings


def positional_encoding_3d(point_cloud):
    '''
        points: 3D point cloud (N,D)
        return: relevant pairwise eucliden distance(1) (N,N)  
        
        '''
    
    # Compute pairwise squared Euclidean distances
    pairwise_distances = torch.sum((point_cloud.unsqueeze(1) - point_cloud) ** 2, dim=-1)

    # Optional using explCompute the relevant encoding 
    encoding = torch.exp(-pairwise_distances)

    return encoding




def test_RelPositionalEncoding():
    #Usage example
    max_seq_len = 100
    points = torch.randn(1,max_seq_len, 3)  # Example input tensor of shape (batch_size, num_heads, max_seq_len, 3)
    print("Input shape", points.shape)
    pos_encoding = RelPositionalEncoding(3,100)
    output = pos_encoding(points)
    print("output",output.shape)
    distance_feature = output[:,:,:,3]
    #optional to use exp or not
    # distance_feature = torch.exp(-distance_feature)

    # print("output",distance_feature.squeeze(0).shape)
    plt.imshow(distance_feature.squeeze(0).detach().numpy())


def test_positional_encoding_3d():
    # Example usage
    point_cloud = torch.randn(100, 3)  # Example point cloud with 100 points in 3D
    encoding = positional_encoding_3d(point_cloud)
    print(encoding.shape)
    plt.imshow(encoding)




if __name__ == "__main__":
    test_RelPositionalEncoding()
    test_positional_encoding_3d()

  