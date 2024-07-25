import numpy as np
import torch
from mmdet3d.apis import LidarDet3DInferencer
from mmdet3d.structures import Det3DDataSample
from mmdet3d.visualization import Det3DLocalVisualizer
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.image as mpimg
import copy

from scipy.stats import norm 
import statistics 



def extract_layer_attent_maps(inferencer):
    attention_maps = {}
    neck = inferencer.model.neck
    # print("no of block", len(neck.blocks))
    last_layer_attention = neck.blocks[-1]
    # print("last_layer_attention type:",type(last_layer_attention.attn))
    # print("last_layer_attention shape:",last_layer_attention.attn.atttion_map.shape)
    # print("last_layer_attention cooridnate shape:",neck.voxel_coord.shape)
    attention_maps['attention_map']=last_layer_attention.attn.atttion_map
    attention_maps['points'] = neck.voxel_coord
   
    
    # for block in inferencer.model.neck.blocks:
    #     print("block:",type(block))
    #     print("module:",module)
    #     if 'attention' in name.lower() and hasattr(module, 'attn_probs'):
    #         attention_maps.append((name, module.attn_probs))
    
    return attention_maps

def get_pointcloud(point:np, colormap:str):
    pcd = o3d.geometry.PointCloud()
    xyz = point[:, :3]
    # Assign points to the point cloud object
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd 


def get_normlized_dimensions(attention_maps):
    # Normalize attention maps for visualization
    norm_dim = np.mean(attention_maps, axis=1)
    norm_dim = (norm_dim - np.min(norm_dim)) / (np.max(norm_dim) - np.min(norm_dim))
    return norm_dim

def get_attention_map(config_file_path,chckpoint_file_path,pcd_path):  

    config_file = config_file_path
    checkpoint_file =chckpoint_file_path
    point_cloud_file = pcd_path

    raw_pntcloud = np.fromfile(point_cloud_file, dtype=np.float32).reshape(-1, 4)
    
    # Create LidarDet3DInferencer instance
    inferencer = LidarDet3DInferencer(
        model=config_file,
        weights=checkpoint_file,
        device='cuda:0'
    )

    # Perform inference
    result = inferencer(
        inputs=dict(points=raw_pntcloud),
        return_datasamples=True
    )
    
    ## Extract predictions
    # det3DdataSample = result['predictions'][0].pred_instances_3d
    # Extract attention maps
    attention_maps_info = extract_layer_attent_maps(inferencer)
    return attention_maps_info , raw_pntcloud


def get_3d_scene_data(attention_map_info,raw_pointcloud):
    attention_map = attention_map_info['attention_map'].cpu().numpy().squeeze(0)
    attention_points = attention_map_info['points'].cpu().numpy().squeeze(0)   
 
    # 1,2 Normalize the attention weights with Averaging Across Dimensions on attention_map
    normalized_attention_weights = get_normlized_dimensions(attention_map)
 
   # Use a colormap to map attention weights to colors

    # 3. Creating Spheres for each point in the point cloud
    colors = [(1, 1, 1), (1, 1, 0), (1, 0, 0)]  # White to yellow to red
    n_bins = 100  # Discretize the colormap
    cmap_name = 'custom_attention'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    # Map attention weights to colors using the custom colormap
    weight_colors = cm(normalized_attention_weights)[:, :3]

    # Create spheres for visualization
    spheres = []
    min_size = 0.01
    max_size = 0.5
    point_sizes = min_size + (max_size - min_size) * normalized_attention_weights

    for point, color, size in zip(attention_points, weight_colors, point_sizes):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=size)
        sphere.translate(point)
        sphere.paint_uniform_color(color)  # Paint the sphere with the corresponding color
        spheres.append(sphere)

    # 5. Combining Geometries into a single TriangleMesh object for visualization
    attnt_points_spheres = o3d.geometry.TriangleMesh()
    for sphere in spheres:
        attnt_points_spheres += sphere
        
        
    pcd_raw = get_pointcloud(raw_pointcloud,"None")   #"plasma"
    raw_pnt_nor_dim = get_normlized_dimensions(raw_pointcloud)
    raw_pnt_colors = plt.get_cmap("binary")(raw_pnt_nor_dim)[:, :3]  # Use a colormap to map attention weights to colors
    pcd_raw.colors = o3d.utility.Vector3dVector(raw_pnt_colors)
    
    return pcd_raw,attnt_points_spheres


def visualize_maps(attention_map_info,raw_pointcloud,image_path,to_plot_image_barplot=True):
        
    pcd_raw, attnt_points = get_3d_scene_data(attention_map_info,raw_pointcloud)
    vis = o3d.visualization.Visualizer()
    
    vis.create_window()
    vis.add_geometry(pcd_raw)
    vis.add_geometry(attnt_points)
    vis.get_render_option().point_size = 2
    vis.run()
    
    if to_plot_image_barplot==True:    
        # Display the 2D camera image
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        image = mpimg.imread(image_path)
        ax[0].imshow(image)
        ax[0].axis('off')
        ax[0].set_title('2D Camera Image')

        # Display the color bar
        norm = plt.Normalize(vmin=0, vmax=1)
        sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax[1],orientation='vertical', label='Attention Weights')
        ax[1].axis('off')

        plt.tight_layout()
        plt.show()
        print("pass")
        
   

def testingsizingpointclouds(attention_map_info,raw_pointcloud):

    # Assuming attention_map_info is a dictionary containing 'attention_map' and 'points'
    # For demonstration, we'll create dummy data
    # attention_map_info = {
    #     'attention_map': torch.randn(512, 256),
    #     'points': torch.randn(512, 3)
    # }

    # Extract attention map and points from the dictionary
    attention_map = attention_map_info['attention_map'].cpu().numpy().squeeze(0)
    attention_points = attention_map_info['points'].cpu().numpy().squeeze(0)

    # 1. Averaging Across Dimensions on attention_map to reduce the dimension
    averaged_attention = attention_map.mean(axis=1)

    # 2. Normalize the attention weights
    min_val = averaged_attention.min()
    max_val = averaged_attention.max()
    normalized_attention_weights = (averaged_attention - min_val) / (max_val - min_val)

    # 3. Creating Spheres for each point in the point cloud
    colors = [(1, 1, 1), (1, 1, 0), (1, 0, 0)]  # White to yellow to red
    n_bins = 100  # Discretize the colormap
    cmap_name = 'custom_attention'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
    
    # Map attention weights to colors using the custom colormap
    weight_colors = cm(normalized_attention_weights)[:, :3]

    # Create spheres for visualization
    spheres = []
    min_size = 0.01
    max_size = 0.5
    point_sizes = min_size + (max_size - min_size) * normalized_attention_weights

    for point, color, size in zip(attention_points, weight_colors, point_sizes):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=size)
        sphere.translate(point)
        sphere.paint_uniform_color(color)  # Paint the sphere with the corresponding color
        spheres.append(sphere)

    # 5. Combining Geometries into a single TriangleMesh object for visualization
    attnt_points = o3d.geometry.TriangleMesh()
    for sphere in spheres:
        attnt_points += sphere

    # 4. Visualize the normalized weights using Open3D
    # o3d.visualization.draw_geometries([attnt_points])

    # 6. Add a color bar to the visualization
    # Generate and save the color bar
    plt.figure(figsize=(6, 1))
    norm = plt.Normalize(vmin=0, vmax=1)
    cm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
    # cm.set_array([])
    plt.colorbar(cm, orientation='vertical', label='Attention Weights')
    plt.show()
    # plt.savefig('colorbar.png', bbox_inches='tight')

   
def colorbarplot():
    cm = plt.cm.get_cmap('RdYlBu')
    xy = range(20)
    z = xy
    colors = [(1, 1, 1), (1, 1, 0), (1, 0, 0)]  # White to yellow to red
    n_bins = 100  # Discretize the colormap
    cmap_name = 'custom_attention'
    norm = plt.Normalize(vmin=0, vmax=1)
    cm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
    # cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
    plt.colorbar(cm, orientation='horizontal', label='Attention Weights')
    plt.show()


def plot_gaussian(attention_weights):
   
    # Normalize the attention weights to have a mean of 0 and standard deviation of 1
    mean_weight = np.mean(attention_weights)
    std_weight = np.std(attention_weights)
    normalized_weights = (attention_weights - mean_weight) / std_weight

    # Fit a Gaussian distribution
    mu, sigma = norm.fit(normalized_weights)

    # Generate x values for plotting the Gaussian distribution
    x = np.linspace(min(normalized_weights), max(normalized_weights), 1000)

    # Calculate the Gaussian distribution values
    gaussian_distribution = norm.pdf(x, mu, sigma)

    # Plot the Gaussian distribution
    plt.figure(figsize=(10, 6))
    plt.plot(x, gaussian_distribution, label=f'Gaussian fit: $\mu$={mu:.2f}, $\sigma$={sigma:.2f}')
    plt.hist(normalized_weights, bins=30, density=True, alpha=0.6, color='g', label='Histogram of attention weights')
    plt.title('Gaussian Distribution Fitted to Attention Weights')
    plt.xlabel('Value')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.show()

def main():
    rootpath="/workspace/data/kitti_detection/model_output_results/convit3D_kitti_24_June_2024/"

    # scene_id='000010'
    # scene_id='000003'
    scene_id='000008'
    pcd_file ='/workspace/data/kitti_detection/kitti/testing/velodyne_reduced/'+scene_id+'.bin'
    image_path = '/workspace/data/kitti_detection/kitti/testing/image_2/'+scene_id+'.png'
    ###Paths to model config, checkpoint, and input data
    config_file = rootpath+'convit3D_pointNet_transformer_kitti.py'
    checkpoint_file =rootpath+'epoch_80.pth'
    
    
    attention_map_info,raw_pointclouds =get_attention_map(config_file,checkpoint_file,pcd_file)
    visualize_maps(attention_map_info,raw_pointclouds,image_path,to_plot_image_barplot=False)
    # testingsizingpointclouds(attention_map_info,raw_pointclouds)
    # colorbarplot()
    print()


if __name__ == '__main__':
    main()



