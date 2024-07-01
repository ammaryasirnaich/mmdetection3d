import numpy as np
import torch
from mmdet3d.apis import LidarDet3DInferencer
from mmdet3d.structures import Det3DDataSample
from mmdet3d.visualization import Det3DLocalVisualizer
import open3d as o3d
import matplotlib.pyplot as plt


def get_attention_map():  
    rootpath="/workspace/data/kitti_detection/model_output_results/convit3D_kitti_24_June_2024/"
  
    # file ='/workspace/mmdetection3d/demo/data/kitti/000008.bin'
    pcd_file ='/workspace/data/kitti_detection/kitti/testing/velodyne_reduced/000003.bin'
  
    # Paths to model config, checkpoint, and input data
    config_file = rootpath+'convit3D_pointNet_transformer_kitti.py'
    checkpoint_file =rootpath+'epoch_80.pth'
    point_cloud_file = pcd_file

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
    
    print("shape of point:",point.shape)
    
    xyz = point[:, :3]
  
    # Assign points to the point cloud object
    pcd.points = o3d.utility.Vector3dVector(xyz)
    # if(point.shape[1]>3):
    #     ###@ giving color based on intensity
    #     intensity = point[:, 3]

    #     # Normalize intensity values to the range [0, 1]
    #     normalized_intensity = (intensity - np.min(intensity)) / (np.max(intensity) - np.min(intensity))
    #     # Map normalized intensity values to colors using a colormap (e.g., 'plasma' colormap)
    #     color_map = plt.get_cmap(colormap)
    #     points_colors = color_map(normalized_intensity)[:, :3]  # Exclude alpha channel
    #     pcd.colors = o3d.utility.Vector3dVector(points_colors)
    return pcd 


def get_normlized_dimensions(attention_maps):
    # Normalize attention maps for visualization
    norm_dim = np.mean(attention_maps, axis=1)
    norm_dim = (norm_dim - np.min(norm_dim)) / (np.max(norm_dim) - np.min(norm_dim))
    return norm_dim



def visualize_maps(attention_map_info,raw_pointcloud):
   
    
    attention_map = attention_map_info['attention_map'].cpu().numpy().squeeze(0)
    attention_points = attention_map_info['points'].cpu().numpy().squeeze(0)
   
    point_attention = np.concatenate((attention_points,attention_map),axis=1)
    # print("shape of point attention:", point_attention.shape)
    # np.savez('poin_attention',point_attention)
    np.savez('raw_pointclouds',raw_pointcloud)
   
    # 'plasma', 'viridis'
    pcd_raw = get_pointcloud(raw_pointcloud,"None")   #"plasma"
    pcd_atten = get_pointcloud(attention_points,"None")
    # vis.draw_geometries([get_pointcloud(points)], window_name="Attention Map Visualization")
   
   
   # Set colors based on attention weights
    attention_weights = get_normlized_dimensions(attention_map)
    print("attention_weights:", attention_weights.shape)
    weight_colors = plt.get_cmap("hot")(attention_weights)[:, :3]  # Use a colormap to map attention weights to colors
    print(weight_colors.shape)
    pcd_atten.colors = o3d.utility.Vector3dVector(weight_colors)
    
    # Create a list of spheres for each point to visualize their sizes
    spheres = []
    min_size = 0.1
    max_size = 0.5
    point_sizes = min_size + (max_size - min_size) * attention_weights

    for point, color, size in zip(attention_points, weight_colors, point_sizes):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=size)
        sphere.translate(point)
        sphere.paint_uniform_color(color)  # Paint the sphere with the corresponding color
        spheres.append(sphere)

    # Combine all spheres into a single geometry
    combined_mesh = o3d.geometry.TriangleMesh()
    for sphere in spheres:
        combined_mesh += sphere


    
    nor_dim = get_normlized_dimensions(raw_pointcloud)
    raw_pnt_colors = plt.get_cmap("binary")(nor_dim)[:, :3]  # Use a colormap to map attention weights to colors
    print(raw_pnt_colors.shape)
    pcd_raw.colors = o3d.utility.Vector3dVector(raw_pnt_colors)
    
    vis = o3d.visualization.Visualizer()
    
    
    vis.create_window()
    vis.add_geometry(pcd_raw)
    vis.add_geometry(pcd_atten)
    
    vis.add_geometry(combined_mesh)

    vis.get_render_option().point_size = 2
    vis.run()
    
    
def testingsizingpointclouds():
    # Generate a hypothetical point cloud (3D points)
    num_points = 100
    point_cloud = np.random.rand(num_points, 3)  # Random points in 3D

    # Generate hypothetical attention weights
    attention_weights = np.random.rand(num_points)

    # Calculate weight colors
    weight_colors = plt.get_cmap("hot")(attention_weights)[:, :3]

    # Resize points based on attention weights
    # Normalize the attention weights to get sizes in a reasonable range for visualization
    min_size = 0.01
    max_size = 0.1
    point_sizes = min_size + (max_size - min_size) * attention_weights

    # Create an Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.colors = o3d.utility.Vector3dVector(weight_colors)
    
    # Create a list of spheres for each point to visualize their sizes
    spheres = []
    for point, color, size in zip(point_cloud, weight_colors, point_sizes):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=size)
        sphere.translate(point)
        sphere.paint_uniform_color(color)  # Paint the sphere with the corresponding color
        spheres.append(sphere)

    # Combine all spheres into a single geometry
    combined_mesh = o3d.geometry.TriangleMesh()
    for sphere in spheres:
        combined_mesh += sphere

    # Visualize the point cloud with resized points
    o3d.visualization.draw_geometries([combined_mesh])
    o3d.visualization.get_render_option().point_size = 2


def main():
    attention_map_info,raw_pointclouds =get_attention_map()
    visualize_maps(attention_map_info,raw_pointclouds)
    # testingsizingpointclouds()
    print()


if __name__ == '__main__':
    main()



