import numpy as np
import torch
from mmdet3d.apis import LidarDet3DInferencer
from mmdet3d.structures import Det3DDataSample
from mmdet3d.visualization import Det3DLocalVisualizer
import open3d as o3d
import matplotlib.pyplot as plt


def get_attention_map():  
    rootpath="/workspace/data/kitti_detection/model_output_results/convit3D_kitti_24_June_2024/"
  
    file ='/workspace/mmdetection3d/demo/data/kitti/000008.bin'
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
    attention_maps_info = extract_attention_maps(inferencer)
    return attention_maps_info , raw_pntcloud

def extract_attention_maps(inferencer):
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

def get_pointcloud(point:np, color:str):
    pcd = o3d.geometry.PointCloud()
    
    xyz = point[:, :3]
    print("type of pointcloud:", type(point))
    
    # Assign points to the point cloud object
    pcd.points = o3d.utility.Vector3dVector(xyz)
    
    if(point.shape[1]>3):
        ###@ giving color based on intensity
        intensity = point[:, 3]

        # Normalize intensity values to the range [0, 1]
        normalized_intensity = (intensity - np.min(intensity)) / (np.max(intensity) - np.min(intensity))

        # Map normalized intensity values to colors using a colormap (e.g., 'plasma' colormap)
        color_map = plt.get_cmap('plasma')
        points_colors = color_map(normalized_intensity)[:, :3]  # Exclude alpha channel
        pcd.colors = o3d.utility.Vector3dVector(points_colors)
        
    return pcd 

def visualize_maps(attention_map_info,raw_pointcloud):
   
    
    attention_map = attention_map_info['attention_map']
    attention_points = attention_map_info['points']
    
    attention_points = attention_points.cpu().numpy().squeeze(0)
    
    
    pcd_raw = get_pointcloud(attention_points,"")
    # vis.draw_geometries([get_pointcloud(points)], window_name="Attention Map Visualization")
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    vis.add_geometry(pcd_raw)

    vis.get_render_option().point_size = 2
    vis.run()
    

def main():
    attention_map_info,raw_pointclouds =get_attention_map()
    
    visualize_maps(attention_map_info,raw_pointclouds)
    
    
    print()


if __name__ == '__main__':
    main()
