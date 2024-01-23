
import sys
import numpy as np
import open3d as o3d

import time
   


# cloud = o3d.geometry.PointCloud()

ply_point_cloud = o3d.data.PLYPointCloud()

pcd = o3d.io.read_point_cloud(ply_point_cloud.path)

visualizer = o3d.visualization.Visualizer()
visualizer.create_window()
visualizer.add_geometry(pcd)
view_ctl = visualizer.get_view_control()


# Initial zoom
initial_zoom =0.2  # incresing the zoom value will zoom out the view
view_ctl.set_zoom(initial_zoom)


view_ctl.rotate(0, 1000)  # Adjust rotation values as neede
# view_ctl.set_up((0, 0, 1))  # set the positive direction of the x-axis as the up direction
# view_ctl.set_up((0, -1, 0))  # set the negative direction of the y-axis as the up direction
# view_ctl.set_front((1, 0, 0))  # set the positive direction of the x-axis toward you
# view_ctl.set_lookat((0, 0, 0))  # set the original point as the center point of the window


 # Initialize rotation and zoom variables
rotation_angle = 0
zoom_level = initial_zoom


visualizer.poll_events()
visualizer.update_renderer()
visualizer.run()
visualizer.capture_screen_image("/workspace/data/kitti_detection/open_chair.png")

