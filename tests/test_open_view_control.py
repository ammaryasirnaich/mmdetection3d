
import sys
import numpy as np
import open3d as o3d

# cloud = o3d.geometry.PointCloud()
ply_point_cloud = o3d.data.PLYPointCloud()

pcd = o3d.io.read_point_cloud(ply_point_cloud.path)

visualizer = o3d.visualization.Visualizer()
visualizer.create_window()
visualizer.add_geometry(pcd)
view_ctl = visualizer.get_view_control()
# view_ctl=  view_ctl.convert_to_pinhole_camera_parameters()  # noqa: E501
# view_ctl.scale(0)
# visualizer.update_renderer()
# visualizer.run()

# #


# view_ctl.set_zoom(1)
# view_ctl.rotate(90)
view_ctl.set_up((0, 0, 1))  # set the positive direction of the x-axis as the up direction
view_ctl.set_up((0, -1, 0))  # set the negative direction of the y-axis as the up direction
view_ctl.set_front((1, 0, 0))  # set the positive direction of the x-axis toward you
view_ctl.set_lookat((0, 0, 0))  # set the original point as the center point of the window
visualizer.update_renderer()
visualizer.run()
# 