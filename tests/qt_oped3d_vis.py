import sys
import numpy as np
import open3d as o3d
from PySide2.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget
from multiprocessing import Process, Pipe

def run_open3d_visualizer(conn):
    # Sample data
    cloud = o3d.geometry.PointCloud()
    
    pcd_file_path = "/home/ammar/Documents/PntCloud_sample_dataset/armadillo.ply"
    
    ply_point_cloud = o3d.data.PLYPointCloud()
    
    pcd = o3d.io.read_point_cloud(ply_point_cloud.path)
    # cloud.points = o3d.utility.Vector3dVector([
    #     [0, 0, 0],
    #     [1, 0, 0],
    #     [0, 1, 0],
    #     [0, 0, 1]
    # ])

    # Create Open3D visualizer
    # cloud.point = pcd.points
    # cloud.color = pcd.colors
    

    print(pcd)
    print(np.asarray(pcd.points))


    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Open3D Visualization', width=800, height=600)
 

    # while not conn.poll():
    #     vis.poll_events()
    #     vis.update_renderer()
    
    zoom=0.3412
    front=[0.4257, -0.2125, -0.8795]
    lookat=[2.6172, 2.0475, 1.532]
    up=[-0.0694, -0.9768, 0.2024]

    # Apply the parameters

    o3d.visualization.draw_geometries([pcd])
    ctr = o3d.get_view_control()

    ctr.set_front(front)
    ctr.set_lookat(lookat)
    ctr.set_up(up)
    ctr.set_zoom(zoom)
    o3d.visualization.update_renderer()
            
    
        
        

    # vis.destroy_window()
    # conn.send("Done")
    # conn.close()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Open3D inside PySide2')

        layout = QVBoxLayout()
        self.showButton = QPushButton("Show Point Cloud", self)
        self.showButton.clicked.connect(self.showPointCloud)
        layout.addWidget(self.showButton)

        centralWidget = QWidget(self)
        centralWidget.setLayout(layout)
        self.setCentralWidget(centralWidget)

    def showPointCloud(self):
        self.parent_conn, self.child_conn = Pipe()
        self.process = Process(target=run_open3d_visualizer, args=(self.child_conn,))
        self.process.start()
        self.showButton.setEnabled(False)
        self.showButton.setText("Visualizing...")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
