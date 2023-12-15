# python tools/test.py /workspace/data/kitti_detection/model_output_results/convit3d_tansformerHead/convit3D_pointnet_ssdhead.py /workspace/data/kitti_detection/model_output_results/convit3d_tansformerHead/epoch_80.pth --show --show-dir /workspace/mmdetection3d/working_dir/d3ssd

# python demo/pcd_demo.py demo/data/kitti/000008.bin configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py model_zoo/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth --show --out-dir ./working_dir/pointpilller

# python demo/pcd_demo.py /workspace/data/kitti_detection/kitti/testing/velodyne_reduced/000003.bin configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py /workspace/data/kitti_detection/models_to_test/pointpiller/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth --show --out-dir ./working_dir/pointpilller
#
# python demo/pcd_demo.py /workspace/data/kitti_detection/kitti/testing/velodyne_reduced/000003.bin configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py /workspace/data/kitti_detection/models_to_test/pointpiller/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth --out-dir ./


python demo/pcd_demo.py /workspace/data/kitti_detection/kitti/testing/velodyne_reduced/000009.bin /workspace/data/kitti_detection/model_output_results/intensityAware_35pnt/intensityNetKitti_all_objects.py /workspace/data/kitti_detection/model_output_results/intensityAware_35pnt/epoch_80.pth --show --out-dir ./working_dir/IVEF
