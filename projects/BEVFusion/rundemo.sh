CHECKPOINT_FILE="./demo/bevfusion-det.pth"
CONFIG="./configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py"
python ./demo/multi_modality_demo.py demo/data/nuscenes/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin demo/data/nuscenes/ demo/data/nuscenes/n015-2018-07-24-11-22-45+0800.pkl ${CONFIG} ${CHECKPOINT_FILE} --cam-type all --score-thr 0.2 --show
