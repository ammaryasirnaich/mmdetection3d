CONFIG_FILE="/import/digitreasure/ammar_workspace/open_mmlab/mmdetection3d/projects/SDFusion3d/configs/sdf_lidar_cam_nus-3d.py"
# CONFIG_FILE="/import/digitreasure/ammar_workspace/mmdetection3d/configs/pointpillars/pointpillars_hv_secfpn_sbn-all_16xb2-2x_waymoD5-3d-3class.py"
GPU_NUM=1

# CUDA_VISIBLE_DEVICES=1 ../../tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM}
CUDA_VISIBLE_DEVICES=1 python ../../tools/train.py ${CONFIG_FILE}