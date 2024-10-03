CONFIG_FILE="/import/digitreasure/ammar_workspace/open_mmlab/mmdetection3d/projects/SDFusion3d/configs/sdf_lidar_cam_nus_swim_batch2.py"
# CONFIG_FILE="/import/digitreasure/ammar_workspace/mmdetection3d/configs/pointpillars/pointpillars_hv_secfpn_sbn-all_16xb2-2x_waymoD5-3d-3class.py"
GPU_NUM=2

# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0,1 PORT=29501 ../../tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM}

TORCH_NCCL_BLOCKING_WAIT=1 NCCL_TIMEOUT=7200 CUDA_VISIBLE_DEVICES=0,1 PORT=29502 ../../tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM}

# TORCH_NCCL_BLOCKING_WAIT=1 CUDA_VISIBLE_DEVICES=0,1 NCCL_TIMEOUT_MS=300000 ../../tools/dist_train_update.sh ${CONFIG_FILE} ${GPU_NUM}

# NCCL_TIMEOUT=1000  TORCH_NCCL_BLOCKING_WAIT=1 CUDA_VISIBLE_DEVICES=0,1 python ../../tools/train.py ${CONFIG_FILE} 

