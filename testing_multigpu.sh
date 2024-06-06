CONFIG_FILE="/import/digitreasure/ammar_workspace/mmdetection3d/configs/pointpillars/pointpillars_hv_secfpn_sbn-all_16xb2-2x_waymo-3d-car.py"
# CHECKPOINT_FILE="/import/digitreasure/openmm_processed_dataset/waymo/hv_pointpillars_secfpn_sbn_2x16_2x_waymoD5-3d-car_20200901_204315-302fc3e7.pth"
# SHOW_DIR="/workspace/data/kitti_detection/model_output_results/pointResults"
# CUDA_VISIBLE_DEVICES=0,1 
GPU_NUM=2

CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM}