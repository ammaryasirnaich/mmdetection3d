# CONFIG_FILE="/import/digitreasure/ammar_workspace/mmdetection3d/configs/3dconvit/convit3D_pointNet_transformer_waymo.py"
CONFIG_FILE="/import/digitreasure/ammar_workspace/mmdetection3d/configs/pointpillars/pointpillars_hv_secfpn_sbn-all_16xb2-2x_waymo-3d-car.py"
GPU_NUM=2
CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM}