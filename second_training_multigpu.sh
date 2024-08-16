# CONFIG_FILE="/import/digitreasure/ammar_workspace/mmdetection3d/configs/3dconvit/convit3D_pointNet_transformer_nusence_pretrained.py"
# CONFIG_FILE="/import/digitreasure/ammar_workspace/mmdetection3d/configs/3dconvit/convit3D_pointNet_transformer_nusence.py"
CONFIG_FILE="/import/digitreasure/ammar_workspace/mmdetection3d/configs/3dconvit/convit3D_pointNet_transformer_kitti.py"
GPU_NUM=2
CUDA_VISIBLE_DEVICES=0,1 PORT=29501 ./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} 




