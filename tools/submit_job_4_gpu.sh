#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 32       # 32 cores (8 cores per GPU)
#$ -l h_rt=1:0:0  # 240 hours runtime
#$ -l gpu=4         # request 4 GPUs
#$ -l exclusive     # request exclusive acces


###$ -l gpu_type=ampere



# volta|ampere

module load python/3.8.5
source /data/home/acw482/workspace/mmdet/bin/activate
CONFIG_FILE="/data/home/acw482/workspace/mmdetection3d/configs/3dconvit/convit3D_pointNet_transformer_waymo.py"
GPU_NUM=4
/data/home/acw482/workspace/mmdetection3d/tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM}