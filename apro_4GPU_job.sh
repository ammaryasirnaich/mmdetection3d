#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 32       # 32 cores (8 cores per GPU)
#$ -l h_rt=1:0:0    # 1 hour runtime (required to run on the short queue
#$ -l h_vmem=4G    # 11 * 16 = 176G total RAM
#$ -l gpu=4         # request 4 GPUs
#$ -l exclusive     # request exclusive access
#$ -l gpu_type=ampere


module load python/3.8.5
source /data/home/acw482/workspace/mmdet/bin/activate
CONFIG_FILE="/data/home/acw482/workspace/mmdetection3d/configs/3dconvit/convit3D_pointNet_transformer_waymo.py"
GPU_NUM=4
/data/home/acw482/workspace/mmdetection3d/tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM}