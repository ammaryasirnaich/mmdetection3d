#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 16       # 16 cores (8 cores per GPU)
#$ -l h_rt=20:0:0    # 1 hour runtime (required to run on the short queue)
#$ -l h_vmem=8G    # 11 * 16 = 128G total RAM



module load python/3.8.5
module load gcc/12.1.0
source /data/home/acw482/workspace/mmdet/bin/activate
python /data/home/acw482/workspace/mmdetection3d/tools/create_data.py waymo --root-path /data/scratch/acw482/WaymoDataset --out-dir /data/scratch/acw482/WaymoDataset --workers 2 --extra-tag waymo --only-gt-database --version v1.4