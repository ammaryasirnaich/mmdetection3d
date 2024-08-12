
# python tools/train.py /workspace/mmdetection3d/configs/3dconvit/convit3D_pointNet_transformer_kitti.py
# python tools/train.py /workspace/mmdetection3d/configs/3dconvit/convit3D_no_voxel_no_transformer_ssdhead.py
# python tools/train.py /workspace/mmdetection3d/configs/3dconvit/convit3D_pointnet_transformer_ssdhead.py

# python tools/train.py /workspace/mmdetection3d/configs/3dconvit/convit3D_pointnet_transformer_ssdhead.py

# python tools/train.py /workspace/mmdetection3d/configs/3dconvit/convit3D_smallPointNet_transformer_ssdhead.
# python tools/train.py /workspace/mmdetection3d/configs/3dconvit/convit3D_pointNet_transformer_ssdhead_configInherence.py

# python tools/train.py /workspace/mmdetection3d/configs/3dconvit/conViT3D_Pointrpn_head.py


# python tools/train.py /workspace/mmdetection3d/configs/3dconvit/conViT3D_PointRCNN_head.py
# python tools/train.py /workspace/mmdetection3d/configs/3dssd/3dssd_4xb4_kitti-3d-all.py
# python tools/train.py /workspace/mmdetection3d/configs/3dssd/3dssd_4xb4_kitti-3d-all.py
# python tools/train.py /workspace/mmdetection3d/configs/inet/intensityNetKitti_all_objects.py

# CUDA_VISIBLE_DEVICES=1 python  tools/train.py "/import/digitreasure/ammar_workspace/mmdetection3d/configs/pointpillars/pointpiller_nusense_custome.py"

# python tools/train.py /workspace/mmdetection3d/configs/inet/inet_kitti_3d_all_update.py
# python tools/train.py /workspace/mmdetection3d/configs/3dconvit/convit3D_pointNet_transformer_ssdhead_configInherence.py
# pytho  tools/train.py /import/digitreasure/ammar_workspace/mmdetection3d/configs/3dssd/3dssd_4xb4_kitti-3d-car.py

# python tools/train.py /workspace/mmdetection3d/configs/3dconvit/convit3D_pointNet_transformer_ssdhead_configInherence.py

# python tools/train.py /workspace/mmdetection3d/configs/3dconvit/convit3D_pointNet_transformer_waymo.py
# python tools/train.py /workspace/mmdetection3d/configs/pointpillars/pointpillars_hv_secfpn_sbn-all_16xb2-2x_waymoD5-3d-3class.py

# CONFIG_FILE="/import/digitreasure/ammar_workspace/mmdetection3d/configs/3dconvit/currentconvit.py"
# CONFIG_FILE="/import/digitreasure/ammar_workspace/mmdetection3d/configs/3dconvit/convit3D_pointNet_transformer_nusence.py"
CONFIG_FILE="/import/digitreasure/ammar_workspace/mmdetection3d/configs/3dconvit/convit3D_pointNet_transformer_kitti.py"
GPU_NUM=2
CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} 

# CUDA_VISIBLE_DEVICES=0 PORT=29501 ./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} 