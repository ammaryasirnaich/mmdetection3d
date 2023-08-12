
CONFIG_FILE="/workspace/mmdetection3d/work_dirs/convit3d_PointNet_Transformer_rpnhead/conViT3D_Pointrpn_head.py"
CHECKPOINT_FILE="/workspace/mmdetection3d/work_dirs/convit3d_PointNet_Transformer_rpnhead/epoch_41.pth"
PCD_FILE="/workspace/mmdetection3d/demo/data/kitti/000008.bin"


# pwd 
python demo/pcd_demo.py ${PCD_FILE} ${CONFIG_FILE} ${CHECKPOINT_FILE} 


