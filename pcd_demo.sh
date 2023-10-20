
# CONFIG_FILE="/workspace/mmdetection3d/work_dirs/convit3d_PointNet_Transformer_rpnhead/conViT3D_Pointrpn_head.py"
# CHECKPOINT_FILE="/workspace/mmdetection3d/work_dirs/convit3d_PointNet_Transformer_rpnhead/epoch_41.pth"

PCD_FILE="/workspace/mmdetection3d/demo/data/kitti/000008.bin"

# CONFIG_FILE="/workspace/mmdetection3d/configs/3dconvit/boundingBox_on_convit3D.py"
# CHECKPOINT_FILE="/workspace/mmdetection3d/work_dirs/convit3D_PointNet_transformer_ssdhead__14_August/epoch_202.pth"


CONFIG_FILE="/workspace/mmdetection3d/configs/inet/intensityNetKitti_all_objects.py"
CHECKPOINT_FILE="/workspace/data/kitti_detection/IVEF/epoch_80.pth"




# CONFIG_FILE="/workspace/mmdetection3d/configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py"
# CHECKPOINT_FILE="/workspace/data/kitti_detection/models_to_test/pointpiller/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth"



# CONFIG_FILE="/workspace/mmdetection3d/configs/second/second_hv_secfpn_8xb6-amp-80e_kitti-3d-car.py"
# CHECKPOINT_FILE="/workspace/data/kitti_detection/models_to_test/second/hv_second_secfpn_fp16_6x8_80e_kitti-3d-3class_20200925_110059-05f67bdf.pth"


# pwd 
# python demo/pcd_demo.py ${PCD_FILE} ${CONFIG_FILE} ${CHECKPOINT_FILE} 
python tests/intensity_mode_visualization.py ${PCD_FILE} ${CONFIG_FILE} ${CHECKPOINT_FILE} 


