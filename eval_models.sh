# # Running PVRCNN
# python tools/test.py /workspace/mmdetection3d/configs/pv_rcnn/pv_rcnn_8xb2-80e_kitti-3d-3class.py \
#     /workspace/models_to_test/PV_RCNN pv_rcnn_8xb2-80e_kitti-3d-3class_20221117_234428-b384d22f.pth \
#     --show --show-dir /workspace/data/pvrcnn/show_results


# # Running pointpiller
# python tools/test.py /workspace/mmdetection3d/configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py \
#     /workspace/models_to_test/pointpiller hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth \
#     --show --show-dir /workspace/pointpiller/show_results

# # Running PointRCNN
# python tools/test.py /workspace/mmdetection3d/configs/point_rcnn/point-rcnn_8xb2_kitti-3d-3class.py \
#     /workspace/models_to_test/PointRCNN point_rcnn_2x8_kitti-3d-3classes_20211208_151344.pth \
#     --show --show-dir /workspace/pointrcnn/show_results


#  ===== Running evalution of ConViT3D
# python tools/test.py /workspace/mmdetection3d/configs/3dconvit/convit3D_pointnet_transformer_ssdhead.py \
#     /workspace/mmdetection3d/work_dirs/convit3D_PointNet_transformer_ssdhead/epoch_80.pth\
#     --task lidar_det --show --show-dir /workspace/conVit3D/show_results


CONFIG_FILE="/workspace/mmdetection3d/configs/3dconvit/convit3D_pointnet_transformer_ssdhead.py"
CKPT_PATH="/workspace/mmdetection3d/work_dirs/convit3D_PointNet_transformer_ssdhead/epoch_80.pth"
SHOW_DIR="/workspace/data/kitti_detection/model_output_results/ConVitResults"
# python tools/test.py /workspace/mmdetection3d/configs/3dconvit/convit3D_pointnet_transformer_ssdhead.py \
#     /workspace/mmdetection3d/work_dirs/convit3D_PointNet_transformer_ssdhead/epoch_80.pth\
#     --task lidar_det --show-dir /workspace/conVit3D/show_results

python tools/test.py ${CONFIG_FILE} ${CKPT_PATH} --task lidar_det --show-dir ${SHOW_DIR}





#----------------------------- benchmark evalutation for FPS
# ConViT3D FPS ~ 15.8
# python tools/analysis_tools/benchmark.py /workspace/mmdetection3d/configs/3dconvit/convit3D_pointnet_transformer_ssdhead.py \
#            /workspace/mmdetection3d/work_dirs/convit3D_PointNet_transformer_ssdhead/epoch_80.pth\


# PV-RCNN FPS ~ 12.6
# python tools/analysis_tools/benchmark.py /workspace/mmdetection3d/configs/pv_rcnn/pv_rcnn_8xb2-80e_kitti-3d-3class.py \
#            /workspace/data/kitti_detection/models_to_test/PV_RCNN/pv_rcnn_8xb2-80e_kitti-3d-3class_20221117_234428-b384d22f.pth

# Pointpiller FPS ~ 12.6
# python tools/analysis_tools/benchmark.py /workspace/mmdetection3d/configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py \
#     /workspace/data/kitti_detection/models_to_test/pointpiller/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth 