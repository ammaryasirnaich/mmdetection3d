# Running PVRCNN
python tools/test.py /workspace/mmdetection3d/configs/pv_rcnn/pv_rcnn_8xb2-80e_kitti-3d-3class.py \
    /workspace/models_to_test/PV_RCNN pv_rcnn_8xb2-80e_kitti-3d-3class_20221117_234428-b384d22f.pth \
    --show --show-dir /workspace/data/pvrcnn/show_results


# Running pointpiller
python tools/test.py /workspace/mmdetection3d/configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py \
    /workspace/models_to_test/pointpiller hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth \
    --show --show-dir /workspace/pointpiller/show_results

# Running PointRCNN
python tools/test.py /workspace/mmdetection3d/configs/point_rcnn/point-rcnn_8xb2_kitti-3d-3class.py \
    /workspace/models_to_test/PointRCNN point_rcnn_2x8_kitti-3d-3classes_20211208_151344.pth \
    --show --show-dir /workspace/pointrcnn/show_results