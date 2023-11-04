# wget -c  https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/test.txt --no-check-certificate --content-disposition -O /workspace/data/kitti_detection/kitti/ImageSets/test.txt
# wget -c  https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/train.txt --no-check-certificate --content-disposition -O /workspace/data/kitti_detection/kitti/ImageSets/train.txt
# wget -c  https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/val.txt --no-check-certificate --content-disposition -O /workspace/data/kitti_detection/kitti/ImageSets/val.txt
# wget -c  https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/trainval.txt --no-check-certificate --content-disposition -O /workspace/data/kitti_detection/kitti/ImageSets/trainval.txt

# python tools/create_data.py kitti --root-path /workspace/data/kitti_detection/kitti --out-dir /workspace/data/kitti_detection/kitti --extra-tag kitti




python tools/create_data.py waymo --root-path /workspace/data/waymo --out-dir /workspace/data/waymo --workers 8 --extra-tag waymo




