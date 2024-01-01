# wget -c  https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/test.txt --no-check-certificate --content-disposition -O /workspace/data/kitti_detection/kitti/ImageSets/test.txt
# wget -c  https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/train.txt --no-check-certificate --content-disposition -O /workspace/data/kitti_detection/kitti/ImageSets/train.txt
# wget -c  https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/val.txt --no-check-certificate --content-disposition -O /workspace/data/kitti_detection/kitti/ImageSets/val.txt
# wget -c  https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/trainval.txt --no-check-certificate --content-disposition -O /workspace/data/kitti_detection/kitti/ImageSets/trainval.txt

# python tools/create_data.py kitti --root-path /workspace/data/kitti_detection/kitti --out-dir /workspace/data/kitti_detection/kitti --extra-tag kitti




# python tools/create_data.py waymo --root-path /import/digitreasure/openmm_processed_dataset/waymo --out-dir /import/digitreasure/openmm_processed_dataset/waymo --workers 8 --extra-tag waymo --only-gt-database

# export OMP_NUM_THREADS=1
# export USE_SIMPLE_THREADED_LEVEL3= 1
# python tools/create_data.py waymo --root-path /import/digitreasure/openmm_processed_dataset/waymov12 --out-dir /import/digitreasure/openmm_processed_dataset/waymov12 --workers 40 --extra-tag waymo


python tools/create_data.py waymo --root-path /import/digitreasure/openmm_processed_dataset/waymov12/ --out-dir /import/digitreasure/openmm_processed_dataset/waymov12/ --workers 2 --extra-tag waymo --only-gt-database