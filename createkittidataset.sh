# wget -c  https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/test.txt --no-check-certificate --content-disposition -O /workspace/data/kitti_detection/kitti/ImageSets/test.txt
# wget -c  https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/train.txt --no-check-certificate --content-disposition -O /workspace/data/kitti_detection/kitti/ImageSets/train.txt
# wget -c  https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/val.txt --no-check-certificate --content-disposition -O /workspace/data/kitti_detection/kitti/ImageSets/val.txt
# wget -c  https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/trainval.txt --no-check-certificate --content-disposition -O /workspace/data/kitti_detection/kitti/ImageSets/trainval.txt

# Run from project root so "tools" is importable; add project root to PYTHONPATH
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
python tools/create_data.py kitti --root-path /home/naich/dataset/kitti_data --out-dir /home/naich/dataset/kitti_data --extra-tag kitti




# python tools/create_data.py waymo --root-path /import/digitreasure/openmm_processed_dataset/waymo --out-dir /import/digitreasure/openmm_processed_dataset/waymo --workers 8 --extra-tag waymo --only-gt-database

# export OMP_NUM_THREADS=1
# export USE_SIMPLE_THREADED_LEVEL3= 1
# python tools/create_data.py waymo --root-path /import/digitreasure/openmm_processed_dataset/waymov12 --out-dir /import/digitreasure/openmm_processed_dataset/waymov12 --workers 40 --extra-tag waymo


# python tools/create_data.py waymo --root-path /import/digitreasure/openmm_processed_dataset/waymov12/ --out-dir /import/digitreasure/openmm_processed_dataset/waymov12/ --workers 2 --extra-tag waymo --only-gt-database