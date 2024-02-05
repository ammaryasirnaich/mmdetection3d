#source /import/digitreasure/ammar_workspace/mmdet/bin/activate
source /homes/ayn30/mmdet/bin/activate
module load python/3.8.2
cd /homes/ayn30/
python -m venv mmdet
source mmdet/bin/activate
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python -m pip install cupy-cuda11x
python -m pip install spconv-cu118
python -m pip install open3d==0.13.0
python -m pip install timm
python -m pip install webcolors
python -m pip install urllib3==1.26.6
python -m pip install protobuf==3.20.0
python -m pip install waymo-open-dataset-tf-2-6-0==1.4.7
python -m pip install -U openmim
mim install mmengine
mim install 'mmcv==2.0.1'
mim install 'mmdet==3.2.0'



mmcv                        2.1.0
mmdet                       3.2.0
mmdet3d                     1.3.0   



# on server side for waymo library for validation
scl enable devtoolset-7 bash
module load bazel
