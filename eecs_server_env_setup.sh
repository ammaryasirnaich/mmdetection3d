
module load cuda/11.7
module load python/3.8.2
python -m venv mmdet
source mmdet/bin/activate
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install cupy-cuda11x
pip install spconv-cu117
pip install open3d==0.13.0
pip install waymo-open-dataset-tf-2-6-0==1.4.7
pip install -U openmim
pip install urllib3==1.26.6
mim install mmengine
mim install 'mmcv>=2.0.1'
mim install 'mmdet>=3.0.0'
pip install timm