conda create -n dev_cu12 python=3.8.19
conda activate dev_cu12
conda install "conda-forge/linux-64::gcc 11.4.0 h602e360_10"
conda install "conda-forge/linux-64::gxx 11.4.0 h602e360_10"

conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia

conda install nvidia/label/cuda-11.8.0::cuda-toolkit
pip install -U openmim
mim install mmengine
mim install 'mmcv==2.2.0'
mim install 'mmdet==3.3.0'






torch                     2.3.1
torchaudio                2.3.1
torchvision               0.18.1
