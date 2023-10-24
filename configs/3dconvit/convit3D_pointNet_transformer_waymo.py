_base_ = [
    '../_base_/models/convit3D_waymo.py',
    '../_base_/datasets/waymoD5-3d-3class.py',
]

data_root = 'workspace/data/waymo/waymo_mini/kitti_format'

# data settings
train_dataloader = dict( batch_size=1,dataset=dict(dataset=dict(load_interval=1)))
# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (16 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=32)
