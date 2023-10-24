_base_ = [
    '../_base_/models/convit3D_waymo.py',
    '../_base_/datasets/waymoD5-3d-3class.py',
]


# data settings
# train_dataloader = dict( batch_size=2)
# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (16 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)

dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/convit3D_waymo'
load_from = None
resume_from = './work_dirs/convit3D_waymo'
workflow = [('train', 1),('val', 1)]  
# workflow = [('val', 1)]  

