_base_ = [
    '../_base_/models/pointpillars_hv_secfpn_waymo.py',
    '../_base_/datasets/waymoD5-3d-3class.py',
    '../_base_/schedules/schedule-2x.py',
    '../_base_/default_runtime.py',
]




# data settings
train_dataloader = dict(batch_size=4)
# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (16 GPUs) x (2 samples per GPU).cl
auto_scale_lr = dict(enable=False, base_batch_size=4)



checkpoint_config = dict(interval=1)

# training schedule for 1x
train_cfg = dict(_delete_=True, type='EpochBasedTrainLoop', max_epochs=41, val_interval=40)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

log_level = 'INFO'
load_from = None
resume = True


work_dir = './work_dirs/pointpillars_hv_secfpn_sbn-all_16xb2-2x_waymoD5-3d-3class'
# work_dir = './work_dirs/logtesting'
resume=True
load_from = None
resume_from = './work_dirs/pointpillars_hv_secfpn_sbn-all_16xb2-2x_waymoD5-3d-3class'


auto_scale_lr = dict(enable=False, base_batch_size=4)

workflow = [('train', 1),('val', 1)] 