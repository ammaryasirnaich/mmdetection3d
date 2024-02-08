_base_ = [
    '../_base_/models/pointpillars_hv_secfpn_waymo.py',
    '../_base_/datasets/waymoD5-3d-3class.py',
    '../_base_/schedules/schedule-2x.py',
    '../_base_/default_runtime.py',
]

# data settings
train_dataloader = dict(dataset=dict(dataset=dict(load_interval=1)))
# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (16 GPUs) x (2 samples per GPU).
# auto_scale_lr = dict(enable=False, base_batch_size=32)


auto_scale_lr = dict(enable=False, base_batch_size=4)



# training schedule for 1x
train_cfg = dict(_delete_=True, type='EpochBasedTrainLoop', max_epochs=40, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')


custom_hooks = [dict(type='EpochLossValuesLogging')]
checkpoint_config = dict(interval=3)
workflow = [('train', 1),('val', 10)]  
