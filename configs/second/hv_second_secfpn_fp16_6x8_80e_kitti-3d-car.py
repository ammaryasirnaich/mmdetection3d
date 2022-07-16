_base_ = './hv_second_secfpn_6x8_80e_kitti-3d-car.py'
work_dir = "/workspace/mmdetection3d/working_dir/second_car"
# fp16 settings
# fp16 = dict(loss_scale=512.)
fp16 = dict(loss_scale='dynamic')
