import mmcv
import torch
import numpy as np
from PIL import Image
from numpy import random
from typing import Any, Dict
from mmcv.transforms import BaseTransform
from mmdet3d.datasets import GlobalRotScaleTrans
from mmdet3d.registry import TRANSFORMS


@TRANSFORMS.register_module()
class PadMultiViewImage(object):   #object
    """Pad the multi-view image.
    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",
    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, img):
        if self.size_divisor is not None:
            pad_h = int(np.ceil(img.shape[0] / self.size_divisor)) * self.size_divisor
            pad_w = int(np.ceil(img.shape[1] / self.size_divisor)) * self.size_divisor
        else:
            pad_h, pad_w = self.size

        pad_width = ((0, pad_h - img.shape[0]), (0, pad_w - img.shape[1]), (0, 0))
        img = np.pad(img, pad_width, constant_values=self.pad_val)
        return img

    def _pad_imgs(self, results):
        padded_img = [self._pad_img(img) for img in results['img']]
        
        results['ori_shape'] = [img.shape for img in results['img']]
        results['img'] = padded_img
        results['img_shape'] = [img.shape for img in padded_img]
        results['pad_shape'] = [img.shape for img in padded_img]
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        self._pad_imgs(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str


@TRANSFORMS.register_module()
class NormalizeMultiviewImage(object):
    """Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32).reshape(-1)
        self.std = 1 / np.array(std, dtype=np.float32).reshape(-1)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        normalized_imgs = []

        for img in results['img']:
            img = img.astype(np.float32)
            if self.to_rgb:
                img = img[..., ::-1]
            img = img - self.mean
            img = img * self.std
            normalized_imgs.append(img)

        results['img'] = normalized_imgs
        results['img_norm_cfg'] = dict(
            mean=self.mean,
            std=self.std,
            to_rgb=self.to_rgb
        )
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str


@TRANSFORMS.register_module()
class PhotoMetricDistortionMultiViewImage(object):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, results):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        imgs = results['img']
        new_imgs = []
        for img in imgs:
            ori_dtype = img.dtype
            img = img.astype(np.float32)

            # random brightness
            if random.randint(2):
                delta = random.uniform(-self.brightness_delta,
                                    self.brightness_delta)
                img += delta

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = random.randint(2)
            if mode == 1:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower,
                                        self.contrast_upper)
                    img *= alpha

            # convert color from BGR to HSV
            img = mmcv.bgr2hsv(img)

            # random saturation
            if random.randint(2):
                img[..., 1] *= random.uniform(self.saturation_lower,
                                            self.saturation_upper)

            # random hue
            if random.randint(2):
                img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            # convert color from HSV to BGR
            img = mmcv.hsv2bgr(img)

            # random contrast
            if mode == 0:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower,
                                        self.contrast_upper)
                    img *= alpha

            # randomly swap channels
            if random.randint(2):
                img = img[..., random.permutation(3)]

            new_imgs.append(img.astype(ori_dtype))

        results['img'] = new_imgs
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(\nbrightness_delta={self.brightness_delta},\n'
        repr_str += 'contrast_range='
        repr_str += f'{(self.contrast_lower, self.contrast_upper)},\n'
        repr_str += 'saturation_range='
        repr_str += f'{(self.saturation_lower, self.saturation_upper)},\n'
        repr_str += f'hue_delta={self.hue_delta})'
        return repr_str


@TRANSFORMS.register_module()
class RandomTransformImage(object):
    def __init__(self, ida_aug_conf=None, training=True):
        self.ida_aug_conf = ida_aug_conf
        self.training = training

    def __call__(self, results):
        resize, resize_dims, crop, flip, rotate = self.sample_augmentation()
        
        if len(results['lidar2img']) == len(results['img']):
            for i in range(len(results['img'])):
                img = Image.fromarray(np.uint8(results['img'][i]))
                
                # resize, resize_dims, crop, flip, rotate = self._sample_augmentation()
                img, ida_mat = self.img_transform(
                    img,
                    resize=resize,
                    resize_dims=resize_dims,
                    crop=crop,
                    flip=flip,
                    rotate=rotate,
                )
                results['img'][i] = np.array(img).astype(np.uint8)
                results['lidar2img'][i] = ida_mat @ results['lidar2img'][i]

        elif len(results['img']) == 6:
            for i in range(len(results['img'])):
                img = Image.fromarray(np.uint8(results['img'][i]))
                
                # resize, resize_dims, crop, flip, rotate = self._sample_augmentation()
                img, ida_mat = self.img_transform(
                    img,
                    resize=resize,
                    resize_dims=resize_dims,
                    crop=crop,
                    flip=flip,
                    rotate=rotate,
                )
                results['img'][i] = np.array(img).astype(np.uint8)

            for i in range(len(results['lidar2img'])):
                results['lidar2img'][i] = ida_mat @ results['lidar2img'][i]

        else:
            raise ValueError()

        results['ori_shape'] = [img.shape for img in results['img']]
        results['img_shape'] = [img.shape for img in results['img']]
        results['pad_shape'] = [img.shape for img in results['img']]

        return results

    def img_transform(self, img, resize, resize_dims, crop, flip, rotate):
        """
        https://github.com/Megvii-BaseDetection/BEVStereo/blob/master/dataset/nusc_mv_det_dataset.py#L48
        """
        def get_rot(h):
            return torch.Tensor([
                [np.cos(h), np.sin(h)],
                [-np.sin(h), np.cos(h)],
            ])

        ida_rot = torch.eye(2)
        ida_tran = torch.zeros(2)

        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        ida_rot *= resize
        ida_tran -= torch.Tensor(crop[:2])
        
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            ida_rot = A.matmul(ida_rot)
            ida_tran = A.matmul(ida_tran) + b
        
        A = get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b

        ida_rot = A.matmul(ida_rot)
        ida_tran = A.matmul(ida_tran) + b

        ida_mat = torch.eye(4)
        ida_mat[:2, :2] = ida_rot
        ida_mat[:2, 2] = ida_tran

        return img, ida_mat.numpy()

    def sample_augmentation(self):
        """
        https://github.com/Megvii-BaseDetection/BEVStereo/blob/master/dataset/nusc_mv_det_dataset.py#L247
        """
        H, W = self.ida_aug_conf['H'], self.ida_aug_conf['W']
        fH, fW = self.ida_aug_conf['final_dim']

        if self.training:
            resize = np.random.uniform(*self.ida_aug_conf['resize_lim'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.ida_aug_conf['bot_pct_lim'])) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.ida_aug_conf['rand_flip'] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.ida_aug_conf['rot_lim'])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.ida_aug_conf['bot_pct_lim'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0

        return resize, resize_dims, crop, flip, rotate


@TRANSFORMS.register_module()
class GlobalRotScaleTransImage(object):
    def __init__(self,
                 rot_range=[-0.3925, 0.3925],
                 scale_ratio_range=[0.95, 1.05],
                 translation_std=[0, 0, 0]):
        self.rot_range = rot_range
        self.scale_ratio_range = scale_ratio_range
        self.translation_std = translation_std

    def __call__(self, results):
        # random rotate
        rot_angle = np.random.uniform(*self.rot_range)
        self.rotate_z(results, rot_angle)
        results["gt_bboxes_3d"].rotate(np.array(rot_angle))

        # random scale
        scale_ratio = np.random.uniform(*self.scale_ratio_range)
        self.scale_xyz(results, scale_ratio)
        results["gt_bboxes_3d"].scale(scale_ratio)

        # TODO: support translation

        return results

    def rotate_z(self, results, rot_angle):
        rot_cos = torch.cos(torch.tensor(rot_angle))
        rot_sin = torch.sin(torch.tensor(rot_angle))

        rot_mat = torch.tensor([
            [rot_cos, -rot_sin, 0, 0],
            [rot_sin, rot_cos, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        rot_mat_inv = torch.inverse(rot_mat)

        for view in range(len(results['lidar2img'])):
            results['lidar2img'][view] = (torch.tensor(results['lidar2img'][view]).float() @ rot_mat_inv).numpy()

    def scale_xyz(self, results, scale_ratio):
        scale_mat = torch.tensor([
            [scale_ratio, 0, 0, 0],
            [0, scale_ratio, 0, 0],
            [0, 0, scale_ratio, 0],
            [0, 0, 0, 1],
        ])
        scale_mat_inv = torch.inverse(scale_mat)

        for view in range(len(results['lidar2img'])):
            results['lidar2img'][view] = (torch.tensor(results['lidar2img'][view]).float() @ scale_mat_inv).numpy()



@TRANSFORMS.register_module()
class ImageAug3D(BaseTransform):

    def __init__(self, final_dim, resize_lim, bot_pct_lim, rot_lim, rand_flip,
                 is_train):
        self.final_dim = final_dim
        self.resize_lim = resize_lim
        self.bot_pct_lim = bot_pct_lim
        self.rand_flip = rand_flip
        self.rot_lim = rot_lim
        self.is_train = is_train

    def sample_augmentation(self, results):
        H, W = results['ori_shape']
        fH, fW = self.final_dim
        if self.is_train:
            resize = np.random.uniform(*self.resize_lim)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int(
                (1 - np.random.uniform(*self.bot_pct_lim)) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.rand_flip and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.rot_lim)
        else:
            resize = np.mean(self.resize_lim)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.bot_pct_lim)) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def img_transform(self, img, rotation, translation, resize, resize_dims,
                      crop, flip, rotate):
        # adjust image
        img = Image.fromarray(img.astype('uint8'), mode='RGB')
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        rotation *= resize
        translation -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            rotation = A.matmul(rotation)
            translation = A.matmul(translation) + b
        theta = rotate / 180 * np.pi
        A = torch.Tensor([
            [np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)],
        ])
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        rotation = A.matmul(rotation)
        translation = A.matmul(translation) + b

        return img, rotation, translation

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        imgs = data['img']
        new_imgs = []
        transforms = []
        for img in imgs:
            resize, resize_dims, crop, flip, rotate = self.sample_augmentation(
                data)
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)
            new_img, rotation, translation = self.img_transform(
                img,
                post_rot,
                post_tran,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )
            transform = torch.eye(4)
            transform[:2, :2] = rotation
            transform[:2, 3] = translation
            new_imgs.append(np.array(new_img).astype(np.float32))
            transforms.append(transform.numpy())
        data['img'] = new_imgs
        # update the calibration matrices
        data['img_aug_matrix'] = transforms
        return data


@TRANSFORMS.register_module()
class BEVFusionRandomFlip3D:
    """Compared with `RandomFlip3D`, this class directly records the lidar
    augmentation matrix in the `data`."""

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        flip_horizontal = np.random.choice([0, 1])
        flip_vertical = np.random.choice([0, 1])

        rotation = np.eye(3)
        if flip_horizontal:
            rotation = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]) @ rotation
            if 'points' in data:
                data['points'].flip('horizontal')
            if 'gt_bboxes_3d' in data:
                data['gt_bboxes_3d'].flip('horizontal')
            if 'gt_masks_bev' in data:
                data['gt_masks_bev'] = data['gt_masks_bev'][:, :, ::-1].copy()

        if flip_vertical:
            rotation = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]) @ rotation
            if 'points' in data:
                data['points'].flip('vertical')
            if 'gt_bboxes_3d' in data:
                data['gt_bboxes_3d'].flip('vertical')
            if 'gt_masks_bev' in data:
                data['gt_masks_bev'] = data['gt_masks_bev'][:, ::-1, :].copy()

        if 'lidar_aug_matrix' not in data:
            data['lidar_aug_matrix'] = np.eye(4)
        data['lidar_aug_matrix'][:3, :] = rotation @ data[
            'lidar_aug_matrix'][:3, :]
        return data


@TRANSFORMS.register_module()
class BEVFusionGlobalRotScaleTrans(GlobalRotScaleTrans):
    """Compared with `GlobalRotScaleTrans`, the augmentation order in this
    class is rotation, translation and scaling (RTS)."""

    def transform(self, input_dict: dict) -> dict:
        """Private function to rotate, scale and translate bounding boxes and
        points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'points', 'pcd_rotation',
            'pcd_scale_factor', 'pcd_trans' and `gt_bboxes_3d` are updated
            in the result dict.
        """
        if 'transformation_3d_flow' not in input_dict:
            input_dict['transformation_3d_flow'] = []

        self._rot_bbox_points(input_dict)

        if 'pcd_scale_factor' not in input_dict:
            self._random_scale(input_dict)
        self._trans_bbox_points(input_dict)
        self._scale_bbox_points(input_dict)

        input_dict['transformation_3d_flow'].extend(['R', 'T', 'S'])

        lidar_augs = np.eye(4)
        lidar_augs[:3, :3] = input_dict['pcd_rotation'].T * input_dict[
            'pcd_scale_factor']
        lidar_augs[:3, 3] = input_dict['pcd_trans'] * \
            input_dict['pcd_scale_factor']

        if 'lidar_aug_matrix' not in input_dict:
            input_dict['lidar_aug_matrix'] = np.eye(4)
        input_dict[
            'lidar_aug_matrix'] = lidar_augs @ input_dict['lidar_aug_matrix']

        return input_dict


@TRANSFORMS.register_module()
class GridMask(BaseTransform):

    def __init__(
        self,
        use_h,
        use_w,
        max_epoch,
        rotate=1,
        offset=False,
        ratio=0.5,
        mode=0,
        prob=1.0,
        fixed_prob=False,
    ):
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.prob = prob
        self.epoch = None
        self.max_epoch = max_epoch
        self.fixed_prob = fixed_prob

    def set_epoch(self, epoch):
        self.epoch = epoch
        if not self.fixed_prob:
            self.set_prob(self.epoch, self.max_epoch)

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * self.epoch / self.max_epoch

    def transform(self, results):
        if np.random.rand() > self.prob:
            return results
        imgs = results['img']
        h = imgs[0].shape[0]
        w = imgs[0].shape[1]
        self.d1 = 2
        self.d2 = min(h, w)
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        d = np.random.randint(self.d1, self.d2)
        if self.ratio == 1:
            self.length = np.random.randint(1, d)
        else:
            self.length = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        mask = np.ones((hh, ww), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        if self.use_h:
            for i in range(hh // d):
                s = d * i + st_h
                t = min(s + self.length, hh)
                mask[s:t, :] *= 0
        if self.use_w:
            for i in range(ww // d):
                s = d * i + st_w
                t = min(s + self.length, ww)
                mask[:, s:t] *= 0

        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[(hh - h) // 2:(hh - h) // 2 + h,
                    (ww - w) // 2:(ww - w) // 2 + w]

        mask = mask.astype(np.float32)
        mask = mask[:, :, None]
        if self.mode == 1:
            mask = 1 - mask

        # mask = mask.expand_as(imgs[0])
        if self.offset:
            offset = torch.from_numpy(2 * (np.random.rand(h, w) - 0.5)).float()
            offset = (1 - mask) * offset
            imgs = [x * mask + offset for x in imgs]
        else:
            imgs = [x * mask for x in imgs]

        results.update(img=imgs)
        return results