# Copyright (c) OpenMMLab. All rights reserved.
r"""Adapted from `Waymo to KITTI converter
    <https://github.com/caizhongang/waymo_kitti_converter>`_.
"""

try:
    from waymo_open_dataset import label_pb2
    from waymo_open_dataset.protos import metrics_pb2
    from waymo_open_dataset.protos.metrics_pb2 import Objects
except ImportError:
    Objects = None
    raise ImportError(
        'Please run "pip install waymo-open-dataset-tf-2-1-0==1.2.0" '
        'to install the official devkit first.')

from typing import List

import mmengine
from mmengine import print_log


class Prediction2Waymo(object):
    """Predictions to Waymo converter. The format of prediction results could
    be original format or kitti-format.

    This class serves as the converter to change predictions from KITTI to
    Waymo format.

    Args:
        results (list[dict]): Prediction results.
        waymo_results_save_dir (str): Directory to save converted predictions
            in waymo format (.bin files).
        waymo_results_final_path (str): Path to save combined
            predictions in waymo format (.bin file), like 'a/b/c.bin'.
        num_workers (str): Number of parallel processes. Defaults to 4.
    """

    def __init__(self,
                 results: List[dict],
                 waymo_results_final_path: str,
                 classes: dict,
                 num_workers: int = 4):
        self.results = results
        self.waymo_results_final_path = waymo_results_final_path
        self.classes = classes
        self.num_workers = num_workers

        self.k2w_cls_map = {
            'Car': label_pb2.Label.TYPE_VEHICLE,
            'Pedestrian': label_pb2.Label.TYPE_PEDESTRIAN,
            'Sign': label_pb2.Label.TYPE_SIGN,
            'Cyclist': label_pb2.Label.TYPE_CYCLIST,
        }

        if self.from_kitti_format:
            self.T_ref_to_front_cam = np.array([[0.0, 0.0, 1.0, 0.0],
                                                [-1.0, 0.0, 0.0, 0.0],
                                                [0.0, -1.0, 0.0, 0.0],
                                                [0.0, 0.0, 0.0, 1.0]])
            # ``sample_idx`` of the sample in kitti-format is an array
            for idx, result in enumerate(results):
                if len(result['sample_idx']) > 0:
                    self.name2idx[str(result['sample_idx'][0])] = idx
        else:
            # ``sample_idx`` of the sample in the original prediction
            # is an int value.
            for idx, result in enumerate(results):
                self.name2idx[str(result['sample_idx'])] = idx

        if not self.fast_eval:
            # need to read original '.tfrecord' file
            self.get_file_names()
            # turn on eager execution for older tensorflow versions
            if int(tf.__version__.split('.')[0]) < 2:
                tf.enable_eager_execution()

        self.create_folder()

    def get_file_names(self):
        """Get file names of waymo raw data."""
        if self.backend_args != None and 'path_mapping' in self.backend_args:
            for path in self.backend_args['path_mapping'].keys():
                if path in self.waymo_tfrecords_dir:
                    self.waymo_tfrecords_dir = \
                        self.waymo_tfrecords_dir.replace(
                            path, self.backend_args['path_mapping'][path])
            from petrel_client.client import Client
            client = Client()
            contents = client.list(self.waymo_tfrecords_dir)
            self.waymo_tfrecord_pathnames = list()
            for content in sorted(list(contents)):
                if content.endswith('tfrecord'):
                    self.waymo_tfrecord_pathnames.append(
                        join(self.waymo_tfrecords_dir, content))
        else:
            self.waymo_tfrecord_pathnames = sorted(
                glob(join(self.waymo_tfrecords_dir, '*.tfrecord')))
        print(len(self.waymo_tfrecord_pathnames), 'tfrecords found.')

    def create_folder(self):
        """Create folder for data conversion."""
        mmengine.mkdir_or_exist(self.waymo_results_save_dir)

    def parse_objects(self, kitti_result, T_k2w, context_name,
                      frame_timestamp_micros):
        """Parse one prediction with several instances in kitti format and
        convert them to `Object` proto.

        Args:
            kitti_result (dict): Predictions in kitti format.

                - name (np.ndarray): Class labels of predictions.
                - dimensions (np.ndarray): Height, width, length of boxes.
                - location (np.ndarray): Bottom center of boxes (x, y, z).
                - rotation_y (np.ndarray): Orientation of boxes.
                - score (np.ndarray): Scores of predictions.
            T_k2w (np.ndarray): Transformation matrix from kitti to waymo.
            context_name (str): Context name of the frame.
            frame_timestamp_micros (int): Frame timestamp.

        Returns:
            :obj:`Object`: Predictions in waymo dataset Object proto.
        """

        def parse_one_object(instance_idx):
            """Parse one instance in kitti format and convert them to `Object`
            proto.

            Args:
                instance_idx (int): Index of the instance to be converted.

            Returns:
                :obj:`Object`: Predicted instance in waymo dataset
                    Object proto.
            """
            cls = kitti_result['name'][instance_idx]
            length = round(kitti_result['dimensions'][instance_idx, 0], 4)
            height = round(kitti_result['dimensions'][instance_idx, 1], 4)
            width = round(kitti_result['dimensions'][instance_idx, 2], 4)
            x = round(kitti_result['location'][instance_idx, 0], 4)
            y = round(kitti_result['location'][instance_idx, 1], 4)
            z = round(kitti_result['location'][instance_idx, 2], 4)
            rotation_y = round(kitti_result['rotation_y'][instance_idx], 4)
            score = round(kitti_result['score'][instance_idx], 4)

            # y: downwards; move box origin from bottom center (kitti) to
            # true center (waymo)
            y -= height / 2
            # frame transformation: kitti -> waymo
            x, y, z = self.transform(T_k2w, x, y, z)

            # different conventions
            heading = -(rotation_y + np.pi / 2)
            while heading < -np.pi:
                heading += 2 * np.pi
            while heading > np.pi:
                heading -= 2 * np.pi

            box = label_pb2.Label.Box()
            box.center_x = x
            box.center_y = y
            box.center_z = z
            box.length = length
            box.width = width
            box.height = height
            box.heading = heading

            o = metrics_pb2.Object()
            o.object.box.CopyFrom(box)
            o.object.type = self.k2w_cls_map[cls]
            o.score = score

            o.context_name = context_name
            o.frame_timestamp_micros = frame_timestamp_micros

            return o

        objects = metrics_pb2.Objects()

        for instance_idx in range(len(kitti_result['name'])):
            o = parse_one_object(instance_idx)
            objects.objects.append(o)

        return objects

    def convert_one(self, file_idx):
        """Convert action for single file.

        Args:
            file_idx (int): Index of the file to be converted.
        """
        file_pathname = self.waymo_tfrecord_pathnames[file_idx]
        if 's3://' in file_pathname and tf.__version__ >= '2.6.0':
            try:
                import tensorflow_io as tfio  # noqa: F401
            except ImportError:
                raise ImportError(
                    "Please run 'pip install tensorflow-io' to install tensorflow_io first."  # noqa: E501
                )
        file_data = tf.data.TFRecordDataset(file_pathname, compression_type='')

        for frame_num, frame_data in enumerate(file_data):
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(frame_data.numpy()))

            filename = f'{self.prefix}{file_idx:03d}{frame_num:03d}'

            context_name = frame.context.name
            frame_timestamp_micros = frame.timestamp_micros

            if filename in self.name2idx:
                if self.from_kitti_format:
                    for camera in frame.context.camera_calibrations:
                        # FRONT = 1, see dataset.proto for details
                        if camera.name == 1:
                            T_front_cam_to_vehicle = np.array(
                                camera.extrinsic.transform).reshape(4, 4)

                    T_k2w = T_front_cam_to_vehicle @ self.T_ref_to_front_cam

                    kitti_result = \
                        self.results[self.name2idx[filename]]
                    objects = self.parse_objects(kitti_result, T_k2w,
                                                 context_name,
                                                 frame_timestamp_micros)
                else:
                    index = self.name2idx[filename]
                    objects = self.parse_objects_from_origin(
                        self.results[index], context_name,
                        frame_timestamp_micros)

            else:
                print(filename, 'not found.')
                objects = metrics_pb2.Objects()

            with open(
                    join(self.waymo_results_save_dir, f'{filename}.bin'),
                    'wb') as f:
                f.write(objects.SerializeToString())

    def convert_one_fast(self, res_index: int):
    def convert_one(self, res_idx: int):
        """Convert action for single file. It read the metainfo from the
        preprocessed file offline and will be faster.

        Args:
            res_idx (int): The indices of the results.
        """
        sample_idx = self.results[res_index]['sample_idx']
        if len(self.results[res_index]) > 0:
        sample_idx = self.results[res_idx]['sample_idx']
        if len(self.results[res_idx]['labels_3d']) > 0:
            objects = self.parse_objects_from_origin(
                self.results[res_idx], self.results[res_idx]['context_name'],
                self.results[res_idx]['timestamp'])
        else:
            print(sample_idx, 'not found.')
            objects = metrics_pb2.Objects()

        return objects

    def parse_objects_from_origin(self, result: dict, contextname: str,
                                  timestamp: str) -> Objects:
        """Parse obejcts from the original prediction results.

        Args:
            result (dict): The original prediction results.
            contextname (str): The ``contextname`` of sample in waymo.
            timestamp (str): The ``timestamp`` of sample in waymo.

        Returns:
            metrics_pb2.Objects: The parsed object.
        """
        lidar_boxes = result['bboxes_3d']
        scores = result['scores_3d']
        labels = result['labels_3d']

        objects = metrics_pb2.Objects()
        for lidar_box, score, label in zip(lidar_boxes, scores, labels):
            # Parse one object
            box = label_pb2.Label.Box()
            height = lidar_box[5]
            heading = lidar_box[6]

            box.center_x = lidar_box[0]
            box.center_y = lidar_box[1]
            box.center_z = lidar_box[2] + height / 2
            box.length = lidar_box[3]
            box.width = lidar_box[4]
            box.height = height
            box.heading = heading

            object = metrics_pb2.Object()
            object.object.box.CopyFrom(box)

            class_name = self.classes[label]
            object.object.type = self.k2w_cls_map[class_name]
            object.score = score
            object.context_name = contextname
            object.frame_timestamp_micros = timestamp
            objects.objects.append(object)

        return objects

    def convert(self):
        """Convert action."""
        print_log('Start converting ...', logger='current')

        # TODO: use parallel processes.
        # objects_list = mmengine.track_parallel_progress(
        #     self.convert_one, range(len(self)), self.num_workers)

        objects_list = mmengine.track_progress(self.convert_one,
                                               range(len(self)))

        combined = metrics_pb2.Objects()
        for objects in objects_list:
            for o in objects.objects:
                combined.objects.append(o)

        with open(self.waymo_results_final_path, 'wb') as f:
            f.write(combined.SerializeToString())

    def __len__(self):
        """Length of the filename list."""
        return len(self.results)
