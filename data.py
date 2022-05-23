import numpy as np
import tensorflow as tf
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

def RecordToFrame(record):
  frame = open_dataset.Frame()
  frame.ParseFromString(bytearray(record.numpy()))
  return frame

def HasSegmentationLabels(points, labels):
  # return labels[0] >= 0
  return tf.math.reduce_min(labels) >= 0

# Function from
# https://github.com/waymo-research/waymo-open-dataset/blob/master/tutorial/tutorial_3d_semseg.ipynb
def convert_range_image_to_point_cloud_labels(frame,
                                              range_images,
                                              segmentation_labels,
                                              ri_index=0):
  """Convert segmentation labels from range images to point clouds.

  Args:
    frame: open dataset frame
    range_images: A dict of {laser_name, [range_image_first_return,
       range_image_second_return]}.
    segmentation_labels: A dict of {laser_name, [range_image_first_return,
       range_image_second_return]}.
    ri_index: 0 for the first return, 1 for the second return.

  Returns:
    point_labels: {[N, 2]} list of 3d lidar points's segmentation labels. 0 for
      points that are not labeled.
  """
  calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
  point_labels = []
  for c in calibrations:
    range_image = range_images[c.name][ri_index]
    range_image_tensor = tf.reshape(
        tf.convert_to_tensor(range_image.data), range_image.shape.dims)
    range_image_mask = range_image_tensor[..., 0] > 0

    if c.name in segmentation_labels:
      sl = segmentation_labels[c.name][ri_index]
      sl_tensor = tf.reshape(tf.convert_to_tensor(sl.data), sl.shape.dims)
      sl_points_tensor = tf.gather_nd(sl_tensor, tf.where(range_image_mask))
    else:
      num_valid_point = tf.math.reduce_sum(tf.cast(range_image_mask, tf.int32))
      sl_points_tensor = tf.zeros([num_valid_point, 2], dtype=tf.int32)
      
    point_labels.append(sl_points_tensor.numpy())
  return point_labels

def GetPointsAndLabels(frame):
  if len(frame.lasers[0].ri_return1.segmentation_label_compressed) == 0:
    return [np.zeros((1, 3)), -2*np.ones((1,))]
  (range_images, camera_projections, segmentation_labels,
 range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(
    frame)
  points, cp_points = frame_utils.convert_range_image_to_point_cloud(
      frame, range_images, camera_projections, range_image_top_pose)
  # 3d points in vehicle frame.
  points_all = np.concatenate(points, axis=0)
  point_labels = convert_range_image_to_point_cloud_labels(
       frame, range_images, segmentation_labels)
  point_labels_all = np.concatenate(point_labels, axis=0)
  # For now only get first return labels
  # instance id: point_labels_all[:, 0]
  # semantic class: point_labels_all[:, 1]
  return points_all, point_labels_all[:, 1]

def EnsureShape(points, labels):
  points = tf.ensure_shape(points, [None, 3])
  labels = tf.ensure_shape(labels, [None])
  return points, labels

def RecordToPointsAndLabels(record):
  frame = RecordToFrame(record)
  return GetPointsAndLabels(frame)

# Builds a TF dataset of pairs (points, label).
# points: (N x 3) 3D cartesian coordinates for each point in the point cloud.
# label: (N) per-point labels.
# Note that the number of points can vary between examples.
def GetDataset(file_pattern: str) -> tf.data.Dataset:
  file_dataset = tf.data.Dataset.list_files(file_pattern)
  dataset = tf.data.TFRecordDataset(file_dataset, compression_type='')
  dataset = dataset.map(
      lambda record:
      tf.py_function(func=RecordToPointsAndLabels, inp=[record],
                     Tout=[tf.float32, tf.float32]))
  dataset = dataset.map(EnsureShape)
  dataset = dataset.filter(HasSegmentationLabels)
  # Using batch sizes of 1 because examples have variable number of points.
  dataset = dataset.batch(1)
  return dataset
