import constants as const
import math
import metrics
import os
import tensorflow as tf
from tensorflow_graphics.math import interpolation
from typing import List
import util

VOXEL_SIZE_INV = 1.0 / const.PVCONV_VOXEL_SIZE
VOXEL_DIM_X = math.ceil((const.X_MAX - const.X_MIN) * VOXEL_SIZE_INV)
VOXEL_DIM_Y = math.ceil((const.Y_MAX - const.Y_MIN) * VOXEL_SIZE_INV)
VOXEL_DIM_Z = math.ceil((const.Z_MAX - const.Z_MIN) * VOXEL_SIZE_INV)
MIN_CORNER = [const.X_MIN, const.Y_MIN, const.Z_MIN]


def pts_to_voxel_indexes(pt_coords):
  voxel_idx = (pt_coords - MIN_CORNER) * VOXEL_SIZE_INV
  voxel_idx = tf.dtypes.cast(voxel_idx, tf.int32)
  # voxel_idx = tf.ensure_shape(voxel_idx, (1, None, 3))
  return voxel_idx

def points_per_voxel(pt_coords, voxel_indexes):
  # pt_coords = tf.ensure_shape(pt_coords, (1, None, 3))
  count_features = tf.ones(tf.shape(pt_coords)[:-1])
  # Indexes: 1 x N x 3
  # Features: 1 x N
  pts_per_voxel = tf.scatter_nd(
      voxel_indexes,
      count_features, shape=(VOXEL_DIM_X, VOXEL_DIM_Y, VOXEL_DIM_Z))
  # TODO: figure out how to keep the batch dimension in the pts_per_voxel above.
  # In our case it doesn't really matter because batch = 1.
  pts_per_voxel = tf.expand_dims(pts_per_voxel, axis=0)
  pts_per_voxel = tf.expand_dims(pts_per_voxel, axis=-1)
  return pts_per_voxel

class PVConvLayer(tf.keras.layers.Layer):
  def __init__(self, num_outputs):
    super(PVConvLayer, self).__init__()
    self.num_outputs = num_outputs
    self.mlp = tf.keras.layers.Dense(self.num_outputs, activation='relu')
    self.bn_mlp = tf.keras.layers.BatchNormalization()
    self.conv1 = tf.keras.layers.Conv3D(
        filters=self.num_outputs, kernel_size=(3, 3, 3),
        padding='same', activation='relu')
    self.bn_conv1 = tf.keras.layers.BatchNormalization()

  def build(self, input_shape):
    self.mlp.build(input_shape)
    self.conv1.build(input_shape)

  def call(self, inputs, pt_coords, voxel_indexes, pts_per_voxel_inv, training):
    mlp_out = self.mlp(inputs)
    mlp_out = self.bn_mlp(mlp_out, training=training)

    # Sum per-point features for each voxel.
    voxels = tf.scatter_nd(
       voxel_indexes, inputs,
       shape=(VOXEL_DIM_X, VOXEL_DIM_Y, VOXEL_DIM_Z, inputs.shape[-1]))
    # TODO: figure out how to keep the batch dimension in the pts_per_voxel above.
    # In our case it doesn't really matter because batch = 1.
    voxels = tf.expand_dims(voxels, axis=0)

    # Compute means of per-voxel features.
    voxels = voxels * pts_per_voxel_inv
    conv_out = self.conv1(voxels)
    conv_out = self.bn_conv1(conv_out, training=training)
    # Use trilinear interpolation to assign point features based on voxel
    # features.
    vox_out = interpolation.trilinear.interpolate(
        conv_out,
        pt_coords)
    # Fuse per-point MLP and voxel conv outputs using addition.
    pv_out = mlp_out + vox_out
    return pv_out

class PVConvKerasModel(tf.keras.Model):
  def __init__(self, num_classes=23, channel_mult=0.5):
    super(PVConvKerasModel, self).__init__()
    self.local_pvconv1 = PVConvLayer(64*channel_mult)
    self.pvconv2 = PVConvLayer(128*channel_mult)
    self.pvconv3 = PVConvLayer(1024*channel_mult)
    self.pvconv4 = PVConvLayer(512*channel_mult)
    self.pvconv5 = PVConvLayer(256*channel_mult)
    self.pvconv6 = PVConvLayer(128*channel_mult)
    self.final_mlp = tf.keras.layers.Dense(num_classes)
      

  def call(self, inputs, training=False):
    inputs = tf.ensure_shape(inputs, (1, None, 3))
    pt_coords = inputs

    # Precompute voxel indexes for each points and the number of points per voxel.
    voxel_indexes = pts_to_voxel_indexes(pt_coords)
    pts_per_voxel = points_per_voxel(inputs, voxel_indexes)
    pts_per_voxel_inv = tf.math.divide_no_nan(1.0, pts_per_voxel)

    #def call(self, inputs, pt_coords, voxel_multipliers, training):
    # Local features shape: 1 x n x 64
    local_features = self.local_pvconv1(
            inputs, pt_coords, voxel_indexes, pts_per_voxel_inv, training=training)
    # to shape 1 x n x 128
    x = self.pvconv2(
        local_features, pt_coords, voxel_indexes, pts_per_voxel_inv,
        training=training)
    # to shape 1 x n x 1024
    x = self.pvconv3(
        x, pt_coords, voxel_indexes, pts_per_voxel_inv, training=training)
    # 1 x n x 1024 --> 1 x 1 x 1024
    global_features = tf.reduce_max(x, axis=-2, keepdims=True)
    # 1 x 1 x 1024 --> 1 x n x 1024
    global_features = tf.tile(global_features, (1, tf.shape(local_features)[1], 1))
    # Concatenate over last dimension: 1024 + 64 = 1088
    x = tf.keras.layers.concatenate([local_features, global_features], axis=-1)
    # to shape 1 x n x 512
    x = self.pvconv4(
        x, pt_coords, voxel_indexes, pts_per_voxel_inv, training=training)
    # to shape 1 x n x 256
    x = self.pvconv5(
        x, pt_coords, voxel_indexes, pts_per_voxel_inv, training=training)
    # to shape 1 x n x 128
    local_global_features = self.pvconv6(
        x, pt_coords, voxel_indexes, pts_per_voxel_inv, training=training)
    # to 1 x n x num classes
    logits = self.final_mlp(local_global_features)
    return logits


class PVConvModel:
  def __init__(self, num_classes: int, checkpoint_dir: str):
    self.checkpoint_dir = checkpoint_dir
    self.model = PVConvKerasModel(num_classes)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    self.model.compile(optimizer='adam',
        loss=loss_fn,
        metrics = [metrics.MeanIOUFromLogits(
                   num_classes=num_classes)])
    self.epoch = util.LoadLatestCheckpoint(self.model, checkpoint_dir)

  def train(
      self, train_dataset: tf.data.Dataset, val_dataset: tf.data.Dataset,
      tensorboard_dir: str, epochs: int):
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_dir)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(self.checkpoint_dir, util.CHECKPOINT_FILE_PATTERN),
        save_weights_only=True)

    self.model.fit(
        train_dataset, epochs=epochs, # validation_data=val_dataset,
        callbacks=[tensorboard_callback, checkpoint_callback, metrics.ToggleMetrics()],
        initial_epoch=self.epoch)

  def predict(self, x: tf.Tensor) -> tf.Tensor:
    return self.model(x)

  def eval(self, dataset: tf.data.Dataset) -> List[float]:
    """Compute IOU for the dataset."""
    metrics = self.model.evaluate(dataset)
    print(self.model.metric_names)
    return metrics[1:]  # skip loss and only keep iou?

