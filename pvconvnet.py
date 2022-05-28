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

def to_sparse(voxel_grid):
  #voxel_grid = tf.ensure_shape(voxel_grid, (VOXEL_DIM_X, VOXEL_DIM_Y, VOXEL_DIM_Z, 3))
  zero = tf.constant(0, dtype=tf.float32)
  voxel_grid_not_zero = tf.math.not_equal(tf.reduce_max(voxel_grid, axis=-1), zero)
  # Non-zero voxel indexes
  voxel_idx = tf.where(voxel_grid_not_zero)
  voxel_features = tf.gather_nd(params=voxel_grid, indices=voxel_idx)
  return tf.cast(voxel_idx, tf.dtypes.int32), voxel_features

def from_sparse(voxel_idx, voxel_features, voxel_grid_shape):
  # voxel_grid = tf.scatter_nd(voxel_idx, voxel_features, voxel_grid_shape)
  voxel_grid = tf.scatter_nd(
      voxel_idx, voxel_features, voxel_grid_shape, name='scatter_sparse_to_dense')
  return voxel_grid

def coords_to_keys(idx):
    return idx[:, 0] * VOXEL_DIM_X * VOXEL_DIM_Y + idx[:, 1] * VOXEL_DIM_Y + idx[:, 2]

@tf.function
def sparse_conv3d(voxel_idx, voxel_features, weights, pts_per_voxel_inv):

  # voxel_idx has shape batch x num pts x coord
  # We use batch size of 1, so order is based on num pts.
  voxel_idx_order = tf.range(tf.shape(voxel_idx)[-2])
  # Map from voxel indexes to their order in voxel_idx
  # TODO: convert voxel_idx to single number to use as hash
  # voxel_idx_to_order = tf.lookup.StaticHashTable(
  #   tf.lookup.KeyValueTensorInitializer(coords_to_keys(voxel_idx), voxel_idx_order),
  #   default_value=-1)
  # Only support output size == input size
  out_tensor = tf.zeros([tf.shape(voxel_features)[0], tf.shape(weights)[-1]])
  # For each weight position
  # For each output, lookup which inputs are non-zero
  for offset_x in range(-1, 2):
    for offset_y in range(-1, 2):
      for offset_z in range(-1, 2):
        out_voxel_idx = voxel_idx
        in_offset_from_out = tf.constant(
            [[offset_x, offset_y, offset_z]], dtype=tf.int32)

        # N x I
        in_voxel_idx = out_voxel_idx + in_offset_from_out

        # Only keep input voxel indexes that are in our map.
        # N x 1, value is -1 if missing
        # Gather the inv grid and check non-zero
        # order = voxel_idx_to_order.lookup(coords_to_keys(in_voxel_idx))
        valid_idx = tf.where(
            tf.gather_nd(params=pts_per_voxel_inv[0, :, :, :, 0], indices=in_voxel_idx))
        # Concatenate features for inputs that are available
        # Non zero indexes,
        # valid_idx = tf.where(tf.math.greater_equal(order, 0))
        input_features = tf.gather_nd(params=voxel_features, indices=valid_idx)

        # Matmul input_features with the weight matrix
        # return tf.matmul(input_features, weights[0, 0, 0])
        output_features = tf.matmul(
            input_features, weights[offset_x + 1, offset_y + 1, offset_z + 1])
        # Scatter outputs to their locations in the output.
        # Output features are added across weights.
        out_tensor = tf.tensor_scatter_nd_add(
            out_tensor, valid_idx, output_features,
            name='sparseconv_scatter')
  return out_tensor
       

@tf.function
def pts_to_voxel_indexes(pt_coords):
  voxel_idx = (pt_coords - MIN_CORNER) * VOXEL_SIZE_INV - 0.5
  voxel_idx = tf.dtypes.cast(voxel_idx, tf.int32)
  # voxel_idx = tf.ensure_shape(voxel_idx, (1, None, 3))
  return voxel_idx

@tf.function
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


class SparseConv3D(tf.keras.layers.Layer):
  def __init__(self, num_outputs):
    super(SparseConv3D, self).__init__()
    self.num_outputs = num_outputs
    self.kernel = None

  def build(self, input_shape):
    self.kernel = self.add_weight(
        "kernel",
        shape=[3, 3, 3, int(input_shape[-1]), int(self.num_outputs)],
        trainable=True)
    
  def call(self, inputs, voxel_idx, pts_per_voxel_inv):
    # tf.print(tf.reduce_sum(self.kernel[1, 0, 2]))
    voxels = sparse_conv3d(voxel_idx, inputs, self.kernel, pts_per_voxel_inv)
    return tf.keras.layers.ReLU()(voxels)
   

class PVConvBlock(tf.keras.Model):
  def __init__(self, num_outputs):
    super(PVConvBlock, self).__init__()
    self.num_outputs = int(num_outputs)
    self.mlp = tf.keras.layers.Dense(self.num_outputs, activation='relu')
    self.bn_mlp = tf.keras.layers.BatchNormalization()
    self.conv1 = tf.keras.layers.Conv3D(
        filters=self.num_outputs, kernel_size=(3, 3, 3),
        padding='same', activation='relu', data_format='channels_first')
    self.conv2 = tf.keras.layers.Conv3D(
        filters=self.num_outputs, kernel_size=(3, 3, 3),
        padding='same', activation='relu', data_format='channels_first')

    self.sparseconv1 = SparseConv3D(self.num_outputs)
    self.sparseconv2 = SparseConv3D(self.num_outputs)

    self.bn_conv1 = tf.keras.layers.BatchNormalization()
    self.bn_conv2 = tf.keras.layers.BatchNormalization()

  # @tf.function
  def call(self, inputs, pt_coords, voxel_indexes, pts_per_voxel_inv, training):
    mlp_out = self.mlp(inputs)
    mlp_out = self.bn_mlp(mlp_out, training=training)

    # Sum per-point features for each voxel.
    voxels = tf.scatter_nd(
       voxel_indexes, inputs,
       shape=(VOXEL_DIM_X, VOXEL_DIM_Y, VOXEL_DIM_Z, inputs.shape[-1]),
       name='scatter_to_voxels')
    voxel_idx, voxel_features = to_sparse(voxels)
    # voxel_features = tf.ensure_shape(voxel_features, (None, 3))
    voxel_idx = tf.ensure_shape(voxel_idx, (None, 3))
    voxel_features = self.sparseconv1(voxel_features, voxel_idx, pts_per_voxel_inv)
    voxel_features = self.sparseconv2(voxel_features, voxel_idx, pts_per_voxel_inv)
    # def from_sparse(voxel_idx, voxel_features, voxel_grid_shape):
    voxels = from_sparse(
        voxel_idx, voxel_features,
        (VOXEL_DIM_X, VOXEL_DIM_Y, VOXEL_DIM_Z, self.num_outputs))
    # tf.debugging.assert_near(voxels, voxels2, rtol=1e-5)

    # TODO: figure out how to keep the batch dimension in the pts_per_voxel above.
    # In our case it doesn't really matter because batch = 1.
    conv_out = tf.expand_dims(voxels, axis=0)

    # Compute means of per-voxel features.
    # voxels = voxels * pts_per_voxel_inv
    # voxels = tf.transpose(voxels, (0, 4, 1, 2, 3))
    # conv_out = self.conv1(voxels)
    # conv_out = self.bn_conv1(conv_out, training=training)
    # conv_out = self.conv2(conv_out)
    # conv_out = self.bn_conv2(conv_out, training=training)
    # conv_out = tf.transpose(conv_out, (0, 2, 3, 4, 1))

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
    self.local_pvconv1 = PVConvBlock(64*channel_mult)
    self.pvconv2 = PVConvBlock(128*channel_mult)
    #self.pvconv3 = PVConvBlock(1024*channel_mult)
    self.mlp3 = tf.keras.layers.Dense(64*channel_mult, activation='relu')
    self.bn3 = tf.keras.layers.BatchNormalization()

    self.mlp4 = tf.keras.layers.Dense(512, activation='relu')
    self.bn4 = tf.keras.layers.BatchNormalization()
    self.mlp5 = tf.keras.layers.Dense(256, activation='relu')
    self.bn5 = tf.keras.layers.BatchNormalization()
    self.mlp6 = tf.keras.layers.Dense(128, activation='relu')
    self.bn6 = tf.keras.layers.BatchNormalization()

    # self.pvconv4 = PVConvBlock(512*channel_mult)
    # self.pvconv5 = PVConvBlock(256*channel_mult)
    # self.pvconv6 = PVConvBlock(128*channel_mult)

    self.final_mlp = tf.keras.layers.Dense(num_classes)
      
  @tf.function
  def call(self, inputs, training=False):
    inputs = tf.ensure_shape(inputs, (1, None, 3))
    pt_coords = inputs

    # Precompute voxel indexes for each points and the number of points per voxel.
    voxel_indexes = pts_to_voxel_indexes(pt_coords)
    pts_per_voxel = points_per_voxel(inputs, voxel_indexes)
    pts_per_voxel = tf.ensure_shape(
        pts_per_voxel, (1, VOXEL_DIM_X, VOXEL_DIM_Y, VOXEL_DIM_Z, 1))
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
    # x = self.pvconv3(
    #     x, pt_coords, voxel_indexes, pts_per_voxel_inv, training=training)
    x = self.mlp3(x)
    x = self.bn3(x, training=training)
    # 1 x n x 1024 --> 1 x 1 x 1024
    global_features = tf.reduce_max(x, axis=-2, keepdims=True)
    # 1 x 1 x 1024 --> 1 x n x 1024
    global_features = tf.tile(global_features, (1, tf.shape(local_features)[1], 1))
    # Concatenate over last dimension: 1024 + 64 = 1088
    x = tf.keras.layers.concatenate([local_features, global_features], axis=-1)

    # to shape 1 x n x 512
    x = self.mlp4(x)
    x = self.bn4(x, training=training)
    # to shape 1 x n x 256
    x = self.mlp5(x)
    x = self.bn5(x, training=training)
    # to shape 1 x n x 128
    local_global_features = self.mlp6(x)
    local_global_features = self.bn6(local_global_features, training=training)

    # # to shape 1 x n x 512
    # x = self.pvconv4(
    #     x, pt_coords, voxel_indexes, pts_per_voxel_inv, training=training)
    # # to shape 1 x n x 256
    # x = self.pvconv5(
    #     x, pt_coords, voxel_indexes, pts_per_voxel_inv, training=training)
    # # to shape 1 x n x 128
    # local_global_features = self.pvconv6(
    #     x, pt_coords, voxel_indexes, pts_per_voxel_inv, training=training)

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

