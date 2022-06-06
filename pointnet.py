# Based on https://arxiv.org/pdf/1612.00593.pdf but missing normalization.
import metrics
import os
import tensorflow as tf
from typing import List
import util

class PointNetKerasModel(tf.keras.Model):
  def __init__(self, num_classes=23):
    super(PointNetKerasModel, self).__init__()
    self.local_mlp1 = tf.keras.layers.Dense(64, activation='relu')
    self.bn1 = tf.keras.layers.BatchNormalization()
    self.mlp2 = tf.keras.layers.Dense(128, activation='relu')
    self.bn2 = tf.keras.layers.BatchNormalization()
    # 1024 causes OOM, so using 512 instead below.
    # self.mlp3 = tf.keras.layers.Dense(1024, activation='relu')
    self.mlp3 = tf.keras.layers.Dense(512, activation='relu')
    self.bn3 = tf.keras.layers.BatchNormalization()
    self.mlp4 = tf.keras.layers.Dense(512, activation='relu')
    self.bn4 = tf.keras.layers.BatchNormalization()
    self.mlp5 = tf.keras.layers.Dense(256, activation='relu')
    self.bn5 = tf.keras.layers.BatchNormalization()
    self.mlp6 = tf.keras.layers.Dense(128, activation='relu')
    self.bn6 = tf.keras.layers.BatchNormalization()
    self.final_mlp = tf.keras.layers.Dense(num_classes)
      

  @tf.function(experimental_relax_shapes=True)
  def call(self, inputs, training=False):
    inputs = tf.ensure_shape(inputs, (1, None, 3))
    # Local features shape: 1 x n x 64
    local_features = self.local_mlp1(inputs)
    local_features = tf.ensure_shape(local_features, (1, None, 64))
    local_features = self.bn1(local_features, training=training)
    # to shape 1 x n x 128
    x = self.mlp2(local_features)
    x = self.bn2(x, training=training)
    # to shape 1 x n x 1024
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
    # to 1 x n x num classes
    logits = self.final_mlp(local_global_features)
    return logits

class PointNetModel:
  def __init__(self, num_classes: int, checkpoint_dir: str, iou_only_epoch: bool):
    self.checkpoint_dir = checkpoint_dir
    self.model = PointNetKerasModel(num_classes)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    self.model.compile(optimizer='adam',
        loss=loss_fn,
        metrics = [metrics.MeanIOUFromLogits(
                   num_classes=num_classes, only_epoch=iou_only_epoch),
                   tf.keras.metrics.SparseCategoricalAccuracy()])
    self.epoch = util.LoadLatestCheckpoint(self.model, checkpoint_dir)

  def train(
      self, train_dataset: tf.data.Dataset, val_dataset: tf.data.Dataset,
      tensorboard_dir: str, epochs: int, steps_per_epoch: int):
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_dir)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(self.checkpoint_dir, util.CHECKPOINT_FILE_PATTERN),
        save_weights_only=True)

    self.model.fit(
        train_dataset, epochs=epochs, # validation_data=val_dataset,
        callbacks=[tensorboard_callback, checkpoint_callback, metrics.ToggleMetrics()],
        initial_epoch=self.epoch,
        steps_per_epoch=steps_per_epoch)

  def predict(self, x: tf.Tensor) -> tf.Tensor:
    return self.model(x)

  def eval(self, dataset: tf.data.Dataset, steps: int) -> List[float]:
    """Compute IOU for the dataset."""
    metrics = self.model.evaluate(dataset, steps=steps)
    return metrics[1:]  # skip loss and only keep iou?

