import tensorflow as tf
import os

CHECKPOINT_FILE_PATTERN = 'cp-{epoch:02d}.ckpt'

def GetEpochFromCheckpointPath(checkpoint_path):
  file_name = os.path.basename(checkpoint_path)
  epoch_str = file_name[len('cp-'):-len('.ckpt')]
  return int(epoch_str)

def LoadLatestCheckpoint(model: tf.keras.Model, checkpoint_dir: str) -> int:
  """Loads the latest checkpoint and returns its epoch number.

  Returns: epoch number or 0 if no checkpoint is found.
  """
  latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
  if latest_checkpoint is not None:
    epoch = GetEpochFromCheckpointPath(latest_checkpoint)
    model.load_weights(latest_checkpoint)
    print('Loaded checkpoint for epoch %i.' % epoch)
    return epoch
  else:
    print('No checkpoint found.')
    return 0

