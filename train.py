import argparse
import data
import constants
import pointnet
import pvconvnet
import os

def train(
    model_name: str, epochs: int, steps_per_epoch: int,
    tensorboard_dir: str, checkpoint_dir: str):
  train_dataset = data.GetDataset(constants.TRAIN_DATA_PATTERN)
  val_dataset = data.GetDataset(constants.VALIDATION_DATA_PATTERN)
  model_tb_dir = os.path.join(tensorboard_dir, model_name)
  model_cp_dir = os.path.join(checkpoint_dir, model_name)

  if model_name == constants.POINTNET_MODEL_NAME:
    os.makedirs(model_cp_dir, exist_ok=True)
    model = pointnet.PointNetModel(constants.NUM_CLASSES, model_cp_dir)
    model.train(train_dataset, val_dataset, model_tb_dir, epochs, steps_per_epoch)
  elif model_name == constants.PVCONV_MODEL_NAME:
    os.makedirs(model_cp_dir, exist_ok=True)
    model = pvconvnet.PVConvModel(
        constants.NUM_CLASSES, pvconvnet.SparseType.DENSE,
        model_cp_dir)
    model.train(train_dataset, val_dataset, model_tb_dir, epochs, steps_per_epoch)
  elif model_name == constants.PVCONV_SPARSE_BASIC_MODEL_NAME:
    os.makedirs(model_cp_dir, exist_ok=True)
    model = pvconvnet.PVConvModel(
        constants.NUM_CLASSES, pvconvnet.SparseType.SPARSE_BASIC,
        model_cp_dir)
    model.train(train_dataset, val_dataset, model_tb_dir, epochs, steps_per_epoch)
  elif model_name == constants.PVCONV_SPARSE_SYMM_MODEL_NAME:
    os.makedirs(model_cp_dir, exist_ok=True)
    model = pvconvnet.PVConvModel(
        constants.NUM_CLASSES, pvconvnet.SparseType.SPARSE_SYMM,
        model_cp_dir)
    model.train(train_dataset, val_dataset, model_tb_dir, epochs, steps_per_epoch)
  else:
    raise NotImplementedError('Model %s is not implemented yet.' % model_name)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='Train a point cloud semseg model.')
  parser.add_argument(
      '--model', type=str, choices=constants.MODELS, help='Type of model to train.')
  parser.add_argument(
      '--epochs', type=str, default=1, help='Number of epochs to train for.')
  parser.add_argument(
      '--tensorboard_dir', type=str, default='~/tensorboard',
      help='Directory for storing tensorboard summaries.')
  parser.add_argument(
      '--checkpoint_dir', type=str, default='~/checkpoint',
      help='Directory for storing checkpoints.')
  parser.add_argument(
      '--steps_per_epoch', type=int, default=100,
      help='Steps per epoch.')
  args = parser.parse_args()
  train(
      args.model,
      args.epochs,
      args.steps_per_epoch,
      os.path.expanduser(args.tensorboard_dir),
      os.path.expanduser(args.checkpoint_dir))

  
