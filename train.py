import argparse
import data
import constants
import pointnet
import os

def train(
    model_name: str, epochs: int, tensorboard_dir: str, checkpoint_dir: str):
  train_dataset = data.GetDataset(constants.TRAIN_DATA_PATTERN)
  val_dataset = data.GetDataset(constants.VALIDATION_DATA_PATTERN)
  model_tb_dir = os.path.join(tensorboard_dir, model_name)
  model_cp_dir = os.path.join(checkpoint_dir, model_name)

  if model_name == constants.POINTNET_MODEL_NAME:
    os.makedirs(model_cp_dir, exist_ok=True)
    model = pointnet.PointNetModel(constants.NUM_CLASSES, model_cp_dir)
    model.train(train_dataset, val_dataset, model_tb_dir, epochs)
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
  args = parser.parse_args()
  train(
      args.model,
      args.epochs,
      os.path.expanduser(args.tensorboard_dir),
      os.path.expanduser(args.checkpoint_dir))

  
