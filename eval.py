import argparse
import data
import constants
import numpy as np
import os
import pointnet
import pvconvnet
import time

def example_eval(model, x, y):
  start_time = time.time()
  out = model.predict(x)
  end_time = time.time()
  return end_time - start_time
  
def eval_model(model, dataset):
  ious = model.eval(dataset)
  print('Avg IOU: %.3f' % np.mean(ious))

  time_sum = 0
  count = 0
  for x, y in dataset:
    time_sum += example_eval(model, x, y)
    count += 1

  time_avg = time_sum / count
  print('Avg time: %.5fs' % time_avg)

def evaluate(model_name: str, checkpoint_dir: str):
  test_dataset = data.GetDataset(constants.TEST_DATA_PATTERN)
  model_cp_dir = os.path.join(checkpoint_dir, model_name)

  if model_name == constants.POINTNET_MODEL_NAME:
    model = pointnet.PointNetModel(constants.NUM_CLASSES, model_cp_dir)
    eval_model(model, test_dataset)
  else:
    raise NotImplementedError('Model %s is not implemented yet.' % model_name)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='Evaluate a point cloud semseg model.')
  parser.add_argument(
      '--model', type=str, choices=constants.MODELS,
      help='Type of model to train.')
  parser.add_argument(
      '--checkpoint_dir', type=str, default='~/checkpoint',
      help='Directory for reading checkpoints.')
  args = parser.parse_args()
  evaluate(
      args.model,
      os.path.expanduser(args.checkpoint_dir))

 
