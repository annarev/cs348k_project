import argparse
import data
import constants
import numpy as np
import os
import pointnet
import pvconvnet
import tensorflow as tf
import time

tf.random.set_seed(123)

def example_eval(model, x, y):
  start_time = time.time()
  out = model.predict(x)
  end_time = time.time()
  return end_time - start_time
  
def eval_model(model, dataset, steps):
  print('===================================================================')
  print('Evaluating model')
  ious = model.eval(dataset, steps)
  print('Avg IOU: %.3f' % np.mean(ious))
  print('===================================================================')

def benchmark_model(model, dataset, steps, model_metric_dir):
  # Warmup
  for i, (x, y) in enumerate(dataset):
    example_eval(model, x, y)
    if i >= 5:
      break

  times = []
  for x, y in dataset:
    times.append(example_eval(model, x, y))
    if len(times) >= steps:
      break

  if not times:
    print('No examples in the dataset')

  time_avg = np.mean(times)
  print('===================================================================')
  print('Avg time (%i examples): %.5fs' % (len(times), time_avg))
  print('===================================================================')

  # Store npy file with time values
  times_file = os.path.join(model_metric_dir, 'times.npy')
  np.save(times_file, times)

def eval_and_benchmark(model, dataset, steps, model_metric_dir):
  eval_model(model, dataset, steps)
  benchmark_model(model, dataset, steps, model_metric_dir)

def evaluate(
    model_name: str, checkpoint_dir: str, eval_metric_dir: str, steps: int):
  # test_dataset = data.GetDataset(constants.TEST_DATA_PATTERN)
  # Test dataset reading hangs somehow, using validation dataset instead for now.
  test_dataset = data.GetDataset(
      constants.VALIDATION_DATA_PATTERN, deterministic=True)
  model_cp_dir = os.path.join(checkpoint_dir, model_name)
  model_metric_dir = os.path.join(eval_metric_dir, model_name)
  os.makedirs(model_metric_dir, exist_ok=True)

  model = None 
  if model_name == constants.POINTNET_MODEL_NAME:
    model = pointnet.PointNetModel(
        constants.NUM_CLASSES, model_cp_dir, iou_only_epoch=False)
  elif model_name == constants.PVCONV_MODEL_NAME:
    model = pvconvnet.PVConvModel(
        constants.NUM_CLASSES, pvconvnet.SparseType.DENSE,
        model_cp_dir, iou_only_epoch=False)
  elif model_name == constants.PVCONV_SPARSE_BASIC_MODEL_NAME:
    model = pvconvnet.PVConvModel(
        constants.NUM_CLASSES, pvconvnet.SparseType.SPARSE_BASIC,
        model_cp_dir, iou_only_epoch=False)
  elif model_name == constants.PVCONV_SPARSE_SYMM_MODEL_NAME:
    model = pvconvnet.PVConvModel(
        constants.NUM_CLASSES, pvconvnet.SparseType.SPARSE_SYMM,
        model_cp_dir, iou_only_epoch=False)
  else:
    raise NotImplementedError('Model %s is not implemented yet.' % model_name)
  eval_and_benchmark(model, test_dataset, steps, model_metric_dir)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='Evaluate a point cloud semseg model.')
  parser.add_argument(
      '--model', type=str, choices=constants.MODELS,
      help='Type of model to train.')
  parser.add_argument(
      '--checkpoint_dir', type=str, default='~/checkpoint',
      help='Directory for reading checkpoints.')
  parser.add_argument(
      '--eval_metric_dir', type=str, default='~/eval',
      help='Directory for storing eval metrics.')
  parser.add_argument(
      '--steps', type=int, default=100,
      help='Steps to run during eval.')
  args = parser.parse_args()
  evaluate(
      args.model,
      os.path.expanduser(args.checkpoint_dir),
      os.path.expanduser(args.eval_metric_dir),
      args.steps)

 
