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
  
def eval_model(model, dataset, steps):
  print('===================================================================')
  print('Evaluating model')
  ious = model.eval(dataset, steps)
  print('Avg IOU: %.3f' % np.mean(ious))
  print('===================================================================')

def benchmark_model(model, dataset, steps):
  # Warmup
  for i, (x, y) in enumerate(dataset):
    example_eval(model, x, y)
    if i >= 5:
      break

  time_sum = 0
  count = 0
  for x, y in dataset:
    time_sum += example_eval(model, x, y)
    count += 1
    if count >= steps:
      break

  if count == 0:
    print('No examples in the dataset')

  time_avg = time_sum / count
  print('===================================================================')
  print('Avg time (%i examples): %.5fs' % (count, time_avg))
  print('===================================================================')

def eval_and_benchmark(model, dataset, steps):
  eval_model(model, dataset, steps)
  benchmark_model(model, dataset, steps)

def evaluate(model_name: str, checkpoint_dir: str, steps: int):
  # test_dataset = data.GetDataset(constants.TEST_DATA_PATTERN)
  # Test dataset reading hangs somehow, using validation dataset instead for now.
  test_dataset = data.GetDataset(constants.VALIDATION_DATA_PATTERN)
  model_cp_dir = os.path.join(checkpoint_dir, model_name)

  if model_name == constants.POINTNET_MODEL_NAME:
    model = pointnet.PointNetModel(
        constants.NUM_CLASSES, model_cp_dir, iou_only_epoch=False)
    eval_and_benchmark(model, test_dataset, steps)
  elif model_name == constants.PVCONV_MODEL_NAME:
    model = pvconvnet.PVConvModel(
        constants.NUM_CLASSES, pvconvnet.SparseType.DENSE,
        model_cp_dir, iou_only_epoch=False)
    eval_and_benchmark(model, test_dataset, steps)
  elif model_name == constants.PVCONV_SPARSE_BASIC_MODEL_NAME:
    model = pvconvnet.PVConvModel(
        constants.NUM_CLASSES, pvconvnet.SparseType.SPARSE_BASIC,
        model_cp_dir, iou_only_epoch=False)
    eval_and_benchmark(model, test_dataset, steps)
  elif model_name == constants.PVCONV_SPARSE_SYMM_MODEL_NAME:
    model = pvconvnet.PVConvModel(
        constants.NUM_CLASSES, pvconvnet.SparseType.SPARSE_SYMM,
        model_cp_dir, iou_only_epoch=False)
    eval_and_benchmark(model, test_dataset, steps)
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
  parser.add_argument(
      '--steps', type=int, default=100,
      help='Steps to run during eval.')
  args = parser.parse_args()
  evaluate(
      args.model,
      os.path.expanduser(args.checkpoint_dir),
      args.steps)

 
