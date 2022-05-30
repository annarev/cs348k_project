import argparse
import constants
import numpy as np
from scipy.stats import ttest_ind
import os

TIMES_FILE_NAME = 'times.npy'

def compute_pvalue(model1, model2, eval_metric_dir):
  model1_metric_dir = os.path.join(eval_metric_dir, model1)
  model2_metric_dir = os.path.join(eval_metric_dir, model2)
  model1_times = np.load(os.path.join(model1_metric_dir, TIMES_FILE_NAME))
  model2_times = np.load(os.path.join(model2_metric_dir, TIMES_FILE_NAME))

  t, p = ttest_ind(model1_times, model2_times)
  print('P-value: %.7f' % p)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='Compute p-values (null hypothesis: means are equal).')
  parser.add_argument(
      '--model1', type=str, choices=constants.MODELS,
      help='First model name.')
  parser.add_argument(
      '--model2', type=str, choices=constants.MODELS,
      help='Second model name.')
  parser.add_argument(
      '--eval_metric_dir', type=str, default='~/eval',
      help='Directory for storing eval metrics.')

  args = parser.parse_args()
  compute_pvalue(
      args.model1,
      args.model2,
      os.path.expanduser(args.eval_metric_dir))
