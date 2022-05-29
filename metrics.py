import tensorflow as tf



class MeanIOUFromLogits(tf.keras.metrics.MeanIoU):
  def __init__(self, only_epoch, **kwargs):
    super(MeanIOUFromLogits, self).__init__(**kwargs)
    self.only_epoch = only_epoch
    self.update_metric = tf.Variable(False)    
  
  def update_state(self, y_true, y_pred, sample_weight=None):
    if not self.only_epoch or self.update_metric:
      pred_label = tf.argmax(y_pred, axis=-1)
      super().update_state(y_true, pred_label, sample_weight)

# https://stackoverflow.com/questions/56826495/how-to-make-keras-compute-a-certain-metric-on-validation-data-only
class ToggleMetrics(tf.keras.callbacks.Callback):
  '''On test begin (i.e. when evaluate() is called or
   validation data is run during fit()) toggle metric flag '''
  def on_test_begin(self, logs):
    for metric in self.model.metrics:
      if 'MeanIOUFromLogits' in metric.name:
        metric.update_metric.assign(True)

  def on_test_end(self,  logs):
    for metric in self.model.metrics:
      if 'MeanIOUFromLogits' in metric.name:
        metric.update_metric.assign(False)
