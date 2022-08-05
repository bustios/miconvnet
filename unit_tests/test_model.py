# coding=utf-8

import tensorflow as tf

from .. import model as model_utils
from .. import utils


class ModelTests(tf.test.TestCase):
  """Test model construction and training."""

  def _test_model(self, model_name, dataset_name):
    """Build model `model_name` configured for dataset `dataset_name`.

    Arguments:
      model_name: one of `MIConvNet, SincShallowNet, EEGNet, ShallowConvNet`.
      dataset_name: one of `bci2a, bci2b`.
    """
    hparams = utils.read_config(model_name, dataset_name)
    csp = hparams['apply_csp']
    num_channels = hparams['num_csp'] if csp else hparams['num_eeg_channels']
    resolution = (num_channels, hparams['window_len'], 1)
    batch_size = 2

    model = model_utils.build_model(hparams)
    optimizer = tf.optimizers.Adam(hparams['learning_rate'])

    input_shape = (batch_size, resolution[0], resolution[1], resolution[2])
    random_input = tf.random.uniform(input_shape)
    random_output = tf.zeros(batch_size, dtype=tf.int32)

    if hparams['num_classes'] == 2:
      loss_fn = tf.losses.BinaryCrossentropy(from_logits=True)
      train_accuracy = tf.metrics.BinaryAccuracy(threshold=0.0)
    else:
      loss_fn = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
      train_accuracy = tf.metrics.SparseCategoricalAccuracy()

    train_loss = tf.metrics.Mean('train_loss', dtype=tf.float32)

    with tf.GradientTape() as tape:
      logits = model(random_input, training=True)
      loss_value = loss_fn(random_output, logits)

    # Get and apply gradients.
    gradients = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    train_loss(loss_value)
    train_loss.result()
    train_accuracy(random_output, logits)
    train_accuracy.result()

    model.summary()
    assert True  # If we make it to this line, we're all good!

  def test_miconvnet_bci2a(self):
    """Test MIConvNet for dataset bci2a."""
    self._test_model('MIConvNet', 'bci2a')

  def test_miconvnet_bci2b(self):
    """Test MIConvNet for dataset bci2b."""
    self._test_model('MIConvNet', 'bci2b')

  def test_sinc_shallow_net_bci2a(self):
    """Test SincShallowNet for dataset bci2a."""
    self._test_model('SincShallowNet', 'bci2a')

  def test_sinc_shallow_net_bci2b(self):
    """Test SincShallowNet for dataset bci2b."""
    self._test_model('SincShallowNet', 'bci2b')

  def test_shallow_conv_net_bci2a(self):
    """Test ShallowConvNet for dataset bci2a."""
    self._test_model('ShallowConvNet', 'bci2a')

  def test_shallow_conv_net_bci2b(self):
    """Test ShallowConvNet for dataset bci2b."""
    self._test_model('ShallowConvNet', 'bci2b')

  def test_eegnet_bci2a(self):
    """Test EEGNet for dataset bci2a."""
    self._test_model('EEGNet', 'bci2a')

  def test_eegnet_bci2b(self):
    """Test EEGNet for dataset bci2b."""
    self._test_model('EEGNet', 'bci2b')

if __name__ == '__main__':
  tf.test.main()