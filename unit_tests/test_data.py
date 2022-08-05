# coding=utf-8

import tensorflow as tf

from .. import data as data_utils
from .. import utils


class DataTests(tf.test.TestCase):
  """Test data loader."""

  def test_build_dataset(self):
    """Test data loader for datasets 2a and 2b of BCI Competition IV [1].

    [1] http://www.bbci.de/competition/iv/
    """
    for dataset_name in 'bci2a', 'bci2b':
      hparams = utils.read_config(dataset_name=dataset_name)
      batch_size = hparams['batch_size']
      csp = hparams['apply_csp']
      num_channels = hparams['num_csp'] if csp else hparams['num_eeg_channels']
      window_len = hparams['window_len']
      num_subject = 0  # subject number 1

      data = data_utils.Data(hparams)
      train_dataset = data.build_dataset(num_subject, training=True)
      val_dataset = data.build_dataset(num_subject, training=False)
      train_batch, _ = next(iter(train_dataset))
      assert train_batch.shape == (batch_size, num_channels, window_len, 1)
      val_batch, _ = next(iter(val_dataset))
      shape = val_batch.shape
      assert (shape[1], shape[2], shape[3]) == (num_channels, window_len, 1)


if __name__ == '__main__':
  tf.test.main()