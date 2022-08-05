# coding=utf-8
import math

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import constraints


class SincFilter(layers.Layer):
  """SincFilter."""

  def __init__(self, low_freqs, kernel_size, sample_rate, bandwidth=4, 
      min_freq=1., padding='SAME'):
    super().__init__(name='sinc_filter_layer')
    if kernel_size % 2 == 0:
      raise ValueError('Kernel size must be odd.')
    
    self.num_filters = len(low_freqs)
    self.kernel_size = kernel_size
    self.sample_rate = sample_rate
    self.min_freq = min_freq
    self.padding = padding
    self.ones = tf.ones((1, 1, 1, self.num_filters))
    window = tf.signal.hamming_window(kernel_size, periodic=False)
    # `self.window` has shape: [kernel_size // 2, 1].
    self.window = tf.expand_dims(window[:kernel_size // 2], axis=-1)
    # `self.n_pi` has shape: [kernel_size // 2, 1].
    self.n_pi = tf.range(-(kernel_size // 2), 0, dtype=tf.float32) / sample_rate
    self.n_pi *= 2 * math.pi
    self.n_pi = tf.expand_dims(self.n_pi, axis=-1)
    # `bandwidths` has shape: [1, num_filters].
    bandwidths = tf.ones((1, self.num_filters)) * bandwidth
    self.bandwidths = tf.Variable(bandwidths, name='bandwidths')
    # `low_freqs` has shape: [1, num_filters].
    self.low_freqs = tf.Variable([low_freqs], name='low_freqs', 
        dtype=tf.float32)

  def build_sinc_filters(self):
    # `low_freqs` has shape: [1, num_filters].
    low_freqs = self.min_freq + tf.math.abs(self.low_freqs)
    # `high_freqs` has shape: [1, num_filters].
    high_freqs = tf.clip_by_value(low_freqs + tf.math.abs(self.bandwidths),
        self.min_freq, self.sample_rate / 2.)
    bandwidths = high_freqs - low_freqs

    low = self.n_pi * low_freqs  # size [kernel_size // 2, num_filters].
    high = self.n_pi * high_freqs  # size [kernel_size // 2, num_filters].

    # `filters_left` has shape: [kernel_size // 2, num_filters].
    filters_left = (tf.math.sin(high) - tf.math.sin(low)) / (self.n_pi / 2.)
    filters_left *= self.window
    filters_left /= 2. * bandwidths

    # `filters_left` has shape: [1, kernel_size // 2, 1, num_filters].
    filters_left = filters_left[tf.newaxis, :, tf.newaxis, :]
    # filters_left = tf.ensure_shape(filters_left, 
    #     shape=(1, self.kernel_size // 2, 1, self.num_filters))
    filters_right = tf.experimental.numpy.flip(filters_left, axis=1)

    # `filters` has shape: [1, kernel_size, 1, num_filters].
    filters = tf.concat([filters_left, self.ones, filters_right], axis=1)
    filters = filters / tf.math.reduce_std(filters)
    return filters

  def call(self, inputs):
    filters = self.build_sinc_filters()
    # `inputs` has shape: [num_epochs, num_channels, num_samples, 1].
    # `filtered` has shape: [num_epochs, num_channels, num_samples, num_filters]
    filtered = tf.nn.convolution(inputs, filters, padding=self.padding)
    return filtered


class EEGNet(tf.keras.Model):
  """Implementation of EEGNet
  http://iopscience.iop.org/article/10.1088/1741-2552/aace8c/meta
  
  Note that this implements the newest version of EEGNet and NOT the earlier
  version (version v1 and v2 on arxiv).
  """
  
  def __init__(
      self,
      num_classes, 
      num_channels, 
      num_temp_filters=8,  # F1=8
      temp_filter_size=32, 
      num_spatial_filters_x_temp=2,  # D=2
      num_sep_convs=16,  # F2=F1*D
      dropout_type='Dropout',
      dropout_rate=0.5,
      max_norm=0.25
    ):
    """
    Arguments:
      num_classes: Number of classes.
      num_channels: Number of channels in EEG epoch.
      num_temp_filters: Number of temporal filters.
      temp_filter_size: Size of temporal convolution in the first layer.
      num_spatial_filters_x_temp: Number of spatial filters per temporal filter.
      num_sep_convs: Number of pointwise filters. Default: 
          num_sep_convs = num_temp_filters * num_spatial_filters_x_temp.
      dropout_type: Either SpatialDropout2D or Dropout, passed as a string.
      dropout_rate: Dropout rate.
      max_norm: Maximum norm weight constraint.
    """
    super().__init__()
    if dropout_type == 'SpatialDropout2D':
      dropout = layers.SpatialDropout2D
    elif dropout_type == 'Dropout':
      dropout = layers.Dropout
    else:
      raise ValueError('dropout_type must be one of `SpatialDropout2D` or '
          f'`Dropout`, passed as a string. `{dropout_type}` was given.')

    self.block_1 = tf.keras.Sequential([
        layers.Conv2D(num_temp_filters, kernel_size=(1, temp_filter_size), 
            padding='SAME', use_bias=False, name='temp_filter'),
        layers.BatchNormalization(name='first_batchnorm'),
        layers.DepthwiseConv2D(kernel_size=(num_channels, 1), 
            depth_multiplier=num_spatial_filters_x_temp, use_bias=False, 
            depthwise_constraint=constraints.max_norm(1.), 
            name='spatial_filters'),
        layers.BatchNormalization(name='second_batchnorm'),
        layers.ELU(),
        layers.AveragePooling2D(pool_size=(1, 4)),  # 128 Hz
        # layers.AveragePooling2D(pool_size=(1, 8)),  # 250 Hz
        dropout(dropout_rate)
    ], name='block_1')
    
    self.block_2 = tf.keras.Sequential([
        layers.SeparableConv2D(num_sep_convs, kernel_size=(1, 16),  # 128 Hz
        # layers.SeparableConv2D(num_sep_convs, kernel_size=(1, 33),  # 250 Hz
            padding='SAME', use_bias=False),
        layers.BatchNormalization(name='third_batchnorm'),
        layers.ELU(),
        layers.AveragePooling2D(pool_size=(1, 8)),  # 128 Hz
        # layers.AveragePooling2D(pool_size=(1, 16)),  # 250 Hz
        dropout(dropout_rate),
        layers.Flatten(),
        layers.Dense(num_classes, name='dense',
            kernel_constraint=constraints.max_norm(max_norm))
    ], name='block_2')
    
  def call(self, epochs):
    """
    Arguments:
      epochs: Batch of epochs. Shape [num_epochs, num_channels, num_samples, 1].
    """
    # `x` has shape: 
    # [num_epochs, 1, num_samples / 4, 
    #  num_temp_filters * num_spatial_filters_x_temp].
    x = self.block_1(epochs)
    logits = self.block_2(x)
    return logits


class ShallowConvNet(tf.keras.Model):
  """Implementation of ShallowConvNet [1] adapted to work with EEG sampled at
  128 Hz.

  [1] Schirrmeister, R. et. al. (2017) Deep learning with convolutional neural 
  networks for EEG decoding and visualization.
  """

  def __init__(self, num_classes, num_channels, num_temp_filters, 
      temp_filter_size, dropout_rate=0.5):
    super().__init__()
    self.model = tf.keras.Sequential([
        layers.Conv2D(num_temp_filters, kernel_size=(1, temp_filter_size),
            kernel_constraint=constraints.max_norm(2., axis=[0, 1, 2]),
            name='first_conv2d'),
        layers.Conv2D(num_temp_filters, kernel_size=(num_channels, 1), 
            kernel_constraint=constraints.max_norm(2., axis=[0, 1, 2]),
            use_bias=False, name='second_conv2d'),
        layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='batchnorm'),
        layers.Activation(tf.math.square),
        layers.AveragePooling2D(pool_size=(1, 35), strides=(1, 7)),  # 128 Hz
        # layers.AveragePooling2D(pool_size=(1, 75), strides=(1, 15)),  # 250 Hz
        layers.Activation(tf.math.log),
        layers.Flatten(),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, kernel_constraint=constraints.max_norm(0.5),
            name='dense')
    ], name='model')

  def call(self, epochs):
    # `epochs` has shape [num_epochs, num_channels, num_samples, 1].
    logits = self.model(epochs)
    return logits


class SincShallowNet(tf.keras.Model):
  """Implementation of Sinc-ShallowNet [1] adapted to work with EEG sampled at
  128 Hz, for that, average pooling filter size and strides were divided by 2.

  [1] Borra, D. et. al. (2020) Interpretable and lightweight convolutional 
  neural network for EEG decoding: Application to movement execution and 
  imagination.
  """

  def __init__(self, num_classes, num_channels, num_temp_filters, 
      temp_filter_size, sample_rate, num_spatial_filters_x_temp):
    super().__init__()
    self.block_1 = tf.keras.Sequential([
        build_sinc_layer(num_temp_filters, temp_filter_size, sample_rate, 
            first_freq=5, freq_stride=1, padding='VALID'),
        layers.BatchNormalization(name='block_1_batchnorm'),
        layers.DepthwiseConv2D(kernel_size=(num_channels, 1), 
            depth_multiplier=num_spatial_filters_x_temp, use_bias=False, 
            name='spatial_filter')
    ], name='block_1')

    self.block_2 = tf.keras.Sequential([
        layers.BatchNormalization(name='block_2_batchnorm'),
        layers.ELU(),
        layers.AveragePooling2D(pool_size=(1, 55), strides=(1, 12)),  # 128 Hz
        # layers.AveragePooling2D(pool_size=(1, 109), strides=(1, 23)), # 250 Hz
        layers.Dropout(0.5)
    ], name='block_2')

    self.block_3 = tf.keras.Sequential([
        layers.Flatten(),
        layers.Dense(num_classes, name='dense')
    ], name='block_3')

  def call(self, epochs):
    x = self.block_1(epochs)
    x = self.block_2(x)
    logits = self.block_3(x)
    return logits


class MIConvNet(tf.keras.Model):

  def __init__(self, num_classes, num_channels, num_temp_filters, 
      temp_filter_size, sample_rate):
    super().__init__()
    self.temporal_filter_1 = build_sinc_layer(num_temp_filters // 2, 
        temp_filter_size, sample_rate, first_freq=6, freq_stride=2, 
        padding='VALID')
    self.temporal_filter_2 = layers.Conv2D(num_temp_filters // 2, 
        kernel_size=(1, temp_filter_size), padding='VALID', use_bias=False, 
        kernel_initializer='lecun_uniform', name='conv_temporal_filter')

    self.block_2 = tf.keras.Sequential([
        layers.BatchNormalization(name='block_2_batchnorm_1'),
        layers.ELU(),
        layers.AveragePooling2D(pool_size=(1, 64), strides=(1, 32)),
        # layers.AveragePooling2D(pool_size=(1, 128), strides=(1, 64)), # 250 Hz
        layers.BatchNormalization(name='block_2_batchnorm_2'),
        layers.DepthwiseConv2D(kernel_size=(1, 6), use_bias=False,
            kernel_initializer='lecun_normal'),
        layers.Flatten()
    ], name='block_2')

    self.flatten = layers.Flatten()
    self.dropout = layers.Dropout(0.5)
    self.dense = layers.Dense(num_classes, name='dense', 
        kernel_regularizer=tf.keras.regularizers.L1(0.01))

  def call(self, epochs):
    # `epochs` has shape: [num_epochs, num_csp, num_samples, 1]
    # `x` has shape: [num_epochs, num_csp, num_samples, num_temp_filters].
    x1 = self.temporal_filter_1(epochs)
    x2 = self.temporal_filter_2(epochs)

    x = tf.concat([x1, x2], axis=-1)

    x1 = tf.reduce_sum(x ** 2, axis=2)
    x1 = tf.math.log(x1) - tf.math.log(epochs.shape[2] * 9.)
    x1 = self.flatten(x1)

    x2 = self.block_2(x)

    x = tf.concat([x1, x2], axis=-1)
    x = self.dropout(x)
    logits = self.dense(x)
    return logits


def build_sinc_layer(num_filters=8, filter_size=33, sample_rate=128, 
    first_freq=6, freq_stride=4, bandwidth=4, padding='SAME'):
  low_freqs = [first_freq]
  for _ in range(num_filters - 1):
    low_freqs.append(low_freqs[-1] + freq_stride)

  return SincFilter(low_freqs, filter_size, sample_rate, bandwidth, 
      padding=padding)


def build_model(hparams):
  """Build keras model."""
  model_name = hparams['model']
  csp = hparams['apply_csp']
  num_channels = hparams['num_csp'] if csp else hparams['num_eeg_channels']
  num_samples = hparams['window_len']
  num_temp_filters = hparams['num_temp_filters']
  temp_filter_size = hparams['temp_filter_size']
  num_classes = hparams['num_classes']
  if num_classes == 2:
    num_classes = 1

  if model_name == 'MIConvNet':
    new_sample_rate = hparams['new_sample_rate']
    model = MIConvNet(num_classes, num_channels, num_temp_filters, 
        temp_filter_size, new_sample_rate)
  
  elif model_name == 'SincShallowNet':
    new_sample_rate = hparams['new_sample_rate']
    num_spatial_filters_x_temp = hparams['num_spatial_filters_x_temp']
    model = SincShallowNet(num_classes, num_channels, num_temp_filters, 
        temp_filter_size, new_sample_rate, num_spatial_filters_x_temp)

  elif model_name == 'EEGNet':
    dropout_rate = hparams['dropout_rate']
    num_spatial_filters_x_temp = hparams['num_spatial_filters_x_temp']
    num_sep_convs = hparams['num_separable_convs']
    dropout_type = hparams['dropout_type']
    model = EEGNet(num_classes, num_channels, num_temp_filters, 
        temp_filter_size, num_spatial_filters_x_temp, num_sep_convs, 
        dropout_type, dropout_rate)

  elif model_name == 'ShallowConvNet':
    dropout_rate = hparams['dropout_rate']
    model = ShallowConvNet(num_classes, num_channels, num_temp_filters, 
        temp_filter_size, dropout_rate)

  else:
    raise ValueError('Invalid name for model type.')

  return model