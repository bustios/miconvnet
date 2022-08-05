# coding=utf-8
import itertools
import os

from absl import logging

import mne
import numpy as np
import scipy
import scipy.io
import tensorflow as tf


EVENT_DICT = {
  '276': 1,  # Idling EEG (eyes open)
  '277': 2,  # Idling EEG (eyes closed)
  '768': 3,  # Start of a trial
  '769': 4,  # Cue onset left (class 1)
  '770': 5,  # Cue onset right (class 2)
  '771': 6,  # Cue onset foot (class 3)
  '772': 7,  # Cue onset tongue (class 4)
  '783': 8,  # Cue unknown
  '1023': 9,  # Rejected trial
  '1072': 10,  # Eye movements
  '32766': 11  # Start of a new run
}
EVENTS_TO_READ = {'768': 3}
EOG_CHANNELS_2A = ('EOG-left', 'EOG-central', 'EOG-right')
EOG_CHANNELS_2B = ('EOG:ch01', 'EOG:ch02', 'EOG:ch03')


class Data:

  def __init__(self, config):
    """
    Arguments:
      config: dictionary with configuration parameters.
    """
    self.config = config

  def build_dataset(self, num_subject, training=True):
    """ Build one of the datasets of BCI Competition IV:
      - Dataset 2a (9 subjects) has a total of 2592 trials for training.
      - Dataset 2b (9 subjects) has a total of 3680 trials for training.

    Arguments:
      num_subject: number of the subject, from 0 to 8.
      training: boolean indicating the dataset type: training or validation.
    Returns:
      A `tf.data.Dataset` object.
    """
    files = self._select_filenames(num_subject, training)
    epochs_array, labels_array = self._read_data_from(files, training)
    dataset = self._build_dataset_from_epochs(epochs_array, labels_array, 
        training)
    logging.info(f'Sessions loaded: {files}')
    logging.info(f'Dataset shape: {epochs_array.shape}')
    return dataset
  
  def _build_dataset_from_epochs(self, epochs_array, labels_array, training):
    num_channels = epochs_array.shape[1]  # num_eeg_channels or num_csp
    win_len = self.config['window_len']
    window_stride = self.config['window_stride']

    def epochs_gen():
      num_samples = epochs_array.shape[2]
      for start in range(0, num_samples - win_len + 1, window_stride):
        batch = epochs_array[:, :, start:start + win_len, :]
        for epoch, label in zip(batch, labels_array):
          yield epoch, label

    output_signature=(
        tf.TensorSpec(shape=(num_channels, win_len, 1), dtype=tf.float64),
        tf.TensorSpec(shape=(), dtype=tf.int32))

    ds = tf.data.Dataset.from_generator(
        epochs_gen,
        output_signature=output_signature
    )

    if training:
      batch_size = self.config['batch_size']
      seed = self.config['seed']
      # BCI Competition IV:
      # - Dataset 2a (9 subjects) has a total of 2592 trials for training
      # - Dataset 2b (9 subjects) has a total of 3680 trials for training.
      ds = ds.shuffle(3680, seed, reshuffle_each_iteration=True)
      ds = ds.batch(batch_size)
      ds = ds.repeat(-1)
    else:
      ds = ds.batch(epochs_array.shape[0])
    
    return ds

  def read_all_dataset(self):
    """Read all epochs and labels from training and validation sessions
    of the dataset specified in `config.json`.

    Returns:
      A tuple of numpy.array (epochs_train, labels_train, epochs_val,
      labels_val).
    """
    epochs_train, labels_train, epochs_val, labels_val = [], [], [], []

    for num_subject in range(9):
      files = self._select_filenames(num_subject, training=True)
      X_train, Y_train = self._read_data_from(files, training=True)
      epochs_train.append(X_train)
      labels_train.append(Y_train)
      files = self._select_filenames(num_subject, training=False)
      X_val, Y_val = self._read_data_from(files, training=False)
      epochs_val.append(X_val)
      labels_val.append(Y_val)

    epochs_train = np.concatenate(epochs_train)
    labels_train = np.concatenate(labels_train)
    epochs_val = np.concatenate(epochs_val)
    labels_val = np.concatenate(labels_val)
    return epochs_train, labels_train, epochs_val, labels_val

  def _read_data_from(self, gdf_files, training=True):
    """Read a list of GDF files and create a `tf.data.Dataset` from them.

    Arguments:
      gdf_files: list of GDF files to read.
      training: boolean indicating whether the dataset is for training.
    Returns:
      A tuple of numpy.array (epochs_array, labels_array).
    """
    dataset_dir = self.config['dataset_dir']
    labels_dir = self.config['labels_dir']
    new_sample_rate = self.config['new_sample_rate']
    dataset_bci2a = self.config['dataset'] == 'bci2a'
    tmin = self.config['tmin']  # segment start
    tmax = self.config['tmax']  # segment end
    fmin = self.config['fmin']  # cutoff frequency for highpass filter

    eog_channels = EOG_CHANNELS_2A if dataset_bci2a else EOG_CHANNELS_2B
    epochs_list = []
    labels_list = []

    for gdf_filename in gdf_files:
      gdf_filepath = os.path.join(dataset_dir, gdf_filename)
      epochs = self._read_epochs(gdf_filepath, eog_channels, tmin, tmax)

      if dataset_bci2a and self.config['num_eeg_channels'] == 3:
        epochs = epochs.pick_channels(['EEG-C3', 'EEG-Cz', 'EEG-C4'])
      sample_rate = epochs.info['sfreq']
      if new_sample_rate is not None and new_sample_rate != sample_rate:
        epochs = epochs.resample(new_sample_rate)
      if fmin > 0:
        epochs.filter(l_freq=fmin, h_freq=None)
      # `array_` has shape: 
      # [num_epochs, num_eeg_channels, (tmax - tmin) * new_samp_freq + 1].
      array_ = epochs.get_data() # Range of values: [-1e-04, 1e-04] approx.
      epochs_list.append(array_)

      labels_file = gdf_filename.split('.')[0] + '.mat'
      labels_filepath = os.path.join(labels_dir, labels_file)
      labels_array = self._read_labels(labels_filepath)
      labels_list.append(labels_array)

    epochs_array = np.concatenate(epochs_list)
    epochs_array = epochs_array[:, :, :-1]

    if self.config['standardize_data']:
      std = 1.025681e-05 if dataset_bci2a else 4.045408e-06
      epochs_array /= std

    labels_array = np.concatenate(labels_list)
    labels_array = np.squeeze(labels_array)  # shape: [num_epochs,]

    if self.config['apply_csp']:
      if training:
        self.csp = mne.decoding.CSP(n_components=self.config['num_csp'], 
            transform_into='csp_space', norm_trace=False)
        epochs_array = self.csp.fit_transform(epochs_array, labels_array)
      else:
        epochs_array = self.csp.transform(epochs_array)

    # `epochs_array_` has shape:
    # [num_epochs, num_csp, (tmax - tmin) * new_samp_freq, 1].
    epochs_array_ = epochs_array[:, :, :, np.newaxis]
    return epochs_array_, labels_array

  def _read_epochs(self, gdf_path, eog_channels, tmin=-2., tmax=8.5):
    """Read a GDF file and a MAT file, and return a mne.Epochs object and
    a numpy array of labels.
    
    Arguments:
      gdf_path: path to the GDF file to be read.
      eog_channels: list or tuple with names of EOG channels.
      tmin: start time (starting from event).
      tmax: end time (starting from event).

    Returns:
      epochs: instance of mne.Epochs.
    """
    raw_eeg = mne.io.read_raw_gdf(gdf_path, eog=eog_channels)
    raw_eeg = raw_eeg.pick_types(eeg=True)
    events_array, _ = mne.events_from_annotations(raw_eeg, event_id=EVENT_DICT)
    epochs = mne.Epochs(raw_eeg, events_array, event_id=EVENTS_TO_READ, 
        tmin=tmin, tmax=tmax, baseline=None, preload=True)

    return epochs

  def _read_labels(self, labels_filepath):
    """Read a MAT file, and return a numpy array of labels.

    Arguments:
      labels_filepath: path to the MAT file containing the trials' labels.

    Returns:
      labels: numpy array with epochs' labels.
    """
    labels_dict = scipy.io.loadmat(labels_filepath)
    # labels_array shape: [num_epochs, 1], values start at 1.
    labels_array = labels_dict['classlabel'] - 1
    return labels_array.astype(np.int32)

  def _select_filenames(self, num_subject, training=True):
    if training:
      # List of lists with all filenames of training sessions from all subjects.
      train_sessions = self.config['train_sessions']
      # A string, one of 'intra_subject', 'inter_subject', 'all'
      approach = self.config['approach'] 
      files = train_sessions.copy()

      if approach == 'intra_subject':
        files = files[num_subject]
      elif approach in ['inter_subject', 'all']:
        if approach == 'inter_subject':
          del files[num_subject]
        files = list(itertools.chain(*files))
      else:
        raise ValueError('Invalid name for approach type.')

    else:
      val_sessions = self.config['val_sessions']
      files = val_sessions[num_subject]

    return files