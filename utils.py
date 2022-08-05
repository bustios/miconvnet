# coding=utf-8
import datetime
import json
import os

from absl import app
import numpy as np
import tensorflow as tf


def get_writer(log_dir, directory):
  log_dir = os.path.join(log_dir, directory)
  os.makedirs(log_dir, exist_ok=True)
  summary_writer = tf.summary.create_file_writer(log_dir)
  return summary_writer


def write_scalar(summary_writer, accumulator, num_step, name):
  """Log scalar in summary."""
  value = accumulator.result()
  with summary_writer.as_default(step=num_step):
    tf.summary.scalar(name, value)
  # Reset metrics
  accumulator.reset_states()
  return value


def write_histogram(summary_writer, model, num_step):
  """Log distributions of model variables."""
  with summary_writer.as_default(step=num_step):
    for variable in model.trainable_variables:
      tf.summary.histogram(variable.name, variable)


def read_config(model_name=None, dataset_name=None):
  """Read configuration file config.json.
  Arguments:
    model_name: If None, model_name's parameters of model_name specified in
      config.json will be loaded.
    dataset_name: If None, dataset specified in config.json will be loaded.
  Returns:
    A dictionary with setup parameters.
  """
  FILE_DIR = os.path.dirname(os.path.realpath(__file__))
  with open(os.path.join(FILE_DIR, 'config.json')) as json_file:
    config = json.load(json_file)

  if model_name is None:
    model_name = config['model']
  if dataset_name is None:
    dataset_name = config['dataset']
  config_ = {}
  config_['model'] = model_name
  config_['dataset'] = dataset_name
  config_['approach'] = config['approach']
  config_['log_dir'] = config['log_dir']
  config_['save_model'] = config['save_model']
  config_['write_summary'] = config['write_summary']
  config_['num_experiments'] = config['num_experiments']
  config_.update(config['hyperparams']['common'])
  config_.update(config['hyperparams'][model_name])
  config_.update(config['datasets'][dataset_name])
  return config_