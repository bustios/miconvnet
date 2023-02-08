# coding=utf-8
import datetime
import os
import pprint
import warnings

from absl import app
from absl import logging
from tqdm import trange
import mne
import tensorflow as tf

from . import data as data_utils
from . import model as model_utils
from . import utils


mne.set_log_level(verbose='CRITICAL')
warnings.filterwarnings('ignore', category=DeprecationWarning)


def train_step(batch, labels, model, loss_fn, optimizer, accum_loss,
    accum_accuracy):
  """Perform a single training step."""
  # Get the prediction of the models and compute the loss.
  with tf.GradientTape() as tape:
    logits = model(batch, training=True)
    loss_value = loss_fn(labels, logits)

  gradients = tape.gradient(loss_value, model.trainable_weights)
  optimizer.apply_gradients(zip(gradients, model.trainable_weights))
  accum_loss(loss_value)
  accum_accuracy(labels, tf.squeeze(logits))


def val_step(batch, labels, model, loss_fn, accum_loss, accum_accuracy):
  """ Perform a sinigle validation step."""
  logits = model(batch, training=False)
  loss_value = loss_fn(labels, logits)
  accum_loss(loss_value)
  accum_accuracy(labels, tf.squeeze(logits))


def write_scalars(summary_writer, loss_accum, acc_accum, num_step):
  """Log the training and validation losses, and training and validation
  accuracies."""
  loss = loss_accum.result()
  acc = acc_accum.result()
  with summary_writer.as_default(step=num_step):
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', acc)
  # Reset metrics
  loss_accum.reset_states()
  acc_accum.reset_states()
  return loss, acc


def train_val(config, subject, train_dataset, val_dataset, enable_checkpoint):
  """Performs a single training and validation cycle."""

  base_learning_rate = config['learning_rate']
  num_train_steps = config['num_train_steps']
  patience = config['early_stopping_patience']
  write_summary = config['write_summary']
  warmup_steps = config['warmup_steps']
  current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
  log_dir = os.path.join(config['log_dir'], str(subject), current_time)
  if write_summary:
    train_summary_writer = utils.get_writer(log_dir, 'train')
    val_summary_writer = utils.get_writer(log_dir, 'val')

  # Loss function and metrics
  if config['num_classes'] == 2:
    loss_fn = tf.losses.BinaryCrossentropy(from_logits=True)
    train_acc = tf.metrics.BinaryAccuracy(threshold=0.0)
    val_acc = tf.metrics.BinaryAccuracy(threshold=0.0)
  else:
    loss_fn = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    train_acc = tf.metrics.SparseCategoricalAccuracy()
    val_acc = tf.metrics.SparseCategoricalAccuracy()

  train_loss = tf.metrics.Mean('train_loss', dtype=tf.float32)
  val_loss = tf.metrics.Mean('val_loss', dtype=tf.float32)

  # Model and optimizer
  model = model_utils.build_model(config)
  optimizer = tf.optimizers.Adam(base_learning_rate, epsilon=1e-07)
  global_step = tf.Variable(
      0, trainable=False, name='global_step', dtype=tf.int64)

  # Prepare checkpoint manager
  ckpt = tf.train.Checkpoint(
      network=model, optimizer=optimizer, global_step=global_step)
  ckpt_manager = tf.train.CheckpointManager(
      checkpoint=ckpt, directory=log_dir, max_to_keep=3)
  if enable_checkpoint and ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
    logging.info(f'Model restored from {ckpt_manager.latest_checkpoint}')
  else:
    logging.info('Initializing from scratch.')

  train_dataset_iterator = iter(train_dataset)
  progress_bar = trange(num_train_steps)
  template = 'Loss: {:.4f}, Accuracy: {:.4f}, ' \
      'Val. Loss: {:.4f}, Val. Accuracy: {:.4f}'

  count_patience = 0
  max_val_acc = 0.
  num_iters_x_epoch = 20 if config['dataset'] == 'bci2a' else 25

  # Training and validation
  for _ in progress_bar:
    # Learning rate warm-up.
    if global_step < warmup_steps:
      optimizer.lr = base_learning_rate * tf.cast(
          global_step, tf.float32) / tf.cast(warmup_steps, tf.float32)

    epochs, labels = next(train_dataset_iterator)
    train_step(epochs, labels, model, loss_fn, optimizer, train_loss, train_acc)

    # Update the global step. We update it before logging the loss and saving
    # the model so that the last checkpoint is saved at the last iteration.
    global_step.assign_add(1)

    if global_step % num_iters_x_epoch == 0:
      for val_epochs, val_labels in val_dataset:
        val_step(val_epochs, val_labels, model, loss_fn, val_loss, val_acc)

      if write_summary:
        step = global_step // num_iters_x_epoch
        loss_acc = [
          utils.write_scalar(train_summary_writer, train_loss, step, 'loss'),
          utils.write_scalar(train_summary_writer, train_acc, step, 'acc'),
          utils.write_scalar(val_summary_writer, val_loss, step, 'loss'),
          utils.write_scalar(val_summary_writer, val_acc, step, 'acc')
        ]
        # utils.write_histogram(train_summary_writer, model, step)
        msg = template.format(*loss_acc)
        progress_bar.set_postfix_str(msg)
        curr_val_acc = loss_acc[-1]
      else:
        curr_val_acc = val_acc.result()
        val_acc.reset_states()

      # Early stopping.
      if curr_val_acc >= max_val_acc:
        max_val_acc = curr_val_acc
        count_patience = 0
        if enable_checkpoint:
          ckpt_manager.save(checkpoint_number=int(max_val_acc * 10000))
      else:
        count_patience += num_iters_x_epoch
        if count_patience >= patience:
          break

  logging.info(f'Max. val. accuracy: {max_val_acc:.4f}')
  # model.summary()

def main(argv):
  del argv

  config = utils.read_config()
  pprint.pprint(config)
  num_experiments = config['num_experiments']
  enable_checkpoint = config['save_model']
  data = data_utils.Data(config)
  num_subjects = 9

  for num_subject in range(num_subjects):
    train_dataset = data.build_dataset(num_subject, training=True)
    val_dataset = data.build_dataset(num_subject, training=False)

    subject = num_subject + 1
    for i in range(num_experiments):
      train_val(config, subject, train_dataset, val_dataset, enable_checkpoint)


if __name__ == '__main__':
  app.run(main)