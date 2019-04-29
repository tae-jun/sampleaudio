import argparse
import numpy as np
import tensorflow as tf
import lib.data as data
from sklearn import metrics
from lib.data.batch import create_datasets
from lib.data.config import *
from lib.initialization import AudioVarianceScaling
from lib.utils import mkpath


def main(args):
  args.model_path = mkpath(args.model_path)
  args.dataset = args.dataset or args.model_path.split('/')[-2].split('-')[1]  # extract dataset name from train_dir.

  if args.dataset == 'mtt':
    config = MTT_CONFIG
    classes = data.mtt.CLASSES
  elif args.dataset == 'scd':
    config = SCD_CONFIG
    classes = data.scd.CLASSES
  elif args.dataset == 'dcs':
    config = DCS_CONFIG
    classes = data.dcs.CLASSES
  else:
    raise Exception('Not implemented.')

  # Create training, validation, and test datasets.
  dataset_path = mkpath(args.data_dir, args.dataset, 'tfrecord')
  dataset_test = create_datasets(dataset_path, args.batch_size, args.num_readers, config, only_test=True)

  # Load the trained model.
  model = tf.keras.models.load_model(args.model_path,
                                     custom_objects={'AudioVarianceScaling': AudioVarianceScaling, 'tf': tf})

  # Evaluate
  evaluate(model, dataset_test, config, classes=classes)


def evaluate(model, dataset_test, config, classes=None):
  # Create the iterator.
  iterator = dataset_test.make_one_shot_iterator()
  seg, label = iterator.get_next()

  # Get dynamic shapes.
  seg_shape = tf.shape(seg)
  batch_size, num_segments, num_samples = seg_shape[0], seg_shape[1], seg_shape[2]
  num_classes = tf.shape(label)[1]

  seg = tf.reshape(seg, shape=(batch_size * num_segments, num_samples, 1))
  pred_segs = model(seg)  # predict all segments
  pred_segs = tf.reshape(pred_segs, shape=(batch_size, num_segments, num_classes))
  pred = tf.reduce_mean(pred_segs, axis=1)  # Average segments for each audio

  y_true, y_prob = [], []
  sess = tf.keras.backend.get_session()
  while True:
    try:
      label_batch, pred_batch = sess.run([label, pred], feed_dict={tf.keras.backend.learning_phase(): 0})
      y_true.append(label_batch)
      y_prob.append(pred_batch)
    except tf.errors.OutOfRangeError:
      break

  y_true, y_prob = np.concatenate(y_true), np.concatenate(y_prob)
  rocauc = metrics.roc_auc_score(y_true, y_prob, average='macro')
  prauc = metrics.average_precision_score(y_true, y_prob, average='macro')

  y_pred = (y_prob > config.threshold).astype(np.float32)
  acc = metrics.accuracy_score(y_true, y_pred)
  f1 = metrics.f1_score(y_true, y_pred, average='samples')

  if classes is not None:
    print(f'\n=> Individual scores of {len(classes)} classes')
    for i, cls in enumerate(classes):
      cls_rocauc = metrics.roc_auc_score(y_true[:, i], y_prob[:, i])
      cls_prauc = metrics.average_precision_score(y_true[:, i], y_prob[:, i])
      cls_acc = metrics.accuracy_score(y_true[:, i], y_pred[:, i])
      cls_f1 = metrics.f1_score(y_true[:, i], y_pred[:, i])
      print(f'[{i:2} {cls:30}] rocauc={cls_rocauc:.4f} prauc={cls_prauc:.4f} acc={cls_acc:.4f} f1={cls_f1:.4f}')
    print()

  print(f'=> Test scores: rocauc={rocauc:.6f}\tprauc={prauc:.6f}\tacc={acc:.6f}\tf1={f1:.6f}')
  return rocauc, prauc, acc, f1


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Evaluate a SampleCNN.')
  parser.add_argument('dataset', type=str, metavar='DATASET',
                      choices=['mtt', 'scd', 'dcs'], help='Dataset for training: {mtt|scd|dcs}')
  parser.add_argument('model_path', type=str, metavar='PATH', help='Path to the saved model.')
  parser.add_argument('--data-dir', type=str, default='./data', metavar='PATH')
  parser.add_argument('--batch-size', type=int, default=23, metavar='N', help='Mini-batch size.')
  parser.add_argument('--num-readers', type=int, default=8, metavar='N', help='Number of TFRecord readers.')

  args = parser.parse_args()

  main(args)
  print('\n=> Done.\n')
