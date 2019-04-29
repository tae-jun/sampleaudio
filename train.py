import argparse
import math
import os
import tensorflow as tf
import tensorflow.keras.backend as K
from datetime import datetime
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from lib.model import SampleCNN
from lib.model_config import ModelConfig
from lib.data.batch import create_datasets
from lib.data.config import *
from lib.initialization import AudioVarianceScaling
from lib.utils import mkpath
from eval import evaluate


def main(args):
  print(f'=> Dataset: {args.dataset}')
  if args.dataset == 'mtt':
    config = MTT_CONFIG
  elif args.dataset == 'scd':
    config = SCD_CONFIG
  elif args.dataset == 'dcs':
    config = DCS_CONFIG
  else:
    raise Exception(f'Not implemented dataset: {args.dataset}')

  dataset_path = mkpath(args.data_dir, args.dataset)
  tfrecord_path = f'{dataset_path}/tfrecord'

  # Configure the model.
  model_config = ModelConfig(block=args.block, amplifying_ratio=args.amplifying_ratio, multi=args.multi,
                             num_blocks=config.num_blocks, dropout=args.dropout, activation=config.activation,
                             num_classes=config.num_classes)

  # Set the training directory.
  args.train_dir = mkpath(args.log_dir, datetime.now().strftime('%Y%m%d_%H%M%S') + f'-{args.dataset}')
  if args.name is None:
    args.name = model_config.get_signature()
  args.train_dir += '-' + args.name
  os.makedirs(args.train_dir, exist_ok=False)
  print('=> Training directory: ' + args.train_dir)

  # Create training, validation, and test datasets.
  dataset_train, dataset_val, dataset_test = create_datasets(tfrecord_path, args.batch_size, args.num_readers, config)

  model = SampleCNN(model_config)
  model_config.print_summary()

  num_params = int(sum([K.count_params(p) for p in set(model.trainable_weights)]))
  print(f'=> #params: {num_params:,}')

  for stage in range(args.num_stages):
    print(f'=> Stage {stage}')
    # Set the learning rate of current stage
    lr = args.lr * (args.lr_decay ** stage)
    # Train the network.
    train(model, lr, dataset_train, dataset_val, config, args)
    # Load the best model.
    model = tf.keras.models.load_model(f'{args.train_dir}/best.h5',
                                       custom_objects={'AudioVarianceScaling': AudioVarianceScaling, 'tf': tf})
    # Evaluate.
    rocauc, prauc, acc, f1 = evaluate(model, dataset_test, config)

  # Change the file name of the best checkpoint with the scores.
  os.rename(f'{args.train_dir}/best.h5', f'{args.train_dir}/final-auc_{rocauc:.6f}-acc_{acc:.6f}-f1_{f1:.6f}.h5')
  # Report the final scores.
  print(f'=> FINAL SCORES [{args.dataset}] {args.name}: '
        f'rocauc={rocauc:.6f}, acc={acc:.6f}, f1={f1:.6f}, prauc={prauc:.6f}')

  model_config.print_summary()

  return rocauc, prauc, acc, f1


def train(model, lr, dataset_train, dataset_val, config, args):
  # Define a optimizer and compile the model.
  optimizer = tf.keras.optimizers.SGD(lr=lr, momentum=args.momentum, decay=1e-6, nesterov=True)
  model.compile(optimizer, loss=config.loss, metrics=config.metrics)

  # Setup callbacks.
  early_stopping = EarlyStopping(monitor='val_loss', patience=args.patience)
  checkpointer_best = ModelCheckpoint(f'{args.train_dir}/best.h5', monitor='val_loss', save_best_only=True)

  # Train!
  steps_train = int(math.ceil(config.num_train_segs / args.batch_size))
  steps_val = int(math.ceil(config.num_val_segs / args.batch_size))
  model.fit(dataset_train, epochs=100, steps_per_epoch=steps_train,
            validation_data=dataset_val, validation_steps=steps_val,
            callbacks=[early_stopping, checkpointer_best])


def parse_args():
  parser = argparse.ArgumentParser(description='Train a SampleCNN.')
  parser.add_argument('dataset', type=str, metavar='DATASET',
                      choices=['mtt', 'scd', 'dcs'], help='Dataset for training: {mtt|scd|dcs}')
  parser.add_argument('name', type=str, metavar='NAME', nargs='?', help='Name of log directory.')
  parser.add_argument('--data-dir', type=str, default='./data', metavar='PATH')
  parser.add_argument('--log-dir', type=str, default='./log', metavar='PATH',
                      help='Directory where to write event logs and models.')

  parser.add_argument('--block', type=str, default='se', choices=['basic', 'se', 'res1', 'res2', 'rese1', 'rese2'],
                      help='Convolutional block to build a model (default: se, options: basic/se/res1/res2/rese1/rese2).')
  parser.add_argument('--amplifying-ratio', type=float, default=0.125, metavar='N')
  parser.add_argument('--multi', action='store_true', help='Use multi-level feature aggregation.')

  parser.add_argument('--batch-size', type=int, default=23, metavar='N', help='Mini-batch size.')
  parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='Momentum for SGD.')
  parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='Learning rate.')
  parser.add_argument('--lr-decay', type=float, default=0.2, metavar='DC', help='Learning rate decay rate.')

  parser.add_argument('--dropout', type=float, default=0.5, metavar='DO', help='Dropout rate.')
  parser.add_argument('--weight-decay', type=float, default=0., metavar='WD', help='Weight decay.')

  parser.add_argument('--num-stages', type=int, default=5, metavar='N', help='Number of stages to train.')
  parser.add_argument('--patience', type=int, default=2, metavar='N', help='Stop training stage after #patiences.')

  parser.add_argument('--num-readers', type=int, default=8, metavar='N', help='Number of TFRecord readers.')

  return parser.parse_args()


if __name__ == '__main__':
  args = parse_args()

  main(args)

  print('\n=> Done.\n')
