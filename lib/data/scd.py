import numpy as np
import pandas as pd
import tensorflow as tf
from glob import glob
from sklearn.utils import shuffle
from lib.utils import mkpath

CLASSES = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go', 'happy',
           'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six',
           'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']
IDX2NAME = {i: name for i, name in enumerate(CLASSES)}
NAME2IDX = {name: i for i, name in enumerate(CLASSES)}


def load_audio_paths(dataset_dir):
  audio_paths = glob(mkpath(dataset_dir, 'raw/*/*.wav'))
  noise_paths = glob(mkpath(dataset_dir, 'raw/_background_noise_/*.wav'))

  with open(mkpath(dataset_dir, 'raw/validation_list.txt')) as f:
    val_paths = f.read().splitlines()
    val_paths = [mkpath(dataset_dir, 'raw', path) for path in val_paths]

  with open(mkpath(dataset_dir, 'raw/testing_list.txt')) as f:
    test_paths = f.read().splitlines()
    test_paths = [mkpath(dataset_dir, 'raw', path) for path in test_paths]

  # Remove validation, test set, and noises from the training set.
  train_paths = list(set(audio_paths) - set(val_paths) - set(test_paths) - set(noise_paths))

  # Sort paths.
  train_paths.sort(), val_paths.sort(), test_paths.sort()

  return train_paths, val_paths, test_paths


def make_dataset_info(dataset_dir, num_audios_per_shard):
  train_paths, val_paths, test_paths = load_audio_paths(dataset_dir)

  paths = train_paths + val_paths + test_paths
  ids = ['/'.join(p.split('/')[-2:]) for p in paths]
  labels = [tf.keras.utils.to_categorical(NAME2IDX[id.split('/')[0]], num_classes=len(CLASSES)) for id in ids]
  splits = ['train'] * len(train_paths) + ['val'] * len(val_paths) + ['test'] * len(test_paths)

  df = pd.DataFrame({'id': ids, 'label': labels, 'split': splits, 'path': paths})

  # Shuffle and shard.
  df = shuffle(df, random_state=123)
  for split in ['train', 'val', 'test']:
    num_audios = sum(df['split'] == split)
    num_shards = num_audios // num_audios_per_shard
    num_remainders = num_audios % num_audios_per_shard

    shards = np.tile(np.arange(num_shards), num_audios_per_shard)
    shards = np.concatenate([shards, np.arange(num_remainders) % num_shards])
    shards = np.random.permutation(shards)

    df.loc[df['split'] == split, 'shard'] = shards

  df['shard'] = df['shard'].astype(int)

  return df
