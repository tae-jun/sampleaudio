import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from lib.utils import mkpath

CLASSES = ['Train horn', 'Air horn, truck horn', 'Car alarm', 'Reversing beeps', 'Ambulance (siren)',
           'Police car (siren)', 'Fire engine, fire truck (siren)', 'Civil defense siren', 'Screaming', 'Bicycle',
           'Skateboard', 'Car', 'Car passing by', 'Bus', 'Truck', 'Motorcycle', 'Train']

C2I = {c: i for i, c in enumerate(CLASSES)}

DIR_TRAIN = 'unbalanced_train_segments_training_set_audio_formatted_and_segmented_downloads'
DIR_TEST = 'unbalanced_train_segments_testing_set_audio_formatted_and_segmented_downloads'
DIR_EVAL = 'evaluation_set_formatted_audio_segments'


def make_dataset_info(dataset_dir, num_audios_per_shard):
  df_train = read_csv(mkpath(dataset_dir, 'raw/groundtruth_weak_label_training_set.csv'))
  df_test = read_csv(mkpath(dataset_dir, 'raw/groundtruth_weak_label_testing_set.csv'))
  df_eval = read_csv(mkpath(dataset_dir, 'raw/groundtruth_weak_label_evaluation_set.csv'))

  df_train['path'] = [mkpath(dataset_dir, f'raw/{DIR_TRAIN}/Y{f}') for f in df_train['file']]
  df_test['path'] = [mkpath(dataset_dir, f'raw/{DIR_TEST}/Y{f}') for f in df_test['file']]
  df_eval['path'] = [mkpath(dataset_dir, f'raw/{DIR_EVAL}/Y{f}') for f in df_eval['file']]

  df_train = pd.concat([df_train, df_test])

  # Split validation set.
  val_files = []
  for c in CLASSES:
    df_class = df_train[df_train['label'] == c]
    val_files += df_class.sample(frac=0.1, random_state=123)['file'].tolist()
  val_files = list(set(val_files))

  is_val = df_train['file'].isin(val_files)
  df_val = df_train[is_val].assign(split='val')
  df_train = df_train[~is_val].assign(split='train')
  df_eval = df_eval.assign(split='test')

  df = pd.concat([df_train, df_val, df_eval])

  # Encode labels.
  label = df.groupby('file')['label'].apply(list)
  label.iloc[:] = [encode(l) for l in label]
  label = label.to_frame().reset_index()
  df = df.drop_duplicates('file').drop('label', axis=1).merge(label, on='file')

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


def read_csv(path):
  df = pd.read_csv(path, delimiter='\t', names=['file', 'start', 'end', 'label'])
  return df


def encode(label):
  x = np.zeros(shape=len(CLASSES), dtype=np.float32)
  x[[C2I[l] for l in label]] = 1.
  return x
