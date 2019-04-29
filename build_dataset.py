import os
import argparse
import tensorflow as tf
import lib.data as data
from multiprocessing import Pool
from lib.utils import mkpath


def main(args):
  dataset_dir = mkpath(args.data_dir, args.dataset)

  if args.dataset == 'mtt':
    config = data.config.MTT_CONFIG
    df = data.mtt.make_dataset_info(dataset_dir, config.num_audios_per_shard)
  elif args.dataset == 'scd':
    config = data.config.SCD_CONFIG
    df = data.scd.make_dataset_info(dataset_dir, config.num_audios_per_shard)
  elif args.dataset == 'dcs':
    config = data.config.DCS_CONFIG
    df = data.dcs.make_dataset_info(dataset_dir, config.num_audios_per_shard)
  else:
    raise Exception('Not implemented dataset: ' + args.dataset)

  process_dataset(dataset_dir, config, df)


def process_dataset(dataset_dir, config, df):
  # Create a directory for outputs.
  os.makedirs(mkpath(dataset_dir, 'tfrecord'), exist_ok=True)

  # Create a pool for multi-processing.
  # The number of processes will be set as same as the number of cpus.
  with Pool(processes=None) as pool:
    for split in ['train', 'val', 'test']:
      print(f'=> Processing split "{split}".')
      df_split = df[df['split'] == split]
      shards = df_split.shard.unique()
      for shard in sorted(shards):
        df_split_shard = df_split[df_split['shard'] == shard]
        filename = f'{split}-{shard + 1:04d}-{len(shards):04d}.tfrecord'
        filepath = mkpath(dataset_dir, 'tfrecord', filename)
        with tf.python_io.TFRecordWriter(filepath) as writer:
          list_args = [(row, config, split) for _, row in df_split_shard.iterrows()]
          for i, examples in enumerate(pool.imap(process_audio, list_args)):
            for example in examples:
              writer.write(example.SerializeToString())
            progress = int(round((i + 1) / len(list_args) * 100))
            print(f'\rShard ({shard+1:04d}/{len(shards):04d}): {progress:3d}%', end='', flush=True)
        print()


def process_audio(args):
  row, config, split = args
  try:
    sequence = True if split == 'test' else False
    examples = data.audio.to_tfrecord_examples(row, config, sequence)
  except Exception as e:
    print('=> Error: cannot load audio (reason below): ' + row['path'])
    print(e)
    examples = []
  return examples


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Build a dataset.')
  parser.add_argument('dataset', type=str, metavar='DATASET', choices=['mtt', 'scd', 'dcs'],
                      help='A dataset to build: {mtt|scd|dcs}')
  parser.add_argument('--data-dir', type=str, default='./data', metavar='PATH')

  args = parser.parse_args()

  main(args)
  print('\n=> Done.\n')
