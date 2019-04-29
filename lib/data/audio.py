import librosa
import tensorflow as tf
import numpy as np


def to_tfrecord_examples(row, config, sequence):
  audio_path, label = row['path'], row['label']
  sr, num_samples, num_segments, len_audio = config.sr, config.num_samples, config.num_segments, config.len_audio

  audio = load_audio(audio_path, sr, len_audio)
  segments = [audio[i * num_samples:(i + 1) * num_samples] for i in range(num_segments)]

  if sequence:
    examples = [segments_to_sequence_example(segments, label)]
  else:
    examples = [segment_to_example(segment, label) for segment in segments]

  return examples


def segment_to_example(segment, label):
  raw_segment = np.array(segment, dtype=np.float32).reshape(-1).tostring()
  raw_label = np.array(label, dtype=np.uint8).reshape(-1).tostring()

  example = tf.train.Example(features=tf.train.Features(feature={
    'label': bytes_feature(raw_label),  # array: dtype=uint8, shape=(num_classes,)
    'segment': bytes_feature(raw_segment)  # array: dtype=float32, shape=(num_samples,)
  }))

  return example


def segments_to_sequence_example(segments, label):
  raw_segments = [np.array(segment, dtype=np.float32).reshape(-1).tostring() for segment in segments]
  raw_label = np.array(label, dtype=np.uint8).reshape(-1).tostring()

  sequence_example = tf.train.SequenceExample(
    context=tf.train.Features(feature={
      'label': bytes_feature(raw_label)  # uint8 Tensor (50,)
    }),
    feature_lists=tf.train.FeatureLists(feature_list={
      'segments': bytes_feature_list(raw_segments)  # list of float32 Tensor (num_samples,)
    }))

  return sequence_example


def load_audio(path, sr, len_audio):
  audio, _ = librosa.load(path, sr=sr, mono=True, duration=len_audio, dtype=np.float32, res_type='kaiser_best')

  total_samples = sr * len_audio
  if len(audio) < total_samples:
    audio = np.repeat(audio, total_samples // len(audio) + 1)[:total_samples]

  return audio


def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_feature_list(values):
  return tf.train.FeatureList(feature=[bytes_feature(v) for v in values])


def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
