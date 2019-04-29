import math
import tensorflow as tf
from tensorflow.keras.initializers import Initializer


def _compute_audio_fans(shape):
  assert len(shape) == 3, 'This initialization is for Conv1D.'

  len_filter, in_channels, out_channels = shape

  receptive_field_size = len_filter * in_channels  # 원래는 len_filter 여야함!!
  fan_in = in_channels * receptive_field_size
  fan_out = out_channels * receptive_field_size

  return fan_in, fan_out


class AudioVarianceScaling(Initializer):
  """VarianceScaling for Audio"""

  def __init__(self,
               scale=1.0,
               mode="fan_in",
               distribution="truncated_normal",
               seed=None,
               dtype=tf.float32):
    if scale <= 0.:
      raise ValueError("`scale` must be positive float.")
    if mode not in {"fan_in", "fan_out", "fan_avg"}:
      raise ValueError("Invalid `mode` argument:", mode)
    distribution = distribution.lower()
    if distribution not in {"uniform", "truncated_normal", "untruncated_normal"}:
      raise ValueError("Invalid `distribution` argument:", distribution)
    self.scale = scale
    self.mode = mode
    self.distribution = distribution
    self.seed = seed
    self.dtype = tf.as_dtype(dtype)

  def __call__(self, shape, dtype=None, partition_info=None):
    if dtype is None:
      dtype = self.dtype
    scale = self.scale
    scale_shape = shape
    if partition_info is not None:
      scale_shape = partition_info.full_shape
    fan_in, fan_out = _compute_audio_fans(scale_shape)
    if self.mode == "fan_in":
      scale /= max(1., fan_in)
    elif self.mode == "fan_out":
      scale /= max(1., fan_out)
    else:
      scale /= max(1., (fan_in + fan_out) / 2.)
    if self.distribution == "normal" or self.distribution == "truncated_normal":
      # constant taken from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
      stddev = math.sqrt(scale) / .87962566103423978
      return tf.truncated_normal(
        shape, 0.0, stddev, dtype, seed=self.seed)
    elif self.distribution == "untruncated_normal":
      stddev = math.sqrt(scale)
      return tf.random_normal(
        shape, 0.0, stddev, dtype, seed=self.seed)
    else:
      limit = math.sqrt(3.0 * scale)
      return tf.random_uniform(
        shape, -limit, limit, dtype, seed=self.seed)

  def get_config(self):
    return {
      "scale": self.scale,
      "mode": self.mode,
      "distribution": self.distribution,
      "seed": self.seed,
      "dtype": self.dtype.name
    }


def taejun_uniform(scale=2., seed=None):
  return AudioVarianceScaling(scale=scale, mode='fan_in', distribution='uniform', seed=seed)
