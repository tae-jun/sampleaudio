class DatasetConfig:

  def __init__(self, num_blocks, num_samples, sr, len_audio, num_audios_per_shard,
               num_classes, loss, metrics, activation, mean, std,
               num_train_audios, num_test_audios, num_val_audios, threshold=0.5):
    self.num_blocks = num_blocks
    self.num_samples = num_samples
    self.sr = sr
    self.len_audio = len_audio
    self.num_segments = len_audio * sr // num_samples
    self.num_audios_per_shard = num_audios_per_shard

    self.num_train_audios = num_train_audios
    self.num_val_audios = num_val_audios
    self.num_test_audios = num_test_audios
    self.num_train_segs = num_train_audios * self.num_segments
    self.num_val_segs = num_val_audios * self.num_segments
    self.num_test_segs = num_test_audios * self.num_segments

    self.num_classes = num_classes
    self.loss = loss
    self.metrics = metrics
    self.activation = activation
    self.threshold = threshold

    self.mean = mean
    self.std = std


MTT_CONFIG = DatasetConfig(num_blocks=9, num_samples=59049, sr=22050, len_audio=29, num_audios_per_shard=100,
                           num_train_audios=15250, num_val_audios=1529, num_test_audios=4332,
                           loss='binary_crossentropy', metrics=None, activation='sigmoid', num_classes=50,
                           mean=-0.0001650025078561157, std=0.1551193743944168)

SCD_CONFIG = DatasetConfig(num_blocks=8, num_samples=22050, sr=22050, len_audio=1, num_audios_per_shard=1000,
                           num_train_audios=84843, num_val_audios=9981, num_test_audios=11005,
                           loss='categorical_crossentropy', metrics=['accuracy'], activation='softmax', num_classes=35,
                           mean=-8.520474e-05, std=0.18)

DCS_CONFIG = DatasetConfig(num_blocks=8, num_samples=22050, sr=22050, len_audio=10, num_audios_per_shard=300,
                           num_train_audios=46042, num_val_audios=5618, num_test_audios=1103,
                           loss='binary_crossentropy', metrics=['accuracy'], activation='sigmoid', num_classes=17,
                           mean=-0.0003320679534226656, std=0.20514629781246185, threshold=0.1)
