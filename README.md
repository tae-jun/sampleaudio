# SampleCNNs for Audio Classifications
This repository contains the code that used to the publication below:
> Taejun Kim, Jongpil Lee, and Juhan Nam, "Comparison and Analysis of SampleCNN Architectures for Audio Classification"
in IEEE Journal of Selected Topics in Signal Processing (JSTSP), 2019.



Contents:
* Install Dependencies
* Building Datasets
  * Music auto-tagging: MagnaTagATune
  * Keyword spotting: Speech Commands
  * Acoustic scene tagging: DCASE 2017 Task 4
* Training a SampleCNN

## Dependency Installation
NOTE: The code of this repository is written and tested on **Python 3.6**.
 
* tensorflow 1.10.X (strongly recommend to use 1.10.X because of version compatibility)
* librosa
* ffmpeg
* pandas
* numpy
* scikit-learn
* h5py

To install the required python packages using conda, run the command below:
```sh
conda install tensorflow-gpu=1.10.0 ffmpeg pandas numpy scikit-learn h5py
conda install -c conda-forge librosa
```


## Building Datasets
Download and preprocess a dataset that you want to train a model on.

### Music auto-tagging: [MagnaTagATune][2]
> Edith Law, Kris West, Michael Mandel, Mert Bay and J. Stephen Downie (2009).
[Evaluation of algorithms using games: the case of music annotation.][1]
In  Proceedings of the 10th International Conference on Music Information Retrieval (ISMIR).

Create a directory for the dataset and download required one `.csv` file and three `.zip` files in the directory `data/mtt/raw`:
```sh
mkdir -p data/mtt/raw
cd data/mtt/raw
wget http://mi.soi.city.ac.uk/datasets/magnatagatune/annotations_final.csv
wget http://mi.soi.city.ac.uk/datasets/magnatagatune/mp3.zip.001
wget http://mi.soi.city.ac.uk/datasets/magnatagatune/mp3.zip.002
wget http://mi.soi.city.ac.uk/datasets/magnatagatune/mp3.zip.003
```
 
After download the files, merge and expand the three `.zip` files:
```sh
cat mp3.zip.* > mp3_all.zip
unzip mp3_all.zip -d mp3
```

Your directory structure should look like this:
```sh
data
└── mtt
    └── raw
        ├── annotations_final.csv
        └── mp3
            ├── 0
            ├── ...
            └── f
```

Finally, segment and convert audios to TFRecords using following command:
```sh
python build_dataset.py mtt
```


### Keyword spotting: [Speech Commands][3]
> Pete Warden (2018).
[Speech commands: A dataset for limited-vocabulary speech recognition.][4]
arXiv:1804.03209.

After create a directory for the dataset, download and expand the dataset in the directory `data/scd/raw`:
```sh
mkdir -p data/scd/raw
cd data/scd/raw
wget http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
tar zxvf speech_commands_v0.02.tar.gz
```


Finally, segment and convert audios to TFRecords using following command:
```sh
python build_dataset.py scd
```


### Acoustic scene tagging: [DCASE 2017 Task 4][5]
> Annamaria Mesaros, Toni Heittola, Aleksandr Diment, Benjamin Elizalde, Ankit Shah, Emmanuel Vincent, Bhiksha Raj and Tuomas Virtanen (2017).
[DCASE 2017 challenge setup: tasks, datasets and baseline system.][6]
In Proceedings of the Detection and Classification of Acoustic Scenes and Events 2017 Workshop (DCASE2017).

```sh
mkdir -p data/dcs/raw
cd data/dcs/raw

wget --no-check-certificate -r 'https://docs.google.com/uc?export=download&id=1HOQaUHbTgCRsS6Sr9I9uE6uCjiNPC3d3' -O Task_4_DCASE_2017_training_set.zip
wget --no-check-certificate -r 'https://docs.google.com/uc?export=download&id=1GfP5JATSmCqD8p3CBIkk1J90mfJuPI-k' -O Task_4_DCASE_2017_testing_set.zip
wget https://dl.dropboxusercontent.com/s/bbgqfd47cudwe9y/DCASE_2017_evaluation_set_audio_files.zip

unzip -P DCASE_2017_training_set Task_4_DCASE_2017_training_set.zip
unzip -P DCASE_2017_testing_set Task_4_DCASE_2017_testing_set.zip
unzip -P DCASE_2017_evaluation_set DCASE_2017_evaluation_set_audio_files.zip

wget https://github.com/ankitshah009/Task-4-Large-scale-weakly-supervised-sound-event-detection-for-smart-cars/raw/master/groundtruth_release/groundtruth_weak_label_training_set.csv
wget https://github.com/ankitshah009/Task-4-Large-scale-weakly-supervised-sound-event-detection-for-smart-cars/raw/master/groundtruth_release/groundtruth_weak_label_testing_set.csv
wget https://github.com/ankitshah009/Task-4-Large-scale-weakly-supervised-sound-event-detection-for-smart-cars/raw/master/groundtruth_release/groundtruth_weak_label_evaluation_set.csv
```

Finally, segment and convert audios to TFRecords using following command:
```sh
python build_dataset.py dcs
```

## Training a SampleCNN
You can train a SampleCNN with a block on a dataset that you want.
Here are several examples to run training:
```sh
# Train a SampleCNN with SE block (default) on MagnaTagATune dataset (music auto-tagging)
python train.py mtt

# Train a SampleCNN with ReSE-2 block on Speech Commands dataset (keyword spotting)
python train.py scd --block rese2

# Train a SampleCNN with basic block on DCASE 2017 Task 4 dataset (acoustic scene tagging
python train.py dcs --block basic
```
Trained models are saved under `log` directory with a datetime that you started running.
Here is an example of saved model:
```sh
log/
    └── 20190424_213449-scd-se/
        └── final-auc_0.XXXXXX-acc_0.XXXXXX-f1_0.XXXXXX.h5
```

You can see the available options for training using the command below:
```sh
$ python train.py -h

usage: train.py [-h] [--data-dir PATH] [--log-dir PATH]
                [--block {basic,se,res1,res2,rese1,rese2}]
                [--amplifying-ratio N] [--multi] [--batch-size N]
                [--momentum M] [--lr LR] [--lr-decay DC] [--dropout DO]
                [--weight-decay WD] [--num-stages N] [--patience N]
                [--num-readers N]
                DATASET [NAME]

Train a SampleCNN.

positional arguments:
  DATASET               Dataset for training: {mtt|scd|dcs}
  NAME                  Name of log directory.

optional arguments:
  -h, --help            show this help message and exit
  --data-dir PATH
  --log-dir PATH        Directory where to write event logs and models.
  --block {basic,se,res1,res2,rese1,rese2}
                        Convolutional block to build a model (default: se,
                        options: basic/se/res1/res2/rese1/rese2).
  --amplifying-ratio N
  --multi               Use multi-level feature aggregation.
  --batch-size N        Mini-batch size.
  --momentum M          Momentum for SGD.
  --lr LR               Learning rate.
  --lr-decay DC         Learning rate decay rate.
  --dropout DO          Dropout rate.
  --weight-decay WD     Weight decay.
  --num-stages N        Number of stages to train.
  --patience N          Stop training stage after #patiences.
  --num-readers N       Number of TFRecord readers.
```


[1]: http://ismir2009.ismir.net/proceedings/OS5-5.pdf
[2]: http://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset
[3]: https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html
[4]: https://arxiv.org/pdf/1804.03209.pdf
[5]: http://dcase.community/challenge2017/task-large-scale-sound-event-detection
[6]: http://dcase.community/documents/workshop2017/proceedings/DCASE2017Workshop_Mesaros_100.pdf
