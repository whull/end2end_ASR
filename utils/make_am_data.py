
import os
import csv
import traceback

import tensorflow as tf

from . import tools
from . import am_utils


def make_tfrecord(hp, txt_dir, vocab_file, tfrecord_file):
    phone2id, _ = tools.load_vocab(vocab_file)
    flag = 0
    writer = tf.python_io.TFRecordWriter(os.path.join(tfrecord_file, "train_0.tfrecord"))
    for wav_file, label in csv.reader(open(txt_dir, "r")):
        try:
            melspec = am_utils.get_spectrogram(wav_file, hp)
            label = [phone2id[i] if i in phone2id else phone2id["<unk>"] for i in eval(label)] + [phone2id["<eos>"]]

            feat = dict()
            feat["label"] = tf.train.Feature(int64_list=tf.train.Int64List(value=label))
            feat["mel"] = tf.train.Feature(float_list=tf.train.FloatList(value=melspec.reshape(-1)))
            feat["mel_shape"] = tf.train.Feature(int64_list=tf.train.Int64List(value=melspec.shape))

            example = tf.train.Example(features=tf.train.Features(feature=feat))
            writer.write(example.SerializeToString())
            flag += 1
            if flag % 1e7 == 0:
                writer.close()
                writer = tf.python_io.TFRecordWriter(tfrecord_file + f"/train_{int(flag//1e7)}.tfrecord")
        except Exception as e:
            traceback.print_exc()
            print(e)
            flag += 1
            break


if __name__ == '__main__':
    from config.hyperparameters import AudioHP
    hp = AudioHP()
    txt_dir = r"data/train_data.csv"
    vocab_file = r"dict/syllable.txt"
    tfrecord_file = r"data"
    make_tfrecord(hp, txt_dir, vocab_file, tfrecord_file)
