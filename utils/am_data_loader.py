# -*- coding: utf-8 -*-
"""
created on 20210413
@author: whull
数据获取
"""

import os
import sys
import csv

import numpy as np
import tensorflow as tf


class LoadTrainDataMask(object):
    def __init__(self, tfrecord_file, hp, ID):
        self.tfrecord_file = np.random.permutation(tfrecord_file)
        print(f"第{ID}个进程，共处理{len(self.tfrecord_file)}个文件，第一个文件为{self.tfrecord_file[0]}")
        self.hp = hp

    def get_batch_input(self, num_parallel_calls=32):
        dataset = tf.data.TFRecordDataset(self.tfrecord_file)
        dataset = dataset.apply(
            tf.contrib.data.shuffle_and_repeat(
                buffer_size=self.hp.batch_size*32, count=self.hp.num_epochs))
        print("Shuffled and Repeated!")
        dataset = dataset.map(self._parse_function, num_parallel_calls=num_parallel_calls)
        print("Mapped!")
        dataset = dataset.padded_batch(batch_size=self.hp.batch_size,
                                       padded_shapes={"mel": [None, self.hp.n_mels*self.hp.r],
                                                      "mel_shape": [None],
                                                      "label": [None],
                                                      "mel_length": [None]
                                                      })
        print("Batch padded!")
        dataset = dataset.prefetch(buffer_size=2)
        print("Prefetched!")

        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()

        return next_element

    def _parse_function(self, example_proto):
        # example_proto，tf_serialized
        features = {
            # we can use VarLenFeature, but it returns SparseTensor
            "mel": tf.VarLenFeature(dtype=tf.float32),
            # FixedLenFeature是按照键值对将features映射到大小为[serilized.size(),df.shape]的矩阵，
            # 这里的FixLenFeature指的是每个键值对应的feature的size是一样的
            "mel_shape": tf.FixedLenFeature(shape=(2,), dtype=tf.int64),
            "label": tf.VarLenFeature(dtype=tf.int64)
        }

        # parse all features in a single example according to the dics
        parsed_example = tf.parse_single_example(serialized=example_proto, features=features)
        # sparse_tensor_to_dense
        parsed_example["mel"] = tf.sparse_tensor_to_dense(parsed_example["mel"])
        parsed_example["label"] = tf.sparse_tensor_to_dense(parsed_example["label"])
        # reshape
        parsed_example["mel"] = tf.reshape(parsed_example["mel"], parsed_example["mel_shape"])
        pad_size = -parsed_example["mel_shape"][0] % 3
        parsed_example["mel"] = tf.pad(parsed_example["mel"],
                                       tf.convert_to_tensor([[0, pad_size], [0, 0]]), "CONSTANT")
        parsed_example["mel"] = tf.reshape(parsed_example["mel"], [-1, self.hp.n_mels*self.hp.r])
        parsed_example["mel_length"] = tf.to_int32(tf.not_equal(
            tf.count_nonzero(parsed_example["mel"], axis=1, dtype=tf.int32), 0))
        # transform dtype
        parsed_example["mel"] = tf.cast(parsed_example["mel"], dtype=tf.float32)
        parsed_example["label"] = tf.cast(parsed_example["label"], dtype=tf.int32)

        if self.hp.use_spec_augment:
            threshold = tf.constant(0.5, dtype=tf.float32)
            random_seed = tf.random_uniform([], minval=0, maxval=1, dtype=tf.float32)

            parsed_example["mel"] = tf.cond(
                tf.greater_equal(random_seed, threshold),
                true_fn=lambda: self._spec_augment(parsed_example["mel"]),
                false_fn=lambda: parsed_example["mel"]
            )

        return parsed_example

    def _spec_augment(self, input_mel):

        mel = tf.reshape(input_mel, (-1, self.hp.n_mels))
        n_frames, mel_channels = tf.shape(mel)[0], tf.shape(mel)[1]

        # Step 2 : Frequency masking
        for _ in range(self.hp.n_freq_mask):
            f = tf.random_uniform([], minval=0, maxval=self.hp.width_freq_mask + 1, dtype=tf.int32)
            f0 = tf.random_uniform([], minval=0, maxval=mel_channels - f, dtype=tf.int32)

            # warped_mel[:, f0:f0 + f] = 0
            freq_mask = tf.concat((tf.ones(shape=(n_frames, mel_channels - f0 - f)),
                                   tf.zeros(shape=(n_frames, f)),
                                   tf.ones(shape=(n_frames, f0)),
                                   ), axis=1)

            mel = mel * freq_mask

        # Step 3 : Time masking
        for _ in range(self.hp.n_time_mask):
            width_time_mask = tf.to_int32(tf.to_float(n_frames) * tf.to_float(self.hp.p))
            t = tf.random_uniform([], minval=0, maxval=width_time_mask + 1, dtype=tf.int32)
            t0 = tf.random_uniform([], minval=0, maxval=n_frames - t, dtype=tf.int32)

            # warp_mel[t0:t0 + t, :] = 0
            time_mask = tf.concat((tf.ones(shape=(n_frames - t0 - t, mel_channels)),
                                   tf.zeros(shape=(t, mel_channels)),
                                   tf.ones(shape=(t0, mel_channels)),
                                   ), axis=0)

            mel = mel * time_mask

        return tf.reshape(mel, tf.shape(input_mel))


def load_eval_data(filepath):
    import ast
    char2idx, idx2char = load_vocabulary()
    same_file_labels = csv.reader(open(filepath, "r", encoding="utf-8"))
    sound_fpaths, converteds = [], []
    for same_file_label in same_file_labels:
        text_list = ast.literal_eval(same_file_label[1])
        if len(text_list) >= 30:
            continue
        converted = np.array([char2idx[char] if char in char2idx else 0 for char in text_list], np.int32)
        converteds.append(converted)
        sound_fpaths.append(same_file_label[0])
    return sound_fpaths, converteds

