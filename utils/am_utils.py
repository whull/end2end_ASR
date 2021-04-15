# -*- coding: utf-8 -*-
"""
created on 20210413
@author: whull
工具包
"""

import os
import sys
import logging

import librosa
import numpy as np


# @原始音频处理
def get_spectrogram(sound_fpath, hp):
    '''Extracts melspectrogram and magnitude from given `sound_file`.
    Args:
      sound_fpath: A string. Full path of a sound file.

    Returns:
      Transposed S: A 2d array. A transposed melspectrogram with shape of (T, n_mels)
      Transposed magnitude: A 2d array. A transposed magnitude spectrogram
        with shape of (T, 1+hp.n_fft//2)
    '''
    # Loading sound file
    y, sr = librosa.load(sound_fpath, sr=None)  # or set sr to hp.sr.
    y /= max(y)
    # stft. D: (1+n_fft//2, T)
    D = librosa.stft(y=y,
                     n_fft=hp.n_fft,
                     hop_length=hp.hop_length,
                     win_length=hp.win_length)

    # magnitude spectrogram
    magnitude = np.abs(D)  # (1+n_fft/2, T)

    # power spectrogram
    power = magnitude ** 2

    # mel spectrogram
    S = librosa.feature.melspectrogram(S=power, n_mels=hp.n_mels, sr=sr)  # (n_mels, T)

    return np.transpose(S.astype(np.float32))


def reduce_frames(arry, hp):
    """Reduces and adjust the shape and content of `arry` according to r.

    Args:
      arry: A 2d array with shape of [T, C]
      r: Reduction factor

    Returns:
      A 2d array with shape of [-1, C*r]
    """
    T, C = arry.shape
    num_paddings = hp.r - (T % hp.r) if T % hp.r != 0 else 0

    padded = np.pad(arry, [[0, num_paddings], [0, 0]], 'constant')
    output = np.reshape(padded, (-1, C * hp.r))
    return output


class ValueWindow(object):
    def __init__(self, window_size=100):
        self._window_size = window_size
        self._values = []

    def append(self, x):
        self._values = self._values[-(self._window_size - 1):] + [x]

    @property
    def sum(self):
        return sum(self._values)

    @property
    def count(self):
        return len(self._values)

    @property
    def average(self):
        return self.sum / max(1, self.count)

    def reset(self):
        self._values = []


def logger(save_file, name, level=logging.INFO, save_level=None):
    # get TF logger
    log = logging.getLogger(name)
    log.setLevel(level)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # create file handler which logs even debug messages
    fh = logging.FileHandler(save_file)
    stream_handler = logging.StreamHandler()

    if not save_level:
        save_level = level
    fh.setLevel(save_level)  # 设置文件 输出等级
    fh.setFormatter(formatter)  # 设置文件 输出格式
    stream_handler.setFormatter(formatter)  # 设置窗口 输出格式

    log.addHandler(fh)  # 为记录器添加 处理方式
    log.addHandler(stream_handler)  # 为记录器添加 处理方式
    return log
