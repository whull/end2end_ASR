# -*- coding: utf-8 -*-
"""
created on 20210413
@author: whull
工具包
"""

import logging


def load_vocab(vocab_file):
    with open(vocab_file, "r") as f:
        phones = []
        for phone in f:
            phones.append(phone.strip())

    phone2id = dict(zip(phones, range(len(phones))))
    id2phone = dict(zip(range(len(phones)), phones))
    return phone2id, id2phone


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
