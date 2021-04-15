# -*- coding: utf-8 -*-
"""
created on 20210413
@author: whull
工具包
"""

import tensorflow as tf


# @训练数据变换
def shift_by_one(inputs, bos):
    '''Shifts the content of `inputs` to the right by one
      so that it becomes the decoder inputs.

    Args:
      inputs: A 3d tensor with shape of [N, T, C]
      bos: int, begin token index

    Returns:
      A 3d tensor with the same shape and dtype as `inputs`.
    '''
    return tf.concat((tf.ones_like(inputs[:, :1])*bos, inputs[:, :-1]), 1)
