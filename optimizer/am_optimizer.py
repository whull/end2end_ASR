# -*- coding: utf-8 -*-
"""
created on 20210413
@author: whull
"""

import tensorflow as tf
try:
    import horovod.tensorflow as hvd
    use_hvd = True
except:
    use_hvd = False


class SummaryAndOptimizer(object):
    def __init__(self, hp):
        self.hp = hp

    def __call__(self, loss, k, **kwargs):
        global_step = tf.train.get_or_create_global_step()
        learning_rate = self.hp.lr * decay_lr(self.hp.warm_up, global_step) * k
        # Horovod: adjust learning rate based on number of GPUs.
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        # Horovod: add Horovod Distributed Optimizer.
        optimizer = hvd.DistributedOptimizer(optimizer) if use_hvd else optimizer

        train_opt = optimizer.minimize(loss, global_step=global_step)

        tf.summary.scalar("loss", loss)
        tf.summary.scalar("lr", learning_rate)
        scalar = []
        for k, v in kwargs.items():
            tf.summary.scalar(k, v)
            scalar.append(v)
        summary = tf.summary.merge_all()

        tuple_op = tf.tuple([global_step, loss, learning_rate, summary],
                            control_inputs=[train_opt])

        return tuple_op


def decay_lr(warmup_steps, global_step):
    """
    warmup和学习率衰减
    :param warmup_steps:
    :param global_step:
    :return:
    """
    global_step = tf.to_float(global_step)
    return 1024 ** -0.5 * tf.minimum(
        (global_step + 1.0) * warmup_steps ** -1.5, (global_step + 1.0) ** -0.5)
