# -*- coding: utf-8 -*-
"""
created on 20210413
@author: whull

"""

import os
import time
import math
from glob import glob
import traceback

import tensorflow as tf
try:
    import horovod.tensorflow as hvd
    use_hvd = True
except:
    use_hvd = False

from optimizer import am_optimizer
from config import hyperparameters
from utils import am_data_loader, tools
from AMmodel.las_lstm_lstm import LASLSTMModel

hvd.init() if use_hvd else None
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank()) if use_hvd else "0"


def main():
    hp = hyperparameters.LASLSTMHyperparams()
    hp.log_iterval = 1
    hp.checkpoint_interval = 500  # 保存模型的频率
    hp.bos, hp.eos = 1, 2
    hp.vocab_file = r"dict/syllables.txt"
    phone2id, _ = tools.load_vocab(hp.vocab_file)
    hp.vocab_size = len(phone2id)

    log = tools.logger(os.path.join(hp.logdir, "train.log"), "train", tf.logging.INFO)

    rank = hvd.rank() if use_hvd else 0
    size = hvd.size() if use_hvd else 1e9
    k = size if use_hvd else 1  # 多卡训练学习率调整
    tfrecord_path = r"data"
    files = sorted(glob(os.path.join(tfrecord_path, "train_*.tfrecord")))
    files_list = [file for i, file in enumerate(files) if i % size == rank]

    traindata = am_data_loader.LoadTrainDataMask(files_list, hp, rank)
    next_element = traindata.get_batch_input()

    model = LASLSTMModel(hp, True)
    loss, acc = model.build_train_network(next_element["mel"], next_element["label"], next_element["mel_length"])
    tuple_op = am_optimizer.SummaryAndOptimizer(hp)(loss, k, acc=acc)

    bcast = hvd.broadcast_global_variables(0) if use_hvd else None
    saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=8)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(bcast) if use_hvd else None

        restore_path = tf.train.latest_checkpoint(hp.logdir)
        if restore_path is None:
            log.info('Starting new training')
        else:
            saver.restore(sess, restore_path)
            log.info('Resuming from checkpoint: %s' % restore_path)

        time_window = tools.ValueWindow(100)
        loss_window = tools.ValueWindow(100)

        if rank == 0:
            summary_writer = tf.summary.FileWriter(hp.logdir)

        while True:
            try:
                start_time = time.time()
                step, loss, lr, _summary = sess.run(tuple_op)

                if rank == 0:
                    time_window.append(time.time() - start_time)
                    loss_window.append(loss)
                    if step % hp.log_iterval == 0:
                        log.info('Step %-7d [%.03f sec/step, lr=%.06f, loss=%.05f, avg_loss=%.05f]' % (
                            step, time_window.average, lr, loss, loss_window.average))

                        summary_writer.add_summary(_summary, step)

                    if loss > 100 or math.isnan(loss):
                        log.info('Loss exploded to %.05f at step %d!' % (loss, step))
                        raise Exception('Loss Exploded')

                    if step % hp.checkpoint_interval == 0:
                        log.info('Saving checkpoint to: %s-%d' % (os.path.join(hp.logdir, "model.ckpt"), step))
                        saver.save(sess, os.path.join(hp.logdir, "model.ckpt"), global_step=step)
            except Exception as e:
                traceback.print_exc()
                log.info(e)
                break


if __name__ == '__main__':
    main()
