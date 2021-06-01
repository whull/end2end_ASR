# -*- coding: utf-8 -*-
"""
created on 20210413
@author: whull
asr 参数设置
"""

import os


class AudioHP:
    # signal processing
    sr = 16000  # Sampling rate.
    n_fft = 512  # fft points (samples)
    frame_shift = 0.01  # seconds
    frame_length = 0.025  # seconds
    hop_length = int(sr * frame_shift)  # samples  This is dependent on the frame_shift.
    win_length = int(sr * frame_length)  # samples This is dependent on the frame_length.
    n_mels = 80  # Number of Mel banks to generate
    r = 3  # reduction factor. Paper => 2, 3, 5


class LASLSTMHyperparams(AudioHP):
    """Hyper parameters"""
    # model
    # encode
    num_encode_layer = 3
    encode_num_units = 512
    cell_type = 'on_lstm'  # 默认basic_lstm, 可选[on_lstm, basic_lstm]
    num_levels = 4  # lstm_type='on_lstm'时使用，能被encode_num_units整除
    bi_direction = True

    # decode
    embed_size = 100  # alias = E
    decode_num_units = 1024  # bi_direction=True时，encode_num_units * 2
    is_beam_search = True
    beam_size = 5

    use_spec_augment = True
    n_freq_mask = 2
    n_time_mask = 2
    width_freq_mask = 15
    # an upper bound on the time mask so that a time mask cannot be wider than p times the number of time steps
    p = 0.1
    # training scheme
    lr = 1
    warm_up = 8000
    logdir = "logdir"
    batch_size = 10  # 200

    num_epochs = 200


class MHALSTMHyperparams(AudioHP):
    pass
