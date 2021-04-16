# -*- coding: utf-8 -*-
"""
created on 20210413
@author: whull
"""

import tensorflow as tf
from tensorflow.python.util import nest

from . import attention_wrapper, modules


def local_attention_lstm(units, input, is_training):
    decoder_inputs_length = tf.count_nonzero(input, axis=2, dtype=tf.int32)
    decoder_inputs_length = tf.count_nonzero(decoder_inputs_length, axis=1, dtype=tf.int32)
    attention_mechanism = attention_wrapper.LocationBasedAttention(num_units=units,
                                                                   memory=input,
                                                                   convk=5,
                                                                   convr=16,
                                                                   memory_sequence_length=decoder_inputs_length)
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(units)
    lstm_cell_drop = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell,
                                                   output_keep_prob=0.5 if is_training else 1)
    cell_with_attetion = attention_wrapper.AttentionWrapper(lstm_cell_drop,
                                                            attention_mechanism,
                                                            units)
    return cell_with_attetion


class DecoderLSTM(object):
    def __init__(self, hp, is_training):
        self.hp = hp
        self.is_training = is_training

    def __call__(self, decoder_inputs, memory, encoder_states):
        vocab_size = self.hp.vocab_size
        bos = self.hp.bos
        eos = self.hp.eos

        if self.is_training:
            memory = memory
            encoder_states = encoder_states
            batch_size = self.hp.batch_size
        else:
            memory = tf.contrib.seq2seq.tile_batch(memory, multiplier=self.hp.beam_size)
            encoder_states = nest.map_structure(lambda s: tf.contrib.seq2seq.tile_batch(s, self.hp.beam_size),
                                                encoder_states)
            batch_size = self.hp.batch_size * self.hp.beam_size

        cell_with_attention = local_attention_lstm(units=self.hp.decode_num_units, input=memory,
                                                   is_training=self.is_training)

        h_decode_initial = cell_with_attention.zero_state(batch_size=batch_size, dtype=tf.float32).clone(
            cell_state=encoder_states)
        output_layer = tf.layers.Dense(vocab_size,
                                       kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

        embedding = tf.get_variable('decoder_embedding', [vocab_size, self.hp.embed_size])

        print('-' * 20)
        if self.is_training:
            # 定义decoder阶段的输入，进行embedding?
            decoder_inputs_emb = tf.nn.embedding_lookup(embedding, modules.shift_by_one(decoder_inputs, bos))
            # 定义 targets_length
            targets_length = tf.count_nonzero(decoder_inputs, axis=1, dtype=tf.int32)
            # 训练阶段，使用TrainingHelper+BasicDecoder的组合，这一般是固定的，当然也可以自己定义Helper类，实现自己的功
            training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_inputs_emb,
                                                                sequence_length=targets_length,
                                                                time_major=False,
                                                                name='training_helper')

            training_decoder = tf.contrib.seq2seq.BasicDecoder(cell=cell_with_attention,
                                                               helper=training_helper,
                                                               initial_state=h_decode_initial,
                                                               output_layer=output_layer)

            # 调用dynamic_decode进行解码，decoder_outputs是一个namedtuple，里面包含两�?rnn_outputs, sample_id)
            # rnn_output: [batch_size, decoder_targets_length, vocab_size]，保存decode每个时刻每个单词的概率，可以用来计算loss
            # sample_id: [batch_size], tf.int32，保存最终的编码结果。可以表示最后的答案
            decoder_outputs, final_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder=training_decoder,
                                                                                impute_finished=True,
                                                                                maximum_iterations=50)

        else:
            start_tokens = tf.fill([self.hp.batch_size], bos)
            print('start_tokens, end_token', start_tokens, eos)
            if self.hp.is_beam_search:
                inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=cell_with_attention,
                                                                         embedding=embedding,
                                                                         start_tokens=start_tokens,
                                                                         end_token=eos,
                                                                         initial_state=h_decode_initial,
                                                                         beam_width=self.hp.beam_size,
                                                                         output_layer=output_layer)

            else:
                decoding_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=embedding,
                                                                           start_tokens=start_tokens,
                                                                           end_token=eos)
                inference_decoder = tf.contrib.seq2seq.BasicDecoder(cell=cell_with_attention,
                                                                    helper=decoding_helper,
                                                                    initial_state=h_decode_initial,
                                                                    output_layer=output_layer)

            decoder_outputs, final_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder=inference_decoder,
                                                                                maximum_iterations=30)

        return decoder_outputs
