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

from .modules import onlstm_cell, attention_lstm


class LASLSTMModel(object):
    def __init__(self, hp, is_training=True):
        self.hp = hp
        self.is_training = is_training

    def build_train_network(self, x, y, xlen):
        encoder_memory, encoder_states = self.encoder(x, xlen, self.is_training, self.hp.cell_type)
        decoder_outputs = self.decoder(y, encoder_memory, encoder_states, self.is_training)
        loss, acc = self.loss_layer(y, decoder_outputs.rnn_output)

        return loss, acc

    def build_inference_network(self):
        # 对于使用beam_search的时候，它里面包含两项(predicted_ids, beam_search_decoder_output)
        # predicted_ids: [batch_size, decoder_targets_length, beam_size],保存输出结果
        # beam_search_decoder_output: BeamSearchDecoderOutput instance namedtuple(scores, predicted_ids, parent_ids)
        # 所以对应只需要返回predicted_ids或者sample_id即可翻译成最终的结果
        if self.hp.is_beam_search:
            print('7' * 20)
            self.preds = self.outputs.predicted_ids
            print('self.preds', self.preds)
        else:
            print('8' * 20)
            self.preds = tf.expand_dims(self.outputs.sample_id, -1)

    def encoder(self, encoder_inputs, xlen, is_training, cell_type="lstm"):
        with tf.variable_scope("encoder"):
            return self.encoder_imp(encoder_inputs, xlen, is_training, cell_type)

    def encoder_imp(self, encoder_inputs, xlen, is_training, cell_type):
        '''
            Args:
              inputs: A 2d tensor with shape of [N, T], dtype of int32.
              is_training: Whether or not the layer is in training mode.

            Returns:
              A collection of Hidden vectors, whose shape is (N, T, E).
            '''
        num_units = self.hp.encode_num_units if self.hp.encode_num_units else encoder_inputs.get_shape().as_list[-1]
        inputs_length = tf.count_nonzero(xlen, axis=1, dtype=tf.int32)

        def lstm_cell():
            return LSTMCell(num_units, cell_type, self.hp.num_levels, is_training)()

        if self.hp.bi_direction:
            cell_forward = tf.nn.rnn_cell.MultiRNNCell(
                [lstm_cell() for _ in range(self.hp.num_encode_layer)])
            cell_backward = tf.nn.rnn_cell.MultiRNNCell(
                [lstm_cell() for _ in range(self.hp.num_encode_layer)])
            (output_forward, output_backward), (state_forward, state_backward) = tf.nn.bidirectional_dynamic_rnn(
                cell_forward,
                cell_backward,
                encoder_inputs,
                sequence_length=inputs_length,
                dtype=tf.float32)
            print('output_forward', output_forward)  # shape=(batch_size, num_steps, num_units)
            memory = tf.concat([output_forward, output_backward], 2)  # [batch_size, num_step, 2*hidden_unit]
            state_c = tf.concat([state_forward[2].c, state_backward[2].c], 1)  # [batch_size, hidden_unit]
            state_h = tf.concat([state_forward[2].h, state_backward[2].h], 1)
            encode_state = tf.contrib.rnn.LSTMStateTuple(state_c, state_h)
            print('state', encode_state)  # shape=(batch_size,num_units*2)
            print('memory', memory)
        else:
            cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell()] for _ in range(self.hp.num_encode_layer))
            memory, encode_state = tf.nn.dynamic_rnn(cell, encoder_inputs, dtype=tf.float32)

        return memory, encode_state

    def decoder(self, decoder_inputs, memory, encoder_states, is_training):
        with tf.variable_scope("decoder"):
            lstm_decoder = attention_lstm.DecoderLSTM(self.hp, is_training)
            return lstm_decoder(decoder_inputs, memory, encoder_states)

    def loss_layer(self, Y, logits):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y,
                                                                       logits=logits)
        mask = tf.to_float(tf.not_equal(Y, 0))
        loss = tf.reduce_sum(cross_entropy * mask) / (tf.reduce_sum(mask) + 1e-7)

        preds = tf.to_int32(tf.argmax(logits, axis=-1))
        acc = tf.reduce_sum(tf.to_float(tf.equal(preds, Y)) * mask) / tf.reduce_sum(mask)

        return loss, acc


class LSTMCell(object):
    def __init__(self, num_units, cell_type='basic_lstm', num_levels=None,
                 is_training=True, keep_dropout=0.5, scope='lstm_layer', reuse=None):
        """
        :param num_units:  隐藏单元数
        :param num_levels:  ON_LSTM中层级数，当cell_type=on_lstm时可用
        :param cell_type:  lstm type，value is basic_lstm or on_lstm
        :param is_training: Whether or not the layer is in training mode.
        :param scope: Optional scope for `variable_scope`
        :param reuse:
        return:
            lstm cell
        """
        self.num_units = num_units  # 隐藏层神经元个数
        self.num_levels = num_levels
        self.cell_type = cell_type
        self.is_training = is_training
        self.keep_dropout = keep_dropout
        self.scope = scope
        self.reuse = reuse

    def __call__(self, ):
        with tf.variable_scope(self.scope, reuse=self.reuse):
            if self.cell_type == 'on_lstm':
                if not self.num_levels:
                    raise ValueError("expect num_levels, but it is None")
                lstm_cell_ = onlstm_cell.ONLSTMCell(self.num_units, self.num_levels)
            elif self.cell_type == 'basic_lstm':
                lstm_cell_ = tf.nn.rnn_cell.BasicLSTMCell(self.num_units)
            else:
                raise ValueError(f"cell_type is {self.cell_type}, not in [basic_lstm, on_lstm]")

            keep_dropout = self.keep_dropout if self.is_training else 1
            lstm_cell_drop = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell_,
                                                           output_keep_prob=keep_dropout)
        return lstm_cell_drop
