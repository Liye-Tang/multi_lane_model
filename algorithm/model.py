#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/6/10
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: model.py
# =====================================

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Conv1D

tf.config.experimental.set_visible_devices([], 'GPU')
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)


class MLPNet(Model):
    def __init__(self, input_dim, num_hidden_layers, num_hidden_units, hidden_activation, output_dim, **kwargs):
        super(MLPNet, self).__init__(name=kwargs['name'])
        self.first_ = Dense(num_hidden_units,
                            activation=hidden_activation,
                            kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2.)),
                            dtype=tf.float32)
        self.hidden = Sequential([Dense(num_hidden_units,
                                        activation=hidden_activation,
                                        kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2.)),
                                        dtype=tf.float32) for _ in range(num_hidden_layers-1)])
        output_activation = kwargs['output_activation'] if kwargs.get('output_activation') else 'linear'
        self.outputs = Dense(output_dim,
                             activation=output_activation,
                             kernel_initializer=tf.keras.initializers.Orthogonal(1.),
                             bias_initializer=tf.keras.initializers.Constant(0.),
                             dtype=tf.float32)
        self.build(input_shape=(None, input_dim))

    def call(self, x, **kwargs):
        x = self.first_(x)
        x = self.hidden(x)
        x = self.outputs(x)
        return x


class AlphaModel(Model):
    def __init__(self, **kwargs):
        super(AlphaModel, self).__init__(name=kwargs['name'])
        self.log_alpha = tf.Variable(0., dtype=tf.float32)


class PPONet(Model):
    def __init__(self, input_dim, num_hidden_layers, num_hidden_units, hidden_activation, output_dim, **kwargs):
        super(PPONet, self).__init__(name=kwargs['name'])
        self.first_ = Dense(num_hidden_units,
                            activation=hidden_activation,
                            kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2.)),
                            dtype=tf.float32)
        self.sec_ = Dense(num_hidden_units,
                          activation=hidden_activation,
                          kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2.)),
                          dtype=tf.float32)
        output_activation = kwargs['output_activation'] if kwargs.get('output_activation') else 'linear'
        self.mean = Dense(int(output_dim/2),
                          activation=output_activation,
                          kernel_initializer=tf.keras.initializers.Orthogonal(0.01),
                          bias_initializer=tf.keras.initializers.Constant(0.),
                          dtype=tf.float32)
        self.logstd = tf.Variable(initial_value=tf.zeros((1, int(output_dim/2))), name='pi/logstd', dtype=tf.float32)
        self.build(input_shape=(None, input_dim))

    def call(self, x, **kwargs):
        x = self.first_(x)
        x = self.sec_(x)
        mean = self.mean(x)
        out = tf.concat([mean, mean * tf.constant(0.) + self.logstd], axis=1)
        return out


class MLPNetDSAC(Model):
    def __init__(self, input_dim, num_hidden_layers, num_hidden_units, hidden_activation, output_dim, **kwargs):
        super(MLPNetDSAC, self).__init__(name=kwargs['name'])
        self.first_ = Dense(num_hidden_units,
                            activation=hidden_activation,
                            kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2.)),
                            dtype=tf.float32)
        self.second_ = Dense(num_hidden_units,
                             activation=hidden_activation,
                             kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2.)),
                             dtype=tf.float32)
        self.hidden_mean = Sequential([Dense(num_hidden_units,
                                             activation=hidden_activation,
                                             kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2.)),
                                             dtype=tf.float32) for _ in range(3)])
        self.hidden_logstd = Sequential([Dense(num_hidden_units,
                                               activation=hidden_activation,
                                               kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2.)),
                                               dtype=tf.float32) for _ in range(3)])
        output_activation = kwargs['output_activation'] if kwargs.get('output_activation') else 'linear'
        self.mean = Dense(int(output_dim/2),
                          activation=output_activation,
                          kernel_initializer=tf.keras.initializers.Orthogonal(0.01),
                          bias_initializer=tf.keras.initializers.Constant(0.),
                          dtype=tf.float32)
        self.logstd = Dense(int(output_dim/2),
                            activation=output_activation,
                            kernel_initializer=tf.keras.initializers.Orthogonal(0.01),
                            bias_initializer=tf.keras.initializers.Constant(0.),
                            dtype=tf.float32)

        self.build(input_shape=(None, input_dim))

    def call(self, x, **kwargs):
        x = self.first_(x)
        x = self.second_(x)
        mean = self.hidden_mean(x)
        mean = self.mean(mean)
        logstd = self.hidden_logstd(x)
        logstd = self.logstd(logstd)
        logstd = tf.clip_by_value(logstd, -5., -1.)
        return tf.concat([mean, logstd], axis=-1)


class AttentionNet(Model):
    def __init__(self, d_others, d_obj, d_model, **kwargs):
        '''
        :d_others: the total dimension of all participants
        :d_obj: the dimension of single participant
        '''
        super(AttentionNet, self).__init__(name=kwargs['name'])
        self.embedding = Conv1D(d_model, kernel_size=1)

        self.Uq = Dense(d_model,
                        use_bias=False,
                        kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2.)),
                        dtype=tf.float32)
        self.Ur = Dense(d_model,
                        use_bias=False,
                        kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2.)),
                        dtype=tf.float32)
        self.Ua = tf.Variable(tf.random.normal(shape=(d_model, 1), stddev=0.1, seed=None), trainable=True, dtype=tf.dtypes.float32)
        
        self.d_others = d_others
        self.d_obj = d_obj
        self.d_model = d_model
        self.num_objs = int(d_others / d_obj)
        self.build(input_shape=[(None, self.d_others), (None, self.num_objs)])
    
    def call(self, x, **kwargs):
        """
        :x[0]: all the other participants in the intersection, [B, d_others]
        :x[1]: padding mask [B, N], 1 for real, 0 for padding

        return:
            :output: [B, d_model]
            :weights: [B, N] all elements \in (0, 1)
        """
        x_others, x_mask = x[0], x[1]
        x_others = self.embedding(tf.reshape(x_others, [-1, self.num_objs, self.d_obj])) # [B, N, d_model]
        assert x_others.shape[1] == x_mask.shape[-1], print(x_others.shape[1], x_mask.shape[-1])
        x_real = x_others * tf.expand_dims(x_mask, axis=-1) # fake tensors are all zeros
        query = tf.reduce_sum(x_real, axis=-2) / tf.expand_dims(tf.reduce_sum(x_mask, axis=-1) + 1e-5, axis=-1)

        logits = tf.squeeze(tf.matmul(
            tf.tanh(tf.expand_dims(self.Uq(query), axis=1) + self.Ur(x_real)),
            self.Ua), axis=-1) # [B, N]
        logits = logits + ((1-x_mask) * -1e9)
        attention_weights = tf.nn.softmax(logits, axis=-1) # (B, N)
        output = tf.squeeze(tf.matmul(tf.expand_dims(attention_weights, axis=1), x_others), axis=-2)

        return output, attention_weights


class PINet(Model):
    def __init__(self, d_others, d_obj, d_model, **kwargs):
        '''
        :d_others: the total dimension of all participants
        :d_obj: the dimension of single participant
        '''
        super(PINet, self).__init__(name=kwargs['name'])
        self.embedding = Conv1D(d_model, kernel_size=1)
        self.d_others = d_others
        self.d_obj = d_obj
        self.d_model = d_model
        self.num_objs = int(d_others / d_obj)
        self.build(input_shape=[(None, self.d_others), (None, self.num_objs)])

    def call(self, x, **kwargs):
        """
        :x[0]: all the other participants in the intersection, [B, d_others]
        :x[1]: padding mask [B, N], 1 for real, 0 for padding

        return:
            :output: [B, d_model]
            :weights: [B, N] all elements \in (0, 1)
        """
        x_others, x_mask = x[0], x[1]
        x_others = self.embedding(tf.reshape(x_others, [-1, self.num_objs, self.d_obj]))  # [B, N, d_model]
        assert x_others.shape[1] == x_mask.shape[-1], print(x_others.shape[1], x_mask.shape[-1])
        logits = 1 / self.num_objs * tf.ones_like(x_mask)
        logits = logits + ((1 - x_mask) * -1e9)
        attention_weights = tf.nn.softmax(logits, axis=-1)  # (B, N)
        output = tf.squeeze(tf.matmul(tf.expand_dims(attention_weights, axis=1), x_others), axis=-2)

        return output, attention_weights


Name2ModelCls = dict(mlp=MLPNet,
                     attn=AttentionNet,
                     pi=PINet,
                     ppo=PPONet,
                     alpha=AlphaModel,)


if __name__ == '__main__':
    pass
