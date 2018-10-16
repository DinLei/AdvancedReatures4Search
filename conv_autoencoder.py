#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time    : 2018/10/15 17:33
# @Author  : BigDin
# @Contact : dinglei_1107@outlook.com

import tensorflow as tf
import math


class ConvAutoEncoder:
    def __init__(self,
                 x_shape,
                 filter_shape,
                 pool_shape,
                 num_filters,
                 conv_layers=3,
                 pool_strides=1,
                 active_func=tf.nn.relu,
                 learning_rate=0.001,
                 auto_encoder_units=100,
                 optimizer=tf.train.AdamOptimizer()):
        """
        :param x_shape: (4, 60)
        :param filter_shape: (7, 1)
        :param pool_shape: (7, 1)
        :param num_filters: [64, 64, 32]
        :param conv_layers: 3
        :param pool_strides: 1
        :param active_func: 
        :param optimizer: 
        """
        nrow = x_shape[0]
        ncol = x_shape[1]
        self.x_shape = x_shape

        self.filter_shape = filter_shape
        self.pool_shape = pool_shape
        self.num_filters = num_filters
        self.optimizer = optimizer
        self.conv_layers = conv_layers
        self.pool_strides = pool_strides
        self.active_func = active_func
        self.learning_rate = learning_rate

        self.pool_shape_histories = [tuple(x_shape)]

        self.input_x = tf.placeholder(
            tf.float32, [None, nrow, ncol, 1], name="inputs_")
        self.output_y = tf.placeholder(
            tf.float32, [None, nrow, ncol, 1], name="outputs_")

        self.last_hidden = self._encoder(self.input_x)
        self.reconstruction = self._decoder(self.last_hidden)

        self.auto_encoder_vec = tf.layers.dense(
            self.last_hidden, auto_encoder_units,
            activation=active_func,
            name="auto_encoder_vac"
        )

        self.cost = 0.5 * tf.reduce_sum(
            tf.pow(
                tf.subtract(
                    self.reconstruction,
                    self.input_x), 2.0
            ),
            name="cost"
        )
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _encoder(self, input_x):
        image_shape = self.x_shape
        conv = tf.layers.conv2d(
            input_x,
            self.num_filters[0],
            self.filter_shape,
            padding='same',
            activation=self.active_func
        )
        conv = tf.layers.average_pooling2d(
            conv,
            self.pool_shape,
            self.pool_strides,
            padding='same'
        )
        self.pool_shape_histories.append(
            tuple(
                [
                    math.ceil(image_shape[0] / self.pool_shape[0]),
                    math.ceil(image_shape[1] / self.pool_shape[1])
                ]
            )
        )

        if self.conv_layers > 1:
            for i in range(1, self.conv_layers):
                conv = tf.layers.conv2d(
                    conv,
                    self.num_filters[i],
                    self.filter_shape,
                    padding='same',
                    activation=self.active_func
                )
                conv = tf.layers.average_pooling2d(
                    conv,
                    self.pool_shape,
                    self.pool_strides,
                    padding='same'
                )
                self.pool_shape_histories.append(
                    tuple(
                        [
                            math.ceil(image_shape[0] / self.pool_shape[0]),
                            math.ceil(image_shape[1] / self.pool_shape[1])
                        ]
                    )
                )
        return conv

    def _decoder(self, hidden):
        recon = tf.image.resize_nearest_neighbor(
            hidden,
            self.pool_shape_histories[-2]
        )
        recon = tf.layers.conv2d(
            recon,
            self.num_filters[-1],
            self.filter_shape,
            padding='same',
            activation=self.active_func
        )

        if self.conv_layers > 1:
            counter = self.conv_layers-2
            while counter >= 0:
                recon = tf.image.resize_nearest_neighbor(
                    recon,
                    self.pool_shape_histories[counter]
                )
                recon = tf.layers.conv2d(
                    recon,
                    self.num_filters[counter],
                    self.filter_shape,
                    padding='same',
                    activation=self.active_func
                )
                counter -= 1
        return recon

    def partial_fit(self, input_x):
        cost, opt = self.sess.run(
            (self.cost, self.optimizer),
            feed_dict={self.input_x: input_x})
        return cost

    def calc_total_cost(self, input_x):
        return self.sess.run(
            self.cost,
            feed_dict={self.input_x: input_x}
        )

    def transform(self, input_x):
        return self.sess.run(
            self.auto_encoder_vec,
            feed_dict={self.input_x: input_x}
        )
