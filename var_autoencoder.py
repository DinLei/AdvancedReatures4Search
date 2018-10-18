#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time    : 2018/10/17 11:26
# @Author  : BigDin
# @Contact : dinglei_1107@outlook.com


import tensorflow as tf
from seq_auto_encoder import SeqAutoEncoder


class VariationalAutoEncoder(SeqAutoEncoder):

    def __init__(self,
                 n_layers,
                 optimizer=tf.train.AdamOptimizer()):
        self.n_input = n_layers[0]
        self.n_hidden = n_layers[1]

        network_weights = self._initialize_weights()
        self.weights = network_weights

        # model
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.z_mean = tf.add(tf.matmul(self.x, self.weights['w1']), self.weights['b1'])
        self.z_log_sigma_sq = tf.add(
            tf.matmul(
                self.x, self.weights['log_sigma_w1']
            ),
            self.weights['log_sigma_b1']
        )

        # sample from gaussian distribution
        eps = tf.random_normal(
            tf.stack([tf.shape(self.x)[0], self.n_hidden]), 0, 1, dtype=tf.float32)
        self.z = tf.add(
            self.z_mean,
            tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps)
        )

        self.reconstruction = tf.add(
            tf.matmul(self.z, self.weights['w2']),
            self.weights['b2']
        )

        # cost
        with tf.name_scope("calculate_cost"):
            reconstr_loss = 0.5 * tf.reduce_sum(
                tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
            latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq
                                               - tf.square(self.z_mean)
                                               - tf.exp(self.z_log_sigma_sq), 1)
        self.cost = tf.reduce_mean(reconstr_loss + latent_loss)
        self.optimizer = optimizer.minimize(self.cost)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.get_variable(
            "w1", shape=[self.n_input, self.n_hidden],
            initializer=tf.contrib.layers.xavier_initializer())
        all_weights['log_sigma_w1'] = tf.get_variable(
            "log_sigma_w1",
            shape=[self.n_input, self.n_hidden],
            initializer=tf.contrib.layers.xavier_initializer())
        all_weights['b1'] = tf.Variable(
            tf.zeros([self.n_hidden], dtype=tf.float32))
        all_weights['log_sigma_b1'] = tf.Variable(
            tf.zeros([self.n_hidden], dtype=tf.float32))
        all_weights['w2'] = tf.Variable(
            tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))
        all_weights['b2'] = tf.Variable(
            tf.zeros([self.n_input], dtype=tf.float32))
        return all_weights
