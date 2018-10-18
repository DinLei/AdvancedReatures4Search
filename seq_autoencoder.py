#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time    : 2018/10/17 10:18
# @Author  : BigDin
# @Contact : dinglei_1107@outlook.com


import numpy as np
import tensorflow as tf


class SeqAutoEncoder(object):
    def __init__(self, 
                 n_layers, 
                 transfer_function=tf.nn.softplus, 
                 optimizer=tf.train.AdamOptimizer()):
        """
        
        :param n_layers: (30, 50)
        :param transfer_function: `log(exp(features) + 1)`
        :param optimizer: 
        """
        self.n_layers = n_layers
        self.transfer = transfer_function

        network_weights = self._initialize_weights()
        self.weights = network_weights

        # model
        self.input_x = tf.placeholder(tf.float32, [None, self.n_layers[0]])
        self.hidden_encode = []
        self.hidden_recon = []
        h = self.input_x

        with tf.name_scope("encoding"):
            for layer in range(len(self.n_layers) - 1):
                h = self.transfer(
                    tf.add(tf.matmul(h, self.weights['encode'][layer]['w']),
                           self.weights['encode'][layer]['b']))
                self.hidden_encode.append(h)

        with tf.name_scope("decoding"):
            for layer in range(len(self.n_layers) - 1):
                h = self.transfer(
                    tf.add(tf.matmul(h, self.weights['recon'][layer]['w']),
                           self.weights['recon'][layer]['b']))
                self.hidden_recon.append(h)
        self.reconstruction = self.hidden_recon[-1]

        # cost
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.input_x), 2.0))
        self.optimizer = optimizer.minimize(self.cost)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_weights(self):
        all_weights = dict()
        initializer = tf.contrib.layers.xavier_initializer()
        # Encoding network weights
        encoder_weights = []
        for layer in range(len(self.n_layers) - 1):
            w = tf.Variable(
                initializer((self.n_layers[layer], self.n_layers[layer + 1]),
                            dtype=tf.float32))
            b = tf.Variable(
                tf.zeros([self.n_layers[layer + 1]], dtype=tf.float32))
            encoder_weights.append({'w': w, 'b': b})
        # Recon network weights
        recon_weights = []
        for layer in range(len(self.n_layers) - 1, 0, -1):
            w = tf.Variable(
                initializer((self.n_layers[layer], self.n_layers[layer - 1]),
                            dtype=tf.float32))
            b = tf.Variable(
                tf.zeros([self.n_layers[layer - 1]], dtype=tf.float32))
            recon_weights.append({'w': w, 'b': b})
        all_weights['encode'] = encoder_weights
        all_weights['recon'] = recon_weights
        return all_weights

    def partial_fit(self, input_x):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.input_x: input_x})
        return cost

    def calc_total_cost(self, input_x):
        return self.sess.run(self.cost, feed_dict={self.input_x: input_x})

    def transform(self, input_x):
        return self.sess.run(self.hidden_encode[-1], feed_dict={self.input_x: input_x})

    def generate(self, hidden=None):
        if hidden is None:
            hidden = np.random.normal(size=self.weights['encode'][-1]['b'])
        return self.sess.run(self.reconstruction, feed_dict={self.hidden_encode[-1]: hidden})

    def reconstruct(self, input_x):
        return self.sess.run(self.reconstruction, feed_dict={self.input_x: input_x})

    def get_weights(self):
        # raise NotImplementedError
        return self.sess.run(self.weights)

    def get_biases(self):
        # raise NotImplementedError
        return self.sess.run(self.weights)
