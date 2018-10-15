#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time    : 2018/10/15 17:33
# @Author  : BigDin
# @Contact : dinglei_1107@outlook.com

import tensorflow as tf
import numpy as np


class AutoEncoder:
    def __init__(self, n_layers,
                 transfer_function=tf.nn.softplus,
                 optimizer=tf.train.AdamOptimizer()):
        self.n_layers = n_layers
        self.transfer = transfer_function
        network_weights = self._initialize_weights()
        self.weights = network_weights
        pass

    def _initialize_weights(self):
        all_weights = dict()
        return all_weights

