#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time    : 2018/10/16 16:31
# @Author  : BigDin
# @Contact : dinglei_1107@outlook.com

import tensorflow as tf

tf.flags.DEFINE_string("data_file",
                       "data/test_data",
                       # "../data/nlp_products_time_series_dl",
                       "被初步聚合好的商品销售指标时间序列")

# Model Hyper parameters
x_shape = (4, 30)                           # 组织好的商品时间序列维度（二维）
filter_shape = (7, 1)                       # 卷积核的大小
pool_shape = (7, 1)                         # 池化窗口的大小
active_func = tf.nn.relu                    # 激活函数
optimizer = tf.train.AdamOptimizer()        # 优化函数
tf.flags.DEFINE_integer("time_len", x_shape[1], "卷积层数量")
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("num_filters", "64,64,32", "每个卷积层使用的卷积核的个数")
tf.flags.DEFINE_integer("conv_layers", 3, "卷积层数量")
tf.flags.DEFINE_integer("pool_strides", 1, "池化操作步长")
tf.flags.DEFINE_float("learning_rate", 0.001, "学习率的大小")
tf.flags.DEFINE_integer("auto_encoder_units", 100, "自编码的长度")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
