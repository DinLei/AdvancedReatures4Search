#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time    : 2018/10/15 17:27
# @Author  : BigDin
# @Contact : dinglei_1107@outlook.com

import datetime
from tf_config import *
from data_helper import *
from conv_autoencoder import ConvAutoEncoder


def pre_process():
    print("Loading data...")
    data = time_series_gene(
        c_dim=FLAGS.time_len,
        data_file=FLAGS.data_file)

    keys = []
    records = []
    for ele in data:
        keys.append(ele[0])
        records.append(ele[1])
    keys = np.array(keys)
    records = np.array(records)

    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(keys)))
    x_shuffled = records[shuffle_indices]
    k_shuffled = keys[shuffle_indices]

    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(keys)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    k_train, k_dev = k_shuffled[:dev_sample_index], k_shuffled[dev_sample_index:]

    print("Train/Dev split: {:d}/{:d}".format(len(k_train), len(k_dev)))
    return x_train, k_train, x_dev, k_dev


def train(x_train, x_dev):
    avg_cost = 0.
    display_step = 0
    n_samples = len(x_train)

    cae = ConvAutoEncoder(
        x_shape=x_shape,
        filter_shape=filter_shape,
        pool_shape=pool_shape,
        num_filters=list(
            map(int, FLAGS.num_filters.split(","))
        )
    )

    batches = batch_iter(
        data=x_train,
        batch_size=FLAGS.batch_size,
        num_epochs=FLAGS.num_epochs
    )

    for batch in batches:
        display_step += 1
        cost = cae.partial_fit(batch)
        avg_cost += cost / n_samples * FLAGS.batch_size
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, avg_cost: {:g}".format(
            time_str, display_step, cost)
        )
    print("Total cost: " + str(cae.calc_total_cost(x_dev)))


def main(argv=None):
    x_train, k_train, x_dev, k_dev = pre_process()
    train(x_train, x_dev)


if __name__ == '__main__':
    tf.app.run()
