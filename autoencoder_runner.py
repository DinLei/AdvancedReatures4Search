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
        records.append(np.expand_dims(ele[1], -1))
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
    display_step = 1
    n_samples = len(x_train)

    cae = ConvAutoEncoder(
        x_shape=x_shape,
        filter_shape=filter_shape,
        pool_shape=pool_shape,
        num_filters=list(
            map(int, FLAGS.num_filters.split(","))
        )
    )

    num_batches_per_epoch = int((n_samples - 1) / FLAGS.batch_size) + 1
    for epoch in range(FLAGS.num_epochs):
        avg_cost = 0.
        shuffle_indices = np.random.permutation(np.arange(n_samples))
        shuffled_data = x_train[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * FLAGS.batch_size
            end_index = min((batch_num + 1) * FLAGS.batch_size, n_samples)
            batch = shuffled_data[start_index:end_index]
            cost = cae.partial_fit(batch)
            avg_cost += cost / n_samples

        if epoch % display_step == 0:
            time_str = datetime.datetime.now().isoformat()
            print("{} >>> \tepoch {},\tavg_cost {:.6f}".format(
                time_str, epoch+1, avg_cost)
            )
    print("\nThe dev examples average cost: {:.6f}".
          format(cae.calc_total_cost(x_dev)/len(x_dev)))


def main(argv=None):
    x_train, k_train, x_dev, k_dev = pre_process()
    train(x_train, x_dev)


if __name__ == '__main__':
    tf.app.run()
