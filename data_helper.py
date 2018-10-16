#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time    : 2018/10/15 16:40
# @Author  : BigDin
# @Contact : dinglei_1107@outlook.com

import os
import csv
import numpy as np


class HiveDataReader(object):

    def __init__(self, delimiter=','):
        self.delimiter = delimiter
        self.null_situation = {"NULL", "null"}

    def __convert(self, data, data_type):
        return [data_type[i](data[i])
                if data[i] not in self.null_situation else None
                for i in range(len(data))]

    def csv_data_generator(self, dir_path, data_type, with_key=True, skip_header=False):
        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)
            if os.path.isfile(file_path):
                with open(file_path, encoding='utf-8') as f:
                    reader = csv.reader(f, delimiter=',')
                    if skip_header:
                        next(reader, None)
                    for row in reader:
                        if len(row) != len(data_type):
                            continue
                        values = self.__convert(row, data_type)
                        if with_key:
                            values.insert(0, None)
                        yield tuple(values)

    def hive_format_generator(self, dir_path, data_type, auto_key=True):
        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)

            if_file_exist = os.path.isfile(file_path)
            if if_file_exist:
                with open(file_path, encoding='utf-8') as reader:
                    for row in reader.readlines():
                        row = row.strip()
                        row = row.split(",")
                        if len(row) != len(data_type):
                            continue
                        try:
                            values = self.__convert(row, data_type)
                            if auto_key:
                                values.insert(0, None)
                        except Exception as e:
                            print(e)
                            print(row)
                            continue
                        yield tuple(values)


def time_series_gene(c_dim=60, data_file="data/nlp_products_time_series_dl"):
    counter = 0
    dt = (int, int, str, float, float, float, float)
    hdr = HiveDataReader()
    raw_data = hdr.hive_format_generator(data_file, dt, auto_key=False)
    last_key = None
    click_uvs = []
    wish_uvs = []
    cart_uvs = []
    order_uvs = []
    for record in raw_data:
        key = tuple(record[:2])
        click, wish, cart, order = record[-4:]
        if key == last_key or last_key is None:
            click_uvs.append(click)
            wish_uvs.append(wish)
            cart_uvs.append(cart)
            order_uvs.append(order)
        else:
            if len(click_uvs) < c_dim:
                click_uvs = [0] * (c_dim - len(click_uvs)) + click_uvs
                wish_uvs = [0] * (c_dim - len(wish_uvs)) + wish_uvs
                cart_uvs = [0] * (c_dim - len(cart_uvs)) + cart_uvs
                order_uvs = [0] * (c_dim - len(order_uvs)) + order_uvs
            yield last_key, np.array([
                click_uvs[-c_dim:], wish_uvs[-c_dim:],
                cart_uvs[-c_dim:], order_uvs[-c_dim:]
            ])
            click_uvs.clear()
            wish_uvs.clear()
            cart_uvs.clear()
            order_uvs.clear()
            if counter % 1000 == 0:
                print("Have completed time series data: {}...".format(counter))
            counter += 1
        last_key = key
    if len(click_uvs) >= c_dim:
        yield last_key, np.array([
                    click_uvs[-c_dim:], wish_uvs[-c_dim:],
                    cart_uvs[-c_dim:], order_uvs[-c_dim:]
                ])
    print("All time series data: {}...".format(counter+1))


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((data_size-1)/batch_size) + 1
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
