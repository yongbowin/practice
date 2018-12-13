#!~/anaconda3/bin/python3.6
# encoding: utf-8

"""
@version: 0.0.1
@author: Yongbo Wang
@contact: yongbowin@outlook.com
@file: Machine-Learning-Practice - test_lstm.py
@time: 4/18/18 2:14 AM
@description: 
"""
import pickle
import random

import keras
import numpy as np

import yaml
from keras.layers import Masking, LSTM, Dense
from keras.models import model_from_yaml, Sequential
import os

from keras.preprocessing import sequence

maxlen = 14
batch_size = 32


def load_data(data_set):
    # acquire project root path
    PROJECT_PATH = os.path.dirname(os.getcwd())
    pkl_file = open(PROJECT_PATH + '/data/' + data_set, 'rb')
    # The length of data is 56, the type is 'list'
    data_list = pickle.load(pkl_file)

    return data_list


def pre_data():
    all_list = load_data('tweet_vec_stem_topic_list.pkl')

    x_data_2_total = []
    x_data_1_total = []
    x_data_0_total = []

    # for item in train_list:
    for item in all_list:
        for tweet_item in item['tweet_list_info']:

            x_data_2 = []
            x_data_1 = []
            x_data_0 = []

            rel = int(tweet_item['relevance'])

            stem_lower_vec = tweet_item['stem_lower_vec']
            text_len = int(tweet_item['N_word'])

            if rel == 2:
                x_data_2.append(stem_lower_vec[:text_len])
                x_data_2.append(tweet_item['relevance'])
                x_data_2_total.append(x_data_2)
            elif rel == 1:
                x_data_1.append(stem_lower_vec[:text_len])
                x_data_1.append(tweet_item['relevance'])
                x_data_1_total.append(x_data_1)
            elif rel == 0:
                x_data_0.append(stem_lower_vec[:text_len])
                x_data_0.append(tweet_item['relevance'])
                x_data_0_total.append(x_data_0)

    print(len(x_data_2_total))
    print(len(x_data_1_total))
    print(len(x_data_0_total))
    print("**************************")

    # x_test, y_test
    x_test_list = []
    y_test_list = []

    test_2_set = x_data_2_total[321:401]
    test_1_set = x_data_1_total[:80]
    test_0_set = x_data_0_total[3000:3080]

    # test_2_set = x_data_2_total[:320]
    # test_1_set = x_data_1_total[100:740]
    # test_0_set = x_data_0_total[9000:9640]

    # test_2_set = x_data_2_total[321:401]
    # test_1_set = x_data_1_total[:80]
    # test_0_set = x_data_0_total[0:9000]

    test_set = test_2_set
    test_set.extend(test_1_set)
    test_set.extend(test_0_set)

    random.shuffle(test_set)

    for i in test_set:
        x_test_list.append(i[0])
        y_test_list.append(i[1])

    x_test = np.array(x_test_list)
    y_test = np.array(y_test_list)

    x_test = sequence.pad_sequences(x_test, maxlen=maxlen, dtype='float64', padding='post')

    y_test1 = y_test
    y_test = keras.utils.to_categorical(y_test, num_classes=3)

    # print(x_train[0])
    # print(x_train[10][0])

    return x_test, y_test, y_test1


def test_data(x_test):
    PROJECT_PATH = os.path.dirname(os.getcwd())

    print('loading model......')
    with open(PROJECT_PATH + '/data/lstm.yml', 'r') as f:
        yaml_string = yaml.load(f)
    model = model_from_yaml(yaml_string)

    print('loading weights......')
    model.load_weights(PROJECT_PATH + '/data/lstm.h5')
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    # y_pred = model.predict(x_test), 每一个元素代表该位置的某条数据的所属类别
    y_pred = model.predict_classes(x_test)

    return y_pred


x_test, y_test, y_test1 = pre_data()

y_pred = test_data(x_test)

# ------------------------------------------
comm_0 = 0
comm_1 = 0
comm_2 = 0
len_test = len(y_test1)
comm_count = 0
for i in range(len_test):
    if int(y_test1[i]) == y_pred[i]:
        comm_count += 1
        if y_pred[i] == 0:
            comm_0 += 1
        elif y_pred[i] == 1:
            comm_1 += 1
        elif y_pred[i] == 2:
            comm_2 += 1

print("comm_count / pred_test", comm_count, "/", len_test, "||", comm_count, "=", comm_0, "+", comm_1, "+", comm_2)

pred_0 = 0
pred_1 = 0
pred_2 = 0
for ii in y_pred:
    if ii==0:
        pred_0 += 1
    elif ii==1:
        pred_1 += 1
    elif ii==2:
        pred_2 += 1

test_0 = 0
test_1 = 0
test_2 = 0
for i in y_test1:
    ii = int(i)
    if ii==0:
        test_0 += 1
    elif ii==1:
        test_1 += 1
    elif ii==2:
        test_2 += 1

# 预测为1或者2的数据中，标签非0的条数
hit_count = 0
pred_hit = 0
for i in range(len_test):
    if (int(y_test1[i])==1 or int(y_test1[i])==2) and (y_pred[i]==1 or y_pred[i]==2):
        hit_count += 1

    if y_pred[i]==1 or y_pred[i]==2:
        pred_hit += 1

print("y_pred:", pred_0, "+", pred_1, "+", pred_2)
print("y_test1:", test_0, "+", test_1, "+", test_2)
print("hit '0' or '1':", hit_count, "|| hit rate:", float(hit_count) / (test_1 + test_2))
print("accuracy, ('1' or '2') / pred_total:", float(hit_count) / pred_hit)
print("match rate:", float(comm_0) / test_0, "||", float(comm_1) / test_1, "||", float(comm_2) / test_2)

print("y_pred:", y_pred)



