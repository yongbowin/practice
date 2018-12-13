#!~/anaconda3/bin/python3.6
# encoding: utf-8

"""
@version: 0.0.1
@author: Yongbo Wang
@contact: yongbowin@outlook.com
@file: Machine-Learning-Practice - tweet_classification_lstm.py
@time: 4/4/18 6:39 PM
@description: 
"""
from __future__ import print_function

import pickle
import random

import keras
import numpy as np

import os

import yaml
from keras import optimizers
from keras.models import Sequential, model_from_yaml
from keras.layers import Dense, Masking, Activation, Dropout
from keras.layers import LSTM
from keras.preprocessing import sequence

maxlen = 38  # cut texts after this number of words (among top max_features most common words)
batch_size = 32


def load_data1(data_set):
    # acquire project root path
    PROJECT_PATH = os.path.dirname(os.getcwd())

    pkl_file = open(PROJECT_PATH + '/data/' + data_set, 'rb')
    # pkl_file = open(PROJECT_PATH + '/Practice/tweet_vec.pkl', 'rb')

    # The length of data is 56, the type is 'list'
    data_list = pickle.load(pkl_file)

    # spilt data, 85% for training, i.e. 48 topics
    train_list = random.sample(data_list, 48)
    # execute (data_list - train_list)
    test_list = [i for i in data_list if i not in train_list]

    # # save train dataset
    # fp = open(PROJECT_PATH + "/Practice/tweet_vec_train.pkl", "wb")
    # pickle.dump(train_list, fp, 0)
    # fp.close()
    #
    # # save test dataset
    # f = open(PROJECT_PATH + "/Practice/tweet_vec_test.pkl", "wb")
    # pickle.dump(test_list, f, 0)
    # f.close()

    return train_list, test_list


def load_data(data_set):
    # acquire project root path
    PROJECT_PATH = os.path.dirname(os.getcwd())
    pkl_file = open(PROJECT_PATH + '/data/' + data_set, 'rb')
    # The length of data is 56, the type is 'list'
    data_list = pickle.load(pkl_file)

    return data_list


def pre_data():
    # train_list, test_list = load_data1('tweet_vec_stem_topic_list.pkl')
    all_list = load_data('tweet_vec_stem_topic_list.pkl')
    # tra_list = load_data('tweet_vec_stem_topic_list_train1.pkl')
    # tes_list = load_data('tweet_vec_stem_topic_list_test1.pkl')

    # x_train, y_train
    x_train_list = []
    y_train_list = []

    x_data_2_total = []
    x_data_1_total = []
    x_data_0_total = []

    x_train_samp_count = 0
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

            # x_train_list.append(stem_lower_vec[:text_len])
            # y_train_list.append(tweet_item['relevance'])

    print(len(x_data_2_total))
    print(len(x_data_1_total))
    print(len(x_data_0_total))
    print("**************************")

    # train_2_set = random.sample(x_data_2_total, 320)
    # train_1_set = random.sample(x_data_1_total, 320)
    # train_0_set = random.sample(x_data_0_total, 320)

    # print("x_data_2_total:", x_data_2_total[0])
    # print("x_data_2_total:", x_data_2_total[10])

    train_2_set = x_data_2_total[:320]
    train_1_set = x_data_1_total[100:740]
    train_0_set = x_data_0_total[9000:9640]

    train_set = train_2_set
    train_set.extend(train_2_set)
    train_set.extend(train_1_set)
    train_set.extend(train_0_set)

    random.shuffle(train_set)

    for i in train_set:
        x_train_list.append(i[0])
        y_train_list.append(i[1])

    x_train = np.array(x_train_list)
    y_train = np.array(y_train_list)

    # x_train = x_train.reshape(x_train.shape[0], 1)

    x_train = np.array(x_train_list)
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen, dtype='float64', padding='post')
    print("x_train:", x_train.shape)

    # x_test, y_test
    x_test_list = []
    y_test_list = []

    test_2_set = x_data_2_total[321:401]
    test_1_set = x_data_1_total[:80]
    test_0_set = x_data_0_total[3000:3080]

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
    y_train1 = y_train
    # print("y_test1 ***>>>", y_test1)

    y_train = keras.utils.to_categorical(y_train, num_classes=3)
    y_test = keras.utils.to_categorical(y_test, num_classes=3)

    # print(x_train[0])
    # print(x_train[10][0])

    return x_train, x_test, y_train, y_test, y_test1, y_train1


def train_data(x_train, x_test, y_train, y_test):
    print('Build model...')
    model = Sequential()
    model.add(Masking(mask_value=0, input_shape=(maxlen, 300)))

    # model.add(LSTM(512, input_shape=(maxlen, 300), activation='sigmoid'))
    model.add(LSTM(512, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    # model.add(Dense(256, activation='relu'))
    # model.add(Dense(1, activation='sigmoid'))
    model.add(Dense(3, activation='softmax'))
    # model.add(Activation('softmax'))

    # opt = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    opt = optimizers.Adam(lr=0.01)

    # try using different optimizers and different optimizer configs
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    print('Train...')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              # epochs=1, validation_data=(x_test, y_test))
              epochs=150)
    score, acc = model.evaluate(x_test, y_test,
                                batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)

    PROJECT_PATH = os.path.dirname(os.getcwd())
    yaml_string = model.to_yaml()
    with open(PROJECT_PATH + '/data/lstm.yml', 'w') as outfile:
        outfile.write(yaml.dump(yaml_string, default_flow_style=True))
    model.save_weights(PROJECT_PATH + '/data/lstm.h5')


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


x_train, x_test, y_train, y_test, y_test1, y_train1 = pre_data()

train_data(x_train, x_test, y_train, y_test)

y_pred = test_data(x_test)

# ------------------------------------------
comm_0 = 0
comm_1 = 0
comm_2 = 0
len_test = len(y_test1)
comm_count = 0
no_zero = 0
rel_no_zero = 0
for i in range(len_test):
    if int(y_test1[i]) != 0 and y_pred[i] != 0:
        rel_no_zero += 1

    if int(y_test1[i]) != 0:
        no_zero += 1

    if int(y_test1[i]) == y_pred[i]:
        comm_count += 1
        if y_pred[i] == 0:
            comm_0 += 1
        elif y_pred[i] == 1:
            comm_1 += 1
        elif y_pred[i] == 2:
            comm_2 += 1

print("comm_count / pred_test", comm_count, "/", len_test, "||", comm_count, "=", comm_0, "+", comm_1, "+", comm_2)

print("rel_no_zero / no_zero:", rel_no_zero, "/", no_zero, "|| rete:", float(rel_no_zero) / no_zero)

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


print("y_pred:", pred_0, "+", pred_1, "+", pred_2)
print("y_test1:", test_0, "+", test_1, "+", test_2)
print("match rate:", float(comm_0) / test_0, "||", float(comm_1) / test_1, "||", float(comm_2) / test_2)

print("y_pred:", y_pred)

train_0 = 0
train_1 = 0
train_2 = 0
for i in y_train1:
    ii = int(i)
    if ii==0:
        train_0 += 1
    elif ii==1:
        train_1 += 1
    elif ii==2:
        train_2 += 1

print("x_train:", train_0, "+", train_1, "+", train_2)
