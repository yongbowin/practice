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
from keras.models import Sequential
from keras.layers import Dense, Masking, Activation, Dropout
from keras.layers import LSTM
from keras.preprocessing import sequence

maxlen = 14  # cut texts after this number of words (among top max_features most common words)
batch_size = 64


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


# train_list, test_list = load_data1('tweet_vec_stem_topic_list.pkl')
all_list = load_data('tweet_vec_stem_topic_list.pkl')
tra_list = load_data('tweet_vec_stem_topic_list_train.pkl')
tes_list = load_data('tweet_vec_stem_topic_list_test.pkl')


# x_train, y_train
x_train_list = []
y_train_list = []
x_train_samp_count = 0
# for item in train_list:
for item in all_list:
# for item in train_list:
    for tweet_item in item['tweet_list_info']:
        if str(tweet_item['tweet_id']) in tra_list:
            x_train_samp_count += 1

            if x_train_samp_count > 50:
                continue

            stem_lower_vec = tweet_item['stem_lower_vec']
            text_len = int(tweet_item['N_word'])

            x_train_list.append(stem_lower_vec[:text_len])
            y_train_list.append(tweet_item['relevance'])


x_train = np.array(x_train_list)
y_train = np.array(y_train_list)

# x_train = x_train.reshape(x_train.shape[0], 1)

x_train = sequence.pad_sequences(x_train, maxlen=maxlen, padding='post')
# print(x_train)
print("x_train:", x_train.shape)

# x_test, y_test
x_test_list = []
y_test_list = []
x_test_samp_count = 0
for item in all_list:
# for item in test_list:
    for tweet_item in item['tweet_list_info']:
        if str(tweet_item['tweet_id']) in tes_list:
            x_test_samp_count += 1

            if x_test_samp_count > 25:
                continue

            stem_lower_vec = tweet_item['stem_lower_vec']
            text_len = int(tweet_item['N_word'])

            x_test_list.append(stem_lower_vec[:text_len])
            y_test_list.append(tweet_item['relevance'])

x_test = np.array(x_test_list)
y_test = np.array(y_test_list)

x_test = sequence.pad_sequences(x_test, maxlen=maxlen, padding='post')

y_test1 = y_test
# print("y_test1 ***>>>", y_test1)

y_train = keras.utils.to_categorical(y_train, num_classes=3)
y_test = keras.utils.to_categorical(y_test, num_classes=3)

print('Build model...')
model = Sequential()
model.add(Masking(mask_value=0, input_shape=(maxlen, 300)))

# model.add(LSTM(512, input_shape=(maxlen, 300), activation='sigmoid'))
model.add(LSTM(512, activation='tanh'))
# model.add(Dropout(0.2))
# model.add(Dense(1, activation='sigmoid'))
model.add(Dense(3, activation='softmax'))
# model.add(Activation('softmax'))

# try using different optimizers and different optimizer configs
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          # epochs=1, validation_data=(x_test, y_test))
          epochs=5000)
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

# y_pred = model.predict(x_test), 每一个元素代表该位置的某条数据的所属类别
y_pred = model.predict_classes(x_test)

print("==================================")
print(y_pred)
print(len(y_pred), len(y_test))
print(type(y_pred))
print("==================================")

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


print("y_pred:", pred_0, "+", pred_1, "+", pred_2)
print("y_test1:", test_0, "+", test_1, "+", test_2)

print(y_pred)