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

import keras
import numpy as np

import os
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

max_features = 20000
maxlen = 80  # cut texts after this number of words (among top max_features most common words)
batch_size = 32


def load_data(data_set):
    # acquire project root path
    PROJECT_PATH = os.path.dirname(os.getcwd())
    pkl_file = open(PROJECT_PATH + '/data/' + data_set, 'rb')
    # The length of data is 56, the type is 'list'
    test_list = pickle.load(pkl_file)

    return test_list


train_list = load_data('tweet_vec_train.pkl')
test_list = load_data('tweet_vec_test.pkl')

# x_train, y_train
x_train_list = []
y_train_list = []
for item in train_list:
    if item['topid'] == 'MB229' \
            or item['topid'] == 'RTS13' \
            or item['topid'] == 'MB410' \
            or item['topid'] == 'MB265' \
            or item['topid'] == 'MB414' \
            or item['topid'] == 'MB239':
        continue

    for tweet_item in item['tweet_list_info']:
        x_train_list.append(tweet_item['tweet_vec'])
        y_train_list.append(tweet_item['relevance'])

x_train = np.array(x_train_list)
y_train = np.array(y_train_list)

# x_test, y_test
x_test_list = []
y_test_list = []
for item in test_list:
    for tweet_item in item['tweet_list_info']:
        x_test_list.append(tweet_item['tweet_vec'])
        y_test_list.append(tweet_item['relevance'])

x_test = np.array(x_test_list)
y_test = np.array(y_test_list)
x_train = np.reshape(x_train, (len(x_train), 1, 300))
x_test = np.reshape(x_test, (len(x_test), 1, 300))

y_test1 = y_test
print("y_test1 ***>>>", y_test1)

y_train = keras.utils.to_categorical(y_train, num_classes=3)
y_test = keras.utils.to_categorical(y_test, num_classes=3)

print('Build model...')
model = Sequential()
# model.add(Embedding(max_features, 128))
# model.add(LSTM(256, dropout=0.2, recurrent_dropout=0.2))
model.add(LSTM(512, input_shape=(1, 300), activation='sigmoid'))
# model.add(Dense(1, activation='sigmoid'))
model.add(Dense(3, activation='softmax'))

# try using different optimizers and different optimizer configs
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=1,
          validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

# y_pred = model.predict(x_test), 每一个元素代表该位置的某条数据的所属类别
y_pred = model.predict_classes(x_test)

print("y_test:", y_test)
print("y_test[0]:", y_test[0])
print("y_pred:", y_pred)
print("y_pred[0]:", y_pred[0])

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

print("y_test length ==>", len(y_test))
print("y_pred length ==>", len(y_pred))

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


print("pred:", pred_0, "+", pred_1, "+", pred_2)
print("test:", test_0, "+", test_1, "+", test_2)

