#!~/anaconda3/bin/python3.6
# encoding: utf-8

"""
@version: 0.0.1
@author: Yongbo Wang
@contact: yongbowin@outlook.com
@file: Keras-Practice - text_vec_lstm.py
@time: 3/17/18 11:28 AM
@description: 
"""
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras import losses, optimizers

import pickle
import random
import os
import numpy as np


def load_data():
    # acquire project root path
    PROJECT_PATH = os.path.dirname(os.getcwd())

    pkl_file = open(PROJECT_PATH + '/Practice/tweet_vec.pkl', 'rb')

    # The length of data is 56, the type is 'list'
    data_list = pickle.load(pkl_file)

    # spilt data, 85% for training, i.e. 48 topics
    train_list = random.sample(data_list, 48)
    # execute (data_list - train_list)
    test_list = [i for i in data_list if i not in train_list]

    return train_list, test_list


model = Sequential()
# now model.output_shape == (None, 32)
# note: `None` is the batch dimension. LSTM(batch_size, input_shape=(10, 64))
# batch_size = hidden neural numbers
# input_shape = (input_length, input_dim)
# The numbers of hidden layer is 512, i.e., output_dim is 512 || input is (1, 300) for each tweet vector
model.add(LSTM(512, input_shape=(1, 300)))

# fully connected layer
model.add(Dense(256))
model.add(Activation('relu'))

# fully connected layer
model.add(Dense(256))
model.add(Activation('relu'))

# output layer
model.add(Dense(2))
model.add(Activation('linear'))

# optimizers
sgd = optimizers.SGD(lr=0.001)
model.compile(loss=losses.mean_squared_error, optimizer=sgd)


# load data
train_list, test_list = load_data()

count = 0
x_list = []
for item in train_list:
    while count < 10:
        x = ((item['tweet_list_info'])[count])['tweet_vec']
        x_list.append(x)
        count += 1
# [array([]), array([]), array([])] ==> array([[], [], []]) as LSTM's input
x_list = np.reshape(x_list, (10, 1, 300))

y = np.zeros((10, 2), dtype=int)
for i in range(10):
    for j in range(2):
        y[i][j] = random.randint(0, 1)

count_test = 0
x_test = []
for item_test in test_list:
    while count_test < 3:
        x = ((item_test['tweet_list_info'])[count_test])['tweet_vec']
        x_test.append(x)
        count_test += 1
# [array([]), array([]), array([])] ==> array([[], [], []]) as LSTM's input
x_test = np.reshape(x_test, (3, 1, 300))

# print("x_list ==========>", x_list)
# print("x_list.shape ==========>", len(x_list))

# x.shape = (1, 300), x_input.shape = (10, 1, 300)
# y.shape = (10, 2)
# to train
model.fit(x_list, y, batch_size=512, epochs=2000)

# to predict
preds = model.predict(x_test, verbose=2)
print("preds =======================>", preds)

