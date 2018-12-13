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

import keras
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import StandardScaler

max_features = 20000
maxlen = 80  # cut texts after this number of words (among top max_features most common words)
batch_size = 32


x_train = np.array([
    [2, 1, 1],
    [1, 2, 1.5],
    [1, 0, 2],
    [1.5, 1, 0],
    [0, 1, 0.5],
    [1, 0, 3]
])
y_train = np.array([
    [0],
    [1],
    [2],
    [0],
    [1],
    [2]
])

x_test = np.array([
    [5, 0.1, 0.2],
    [0.3, 1, 0.5],
    [0.5, 1, 2]
])
y_test = np.array([
    [0],
    [1],
    [2]
])

# 标准化数据
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.fit_transform(x_test)

# 将数据转为LSTM的输入格式
x_train = np.reshape(x_train, (len(x_train), 1, 3))
x_test = np.reshape(x_test, (len(x_test), 1, 3))

# 将标签转化为类似于one-hot编码的格式
y_train = keras.utils.to_categorical(y_train, num_classes=3)
y_test = keras.utils.to_categorical(y_test, num_classes=3)

print('Build model...')
model = Sequential()
# model.add(Embedding(max_features, 128))
# model.add(LSTM(256, dropout=0.2, recurrent_dropout=0.2))
model.add(LSTM(512, input_shape=(1, 3), activation='tanh'))
# model.add(Dense(1, activation='sigmoid'))
model.add(Dense(3, activation='softmax'))

# try using different optimizers and different optimizer configs
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
# epochs之前为1的时候效果不好，当适当增大时分类效果变好
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=2000,
          validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

# y_pred = model.predict(x_test), 每一个元素代表该位置的某条数据的所属类别
y_pred = model.predict_classes(x_test)
print("y_pred ==>", y_pred)
print("y_pred length ==>", len(y_pred))

