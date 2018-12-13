#!~/anaconda3/bin/python3.6
# encoding: utf-8

"""
@version: 0.0.1
@author: Yongbo Wang
@license: Apache Licence
@contact: yongbowin@outlook.com
@site: https://github.com/yongbowin
@project: Keras-Practice
@file: lstm.py
@time: 1/22/18 2:08 AM
@description: 
"""
from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import io

path = get_file('nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
text = io.open(path, encoding='utf-8').read().lower()
# text = io.open("test.txt", encoding='utf-8').read().lower()
# print the number of the text,for example,"Hello, boys!" length is 12 chars.
print('corpus length or text length:', len(text))
# print('text:', text)

# set(text) is a set that includes all chars of text,for instance,{'e', ',', 'b', '!', 'o', 's', 'h', 'y', ' ', 'l'}
# list(text): ['h', 'e', 'l', 'l', 'o', ',', ' ', 'b', 'o', 'y', 's', '!']
# list(set(text)): ['h', 'e', 'b', 'l', 'o', 's', ' ', '!', 'y', ',']
# sorted(list(set(text))): [' ', '!', ',', 'b', 'e', 'h', 'l', 'o', 's', 'y']
chars = sorted(list(set(text)))
# print("set(text):", set(text))
# print("list(text):", list(text))
# print("list(set(text)):", list(set(text)))
# print("sorted(list(set(text))):", sorted(list(set(text))))
print('total chars:', len(chars))
# add index for each char
# char_indices: {' ': 0, '!': 1, ',': 2, 'b': 3, 'e': 4, 'h': 5, 'l': 6, 'o': 7, 's': 8, 'y': 9}
# indices_char: {0: ' ', 1: '!', 2: ',', 3: 'b', 4: 'e', 5: 'h', 6: 'l', 7: 'o', 8: 's', 9: 'y'}
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
# print("char_indices:", char_indices)
# print("indices_char:", indices_char)

# cut str from text,the interval is 40
maxlen = 40
# the next starting position is [3],that is "l"
step = 3
# every element's length is maxlen=40
sentences = []
# That is, It's all the 41th element's set.
next_chars = []
for i in range(0, len(text) - maxlen, step):
    # print("i/(len(text) - maxlen):", i, "/", len(text) - maxlen)
    sentences.append(text[i: i + maxlen])
    # print("sentences:", sentences)
    next_chars.append(text[i + maxlen])
    # print("text[i + maxlen]:", text[i + maxlen])
    # print("next_chars:", next_chars)
print('nb sequences:', len(sentences))

print('Vectorization...')
# output is bool matrix,all the elem are "False", maxlen is row, len(chars) is column
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
# The dim is X * Y * Z
# print("x:", x)
# The dim is X * Y
# print("y:", y)
# one-hot encoder in X * Y * Z and X * Y matrix for each elem in sentences
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        # print("char:", char)
        # char_indices, not indices_char
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1
# print("x:", x)
# print("y:", y)
# z = np.zeros((2, 3, 4), dtype=np.bool)
# print("z:", z)

# build the model: a single LSTM
print('Build model...')
model = Sequential()
# batch_size = 128 || input_shape = (40, 57) || x.shape = (200285, 40, 57) || y.shape = (200285, 57)
# input_shape = (input_length, input_dim)
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
# The fully-connected with len(chars) hidden units
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


# Adopt multinomial in order acquire the next index
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch, logs):
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Generating text after Epoch: %d' % epoch)

    start_index = random.randint(0, len(text) - maxlen - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        # print("[start_index]=", start_index, ": sentence:", sentence)
        # print("[start_index]=", start_index)
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)
        # print()
        # print("generated:", generated)

        for i in range(400):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            # print("preds:", preds)
            next_index = sample(preds, diversity)
            # print("next_index:", next_index)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            # print()
            # print("next_char:", next_char)
            # print()
            sys.stdout.flush()
        print()


# invoked at each epoch end
print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

# print("x =========>", x)
# # (200285, 40, 57)
# print("x.shape =========>", x.shape)
# # print("y =========>", y)
# print("y =========>", y[0])
# # (200285, 57)
# print("y.shape =========>", y.shape)

model.fit(x, y,
          batch_size=128,
          epochs=2,
          callbacks=[print_callback])
