#!~/anaconda3/bin/python3.6
# encoding: utf-8

"""
@version: 0.0.1
@author: Yongbo Wang
@contact: yongbowin@outlook.com
@file: Keras-Practice - DQN_Dense.py
@time: 4/18/18 10:28 PM
@description: 
"""
import random, numpy, math
import pickle
import os

import time
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from keras import backend as K

import tensorflow as tf


# ----------
HUBER_LOSS_DELTA = 1.0
LEARNING_RATE = 0.00025


# ----------
def huber_loss(y_true, y_pred):
    err = y_true - y_pred

    cond = K.abs(err) < HUBER_LOSS_DELTA
    L2 = 0.5 * K.square(err)
    L1 = HUBER_LOSS_DELTA * (K.abs(err) - 0.5 * HUBER_LOSS_DELTA)

    loss = tf.where(cond, L2, L1)  # Keras does not cover where function in tensorflow :-(

    return K.mean(loss)


# -------------------- BRAIN ---------------------------
class Brain:
    def __init__(self, stateCount, actionCount):
        self.stateCount = stateCount
        self.actionCount = actionCount

        self.model = self._createModel()
        self.model_ = self._createModel()

        self.model.load_weights("dqn_weights_sequence.h5")

    def _createModel(self):
        model = Sequential()

        model.add(Dense(units=256, activation='relu', input_dim=self.stateCount))
        model.add(Dense(units=256, activation='relu'))
        model.add(Dense(units=self.actionCount, activation='linear'))

        opt = RMSprop(lr=LEARNING_RATE)
        model.compile(loss=huber_loss, optimizer=opt)

        return model

    def predict(self, s, target=False):
        if target:
            return self.model_.predict(s)
        else:
            return self.model.predict(s)

    def predictOne(self, s, target=False):
        return self.predict(s.reshape(1, self.stateCount), target=target).flatten()


# -------------------- AGENT ---------------------------
MEMORY_CAPACITY = 100000
BATCH_SIZE = 64

GAMMA = 0.99

MAX_EPSILON = 1
MIN_EPSILON = 0.01
LAMBDA = 0.001  # speed of decay


# -------------------- ENVIRONMENT ---------------------
class Environment:
    def __init__(self):
        # load the tweet vectors data
        self.train_list = self.load_data()

    def load_data(self):
        # acquire project root path
        PROJECT_PATH = os.path.dirname(os.getcwd())

        # pkl_file = open(PROJECT_PATH + '/Practice/tweet_vec_list_test.pkl', 'rb')
        pkl_file = open(PROJECT_PATH + '/Practice/tweet_vec_list_train.pkl', 'rb')

        # The length of data is 56, the type is 'list'
        train_list = pickle.load(pkl_file)

        return train_list

    def run(self, brain):
        hit_rate = []
        acurr_rate = []
        # reset, the type of 's' is <class 'numpy.ndarray'>
        for topic_item in self.train_list:
            count_1 = 0
            count_2 = 0
            push_list = []
            rel = 0

            # ['MB254', 'MB425', 'MB419', 'RTS19'], reward = 0
            if topic_item['topid'] == 'MB254' \
                    or topic_item['topid'] == 'MB425' \
                    or topic_item['topid'] == 'MB419' \
                    or topic_item['topid'] == 'RTS19':
                continue

            # each tweet of 'tweet_list_info' ordered by posted time
            tweet_list_len = len(topic_item['tweet_list_info'])

            for num in range(tweet_list_len):
                # The current state 's'
                stem_lower_vec = ((topic_item['tweet_list_info'])[num])['stem_lower_vec']

                # 计算出tweet中所有词的向量表示 的 和的平均，作为该tweet的向量表示
                ii = np.zeros([300])
                count_i = 0
                for i in stem_lower_vec:
                    count_i += 1
                    ii += i

                s = ii / float(count_i)

                r = float(((topic_item['tweet_list_info'])[num])['relevance']) / 2

                pred = brain.predictOne(s)
                # print(pred[0])

                tweet_id = ((topic_item['tweet_list_info'])[num])['tweet_id']
                relevance = ((topic_item['tweet_list_info'])[num])['relevance']
                # print(
                #     "topid ******>", topic_item['topid'],
                #     "|| tweet_id:", tweet_id,
                #     "|| pred:", pred,
                #     "|| max index:", numpy.argmax(pred),
                #     "|| relevance:", relevance
                # )

                push_dict = {}
                if numpy.argmax(pred) == 0:
                    push_dict['tweet_id'] = tweet_id
                    push_dict['relevance'] = relevance
                    if int(relevance) != 0:
                        rel += 1
                    push_list.append(push_dict)

                if int(relevance) != 0:
                    count_2 += 1

                if numpy.argmax(pred) == 0 and int(relevance) != 0:
                    count_1 += 1

            print("hit rate:", float(count_1) / count_2)
            print(topic_item['topid'], "===>", push_list)
            print("push length:", len(push_list), "|| not 0:", rel)

            hit_rate.append(float(count_1) / count_2)
            acurr_rate.append(float(rel) / len(push_list))

        print("hit_rate:", hit_rate)
        print("acurr_rate:", acurr_rate)


# -------------------- MAIN ----------------------------
env = Environment()

stateCount = 300
actionCount = 2

brain = Brain(stateCount, actionCount)

# calculate cost time
start_time = time.time()

count = 0
try:
    # while True:
    #     count += 1
    #     env.run(agent)
    #     print("count ======>", count)

    i = 0
    if i < 1:
        env.run(brain)
        i += 1
finally:
    print("\nDone in %ds" % (time.time() - start_time))