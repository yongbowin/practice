#!~/anaconda3/bin/python3.6
# encoding: utf-8

"""
@version: 0.0.1
@author: Yongbo Wang
@contact: yongbowin@outlook.com
@file: Keras-Practice - DQN_for_stream.py
@time: 3/14/18 9:10 PM
@description:
"""
import random
import numpy
import os
import pickle
import math
import time

from keras.models import Sequential
from keras.layers import *
from keras import losses, optimizers


# ************************* Brain *************************
class Brain:
    def __init__(self, stateCount, actionCount):
        # state count
        self.stateCount = stateCount
        # action count
        self.actionCount = actionCount

        self.model = self._createModel()

        # load trained model (params)
        self.model.load_weights("dqn_weights.h5f")

        # s = numpy.ones(300)
        # p = self.predictOne(s)
        # print("p ++++++++++++++++++++++++++++++++>", p)

    # create neural network
    def _createModel(self):
        model = Sequential()

        # LSTM as input layer, output_dim=512=hidden layer size
        model.add(LSTM(512, input_shape=(1, 300), activation='sigmoid'))

        # fully connected layer
        model.add(Dense(256))
        model.add(Activation('relu'))

        # fully connected layer
        model.add(Dense(256))
        model.add(Activation('relu'))

        # output layer
        # 用于输出结果的降维
        model.add(Dense(2))
        model.add(Activation('linear'))

        # optimizers
        sgd = optimizers.SGD(lr=0.001)
        model.compile(loss=losses.mean_squared_error, optimizer=sgd)

        return model

    def predict(self, s):
        # s.shape == (1,300)
        s = numpy.reshape(s, (s.shape[0], 1, s.shape[1]))

        return self.model.predict(s)

    def predictOne(self, s):
        # stateCount = 300, Firstly, (300,)==>(1,300), i.e., [...]==>[[...]]
        # As input for LSTM, (1,300)
        return self.predict(s.reshape(1, self.stateCount)).flatten()


# ************************* Agent *************************
class Agent:
    def __init__(self, stateCount, actionCount):
        self.stateCount = stateCount
        self.actionCount = actionCount

        self.brain = Brain(stateCount, actionCount)

    # implement ε-greedy policy, return action 'a'
    def act(self, s):
        # select the best action from NN return
        # 返回最大元素的下标

        return numpy.argmax(self.brain.predictOne(s))


# ************************* Environment *************************
class Environment:
    def __init__(self):
        # load the tweet vectors data
        self.test_list = self.load_data()

    def load_data(self):
        # acquire project root path
        PROJECT_PATH = os.path.dirname(os.getcwd())

        pkl_file = open(PROJECT_PATH + '/Practice/tweet_vec_test.pkl', 'rb')

        # The length of data is 56, the type is 'list'
        test_list = pickle.load(pkl_file)

        return test_list

    # run() function to execute a episode
    def run(self, agent):
        # reset, the type of 's' is <class 'numpy.ndarray'>
        count = 0
        for topic_item in self.test_list:
            # if count > 0:
            #     return
            # count += 1

            # 在动作为a时，获取环境观测（下一时刻的状态信息）、reward、终止信号、当前时刻状态信息

            # each tweet of 'tweet_list_info' ordered by posted time
            tweet_list_len = len(topic_item['tweet_list_info'])

            count1 = 0
            count1_1 = 0
            count1_2 = 0
            count2 = 0
            count3 = 0
            count3_1 = 0
            count3_2 = 0
            for num in range(tweet_list_len):
                # The current state 's'
                s = ((topic_item['tweet_list_info'])[num])['tweet_vec']
                # 在状态为s时，决定采取什么动作a
                a = agent.act(s)

                tweet_id = ((topic_item['tweet_list_info'])[num])['tweet_id']
                relevance = ((topic_item['tweet_list_info'])[num])['relevance']

                if relevance != '0':
                    count1 += 1
                    if relevance == '1':
                        count1_1 += 1
                    elif relevance == '2':
                        count1_2 += 1
                if a != 0:
                    count2 += 1
                    if relevance != '0':
                        count3 += 1
                        if relevance == '1':
                            count3_1 += 1
                        elif relevance == '2':
                            count3_2 += 1

                # print(topic_item['topid'], "======>", tweet_id, "rel:", relevance, "===> action:", a)
            print(
                "topid", topic_item['topid'],
                "relevance num:", count1,
                "=", count1_1, "+", count1_2,
                " push num==>", count2,
                "/", tweet_list_len,
                "tweet_rate:", float(count2)/float(tweet_list_len),
                "|| ==> include no-zero:", count3,
                "=", count3_1, "+", count3_2,
                "|| rel_rate:", float(count3)/float(count1)
            )


# ************************* Environment *************************
env = Environment()

stateCount = 300
actionCount = 2

agent = Agent(stateCount, actionCount)

# calculate cost time
start_time = time.time()

# test, cost: xxx
try:
    # while True:
    env.run(agent)
finally:
    print("\nDone in %ds" % (time.time() - start_time))


