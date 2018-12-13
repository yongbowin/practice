#!~/anaconda3/bin/python3.6
# encoding: utf-8

"""
@version: 0.0.1
@author: Yongbo Wang
@contact: yongbowin@outlook.com
@file: Keras-Practice - DQN_for_stream_full.py
@time: 4/9/18 12:23 AM
@description: This code demonstrates use a full DQN implementation
"""
import random
import numpy
import math
import sys
import time
import os
import pickle

from keras.models import Sequential
from keras.layers import *
from keras import optimizers
from keras import losses
from keras import backend as K
import tensorflow as tf


# ----------
HUBER_LOSS_DELTA = 1.0
LEARNING_RATE = 0.01


# ----------
def huber_loss(y_true, y_pred):
    err = y_true - y_pred

    cond = K.abs(err) < HUBER_LOSS_DELTA
    L2 = 0.5 * K.square(err)
    L1 = HUBER_LOSS_DELTA * (K.abs(err) - 0.5 * HUBER_LOSS_DELTA)

    loss = tf.where(cond, L2, L1)  # Keras does not cover where function in tensorflow :-(

    return K.mean(loss)


# ************************* Brain *************************
class Brain:
    def __init__(self, stateCount, actionCount):
        self.stateCount = stateCount
        self.actionCount = actionCount

        self.model = self._createModel()
        self.model_ = self._createModel()

    def _createModel(self):
        model = Sequential()

        # LSTM as input layer, output_dim=512=hidden layer size
        model.add(LSTM(512, input_shape=(1, 300), activation='sigmoid'))

        # fully connected layer
        model.add(Dense(512, activation='relu'))

        # fully connected layer
        model.add(Dense(256, activation='relu'))

        # output layer, 用于输出结果的降维
        model.add(Dense(2, activation='linear'))

        # opt = RMSprop(lr=LEARNING_RATE)
        # model.compile(loss=huber_loss, optimizer=opt)
        opt = optimizers.SGD(lr=LEARNING_RATE)
        model.compile(loss=losses.mean_squared_error, optimizer=opt)

        return model

    def train(self, x, y, epochs=1, verbose=0):
        # [array([]), array([]), array([])] ==> array([[], [], []]) as LSTM's input
        x = np.reshape(x, (len(x), 1, 300))
        self.model.fit(x, y, batch_size=64, epochs=epochs, verbose=verbose)

    def predict(self, s, target=False):
        # s.shape == (1,300)
        s = numpy.reshape(s, (s.shape[0], 1, s.shape[1]))
        if target:
            return self.model_.predict(s)
        else:
            return self.model.predict(s)

    def predictOne(self, s, target=False):
        # stateCount = 300, Firstly, (300,)==>(1,300), i.e., [...]==>[[...]]
        # As input for LSTM, (1,300)
        return self.predict(s.reshape(1, self.stateCount), target=target).flatten()

    def updateTargetModel(self):
        self.model_.set_weights(self.model.get_weights())


# ************************* Memory *************************
class Memory:
    '''
    stored as ( s, a, r, s_ )

    store 'experience', ( s, a, r, s_ )
    '''

    samples = []

    def __init__(self, capacity):
        # init memory capacity
        self.capacity = capacity

    # store 'experience' to internal array, ensure that the capacity is enough
    def add(self, sample):
        self.samples.append(sample)

        if len(self.samples) > self.capacity:
            # remove samples[0]
            self.samples.pop(0)

    # return n samples from memory
    def sample(self, n):
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)

    def isFull(self):
        return len(self.samples) >= self.capacity


# ************************* Agent *************************
MEMORY_CAPACITY = 100000
BATCH_SIZE = 64

# discount factor
GAMMA = 0.99

MAX_EPSILON = 1
MIN_EPSILON = 0.01
# speed of decay
LAMBDA = 0.001

UPDATE_TARGET_FREQUENCY = 100


class Agent:
    steps = 0
    epsilon = MAX_EPSILON

    def __init__(self, stateCount, actionCount):
        self.stateCount = stateCount
        self.actionCount = actionCount

        self.brain = Brain(stateCount, actionCount)
        self.memory = Memory(MEMORY_CAPACITY)

    # implement ε-greedy policy, return action 'a'
    def act(self, s):
        # select an action in ε probability, otherwise select the best action from NN return
        # random.random(), [0,1)
        if random.random() < self.epsilon:
            return random.randint(0, self.actionCount - 1)
        else:
            # 返回最大元素的下标
            # print("length:", len(s))
            return numpy.argmax(self.brain.predictOne(s))

    # add a sample to memory
    def observe(self, sample):  # in (s, a, r, s_) format
        self.memory.add(sample)

        if self.steps % UPDATE_TARGET_FREQUENCY == 0:
            # update target network weight=train network weight for every 1000 steps
            self.brain.updateTargetModel()

        # # debug the Q function in poin S
        # if self.steps % 100 == 0:
        #     S = numpy.array([-0.01335408, -0.04600273, -0.00677248, 0.01517507])
        #     pred = agent.brain.predictOne(S)
        #     print(pred[0])
        #     sys.stdout.flush()

        # slowly decrease Epsilon based on our eperience
        self.steps += 1
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)

    # experience replay
    def replay(self):
        # obtain a batch of samples from memory
        # 由于RandomAgent类的存在，所以在执行该类之前samples中已经有充足的样本
        batch = self.memory.sample(BATCH_SIZE)
        batchLen = len(batch)

        no_state = numpy.zeros(self.stateCount)

        # states: state array, states_: the next state array
        states = numpy.array([o[0] for o in batch])
        states_ = numpy.array([(no_state if o[3] is None else o[3]) for o in batch])

        p = self.brain.predict(states)
        p_ = self.brain.predict(states_, target=True)

        # print("======> p:", p)
        # print("======> len(p):", len(p))

        # TODO: x and y
        x = numpy.zeros((batchLen, self.stateCount))
        y = numpy.zeros((batchLen, self.actionCount))

        # iterator all samples, set a proper target for each sample
        for i in range(batchLen):
            o = batch[i]
            s = o[0]; a = o[1]; r = o[2]; s_ = o[3]

            t = p[i]
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + GAMMA * numpy.amax(p_[i])

            x[i] = s
            y[i] = t

        self.brain.train(x, y)


class RandomAgent:
    memory = Memory(MEMORY_CAPACITY)

    def __init__(self, actionCount):
        self.actionCount = actionCount

    def act(self, s):
        return random.randint(0, self.actionCount - 1)

    def observe(self, sample):  # in (s, a, r, s_) format
        self.memory.add(sample)

    def replay(self):
        pass


# ************************* Environment *************************
class Environment:
    def __init__(self):
        # load the tweet vectors data
        self.train_list, self.test_list = self.load_data()

    def load_data(self):
        # acquire project root path
        PROJECT_PATH = os.path.dirname(os.getcwd())

        pkl_file = open(PROJECT_PATH + '/Practice/tweet_vec_stem_topic_list.pkl', 'rb')

        # The length of data is 56, the type is 'list'
        data_list = pickle.load(pkl_file)

        # spilt data, 85% for training, i.e. 48 topics
        train_list = random.sample(data_list, 48)
        # execute (data_list - train_list)
        test_list = [i for i in data_list if i not in train_list]

        # save train dataset
        fp = open(PROJECT_PATH + "/Practice/tweet_vec_list_train.pkl", "wb")
        pickle.dump(train_list, fp, 0)
        fp.close()

        # save test dataset
        f = open(PROJECT_PATH + "/Practice/tweet_vec_list_test.pkl", "wb")
        pickle.dump(test_list, f, 0)
        f.close()

        return train_list, test_list

    # run() function to execute a episode
    def run(self, agent):
        # reset, the type of 's' is <class 'numpy.ndarray'>
        reward_list = []
        cou = 0
        for topic_item in self.train_list:
            cou += 1
            print("count ==============>", cou)
            # total reward 'R', based on 'relevance'
            sum_gain = 0

            # 在动作为a时，获取环境观测（下一时刻的状态信息）、reward、终止信号、当前时刻状态信息

            # each tweet of 'tweet_list_info' ordered by posted time
            tweet_list_len = len(topic_item['tweet_list_info'])

            for num in range(tweet_list_len):
                # The current state 's'
                s = ((topic_item['tweet_list_info'])[num])['tweet_vec']
                # 在状态为s时，决定采取什么动作a
                a = agent.act(s)
                # print("topid ************>", topic_item['topid'])
                r = 0
                # The next state 's_'
                if num == (tweet_list_len - 1):
                    # EG 'r'
                    for rel in range(tweet_list_len):
                        # TODO: calculate the gain of each tweet
                        sum_gain += float(((topic_item['tweet_list_info'])[rel])['relevance']) / 2.0

                    # expected gain 'eg'
                    r = sum_gain / float(tweet_list_len)

                    reward_list.append(r)

                    s_ = None
                else:
                    s_ = ((topic_item['tweet_list_info'])[num+1])['tweet_vec']

                # 将样本(s, a, r, s_)添加到memory的samples数组中
                agent.observe((s, a, r, s_))
                # memory回放和改善
                agent.replay()

        print("reward_list ==============>", reward_list)
        print("reward_list length ==============>", len(reward_list))


# ************************* Main *************************
env = Environment()

stateCount = 300
actionCount = 2

agent = Agent(stateCount, actionCount)
randomAgent = RandomAgent(actionCount)

# calculate cost time
start_time = time.time()

try:
    # 初始化samples
    while randomAgent.memory.isFull() == False:
        env.run(randomAgent)

    agent.memory.samples = randomAgent.memory.samples
    randomAgent = None

    # while True:
    #     env.run(agent)
    env.run(agent)
finally:
    agent.brain.model.save("dqn_weights.h5")
    print("\nDone in %ds" % (time.time() - start_time))
