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

from keras import losses
from keras.optimizers import *
from keras import backend as K

import tensorflow as tf

# ----------
from keras.preprocessing import sequence

HUBER_LOSS_DELTA = 1.0
LEARNING_RATE = 0.00025

MAX_LEN = 10


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

    def _createModel(self):
        model = Sequential()

        model.add(LSTM(512, input_shape=(MAX_LEN, self.stateCount), activation='sigmoid'))
        model.add(Dropout(0.2))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(2, activation='linear'))

        # opt = optimizers.SGD(lr=LEARNING_RATE)
        opt = RMSprop(lr=LEARNING_RATE)
        model.compile(loss=losses.mean_squared_error, optimizer=opt)

        return model

    def train(self, x, y, epochs=1, verbose=2):
        self.model.fit(x, y, batch_size=64, epochs=epochs, verbose=verbose)

    def predict(self, s, target=False):
        if target:
            return self.model_.predict(s)
        else:
            return self.model.predict(s)

    def predictOne(self, s, target=False):
        return self.predict(s.reshape(1, self.stateCount), target=target).flatten()

    def updateTargetModel(self):
        self.model_.set_weights(self.model.get_weights())


# -------------------- MEMORY --------------------------
class Memory:  # stored as ( s, a, r, s_ )
    samples = []

    def __init__(self, capacity):
        self.capacity = capacity

    def add(self, sample):
        self.samples.append(sample)

        if len(self.samples) > self.capacity:
            self.samples.pop(0)

    def sample(self, n):
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)

    def isFull(self):
        return len(self.samples) >= self.capacity


# -------------------- AGENT ---------------------------
MEMORY_CAPACITY = 100000
# BATCH_SIZE = 70
SAMPLE_SIZE = 70

GAMMA = 0.99

MAX_EPSILON = 1
MIN_EPSILON = 0.01
LAMBDA = 0.001  # speed of decay

UPDATE_TARGET_FREQUENCY = 100


class Agent:
    steps = 0
    epsilon = MAX_EPSILON

    def __init__(self, stateCount, actionCount):
        self.stateCount = stateCount
        self.actionCount = actionCount

        self.brain = Brain(stateCount, actionCount)
        self.memory = Memory(MEMORY_CAPACITY)

    def act(self, s):
        if random.random() < self.epsilon:
            return random.randint(0, self.actionCount - 1)
        else:
            return numpy.argmax(self.brain.predictOne(s))

    def observe(self, sample):  # in (s, a, r, s_) format
        self.memory.add(sample)

        if self.steps % UPDATE_TARGET_FREQUENCY == 0:
            self.brain.updateTargetModel()

        # slowly decrease Epsilon based on our eperience
        self.steps += 1
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)

    def replay(self):
        # 每次从存储池中采样一批数据进行训练 samples = [..., ...]
        sample_batch = self.memory.sample(SAMPLE_SIZE)

        no_state = numpy.zeros(self.stateCount)

        states = numpy.array([o[0] for o in sample_batch])
        states_ = numpy.array([(no_state if o[3] is None else o[3]) for o in sample_batch])

        print("-------------------111------------------------")
        print(states[0].shape)

        states = sequence.pad_sequences(states, maxlen=MAX_LEN, dtype='float64', padding='post')
        states_ = sequence.pad_sequences(states_, maxlen=MAX_LEN, dtype='float64', padding='post')

        print("-------------------222------------------------")
        print(states[0].shape)

        samp_nums = int(SAMPLE_SIZE/MAX_LEN)
        # LSTM （样本数，输入序列的长度，元素维度） == (7, 10, 300) == (int(SAMPLE_SIZE/MAX_LEN) , MAX_LEN, self.stateCount)
        states = np.reshape(states, (samp_nums, MAX_LEN, self.stateCount))
        states_ = np.reshape(states_, (samp_nums, MAX_LEN, self.stateCount))

        p = self.brain.predict(states)
        p_ = self.brain.predict(states_, target=True)

        x = numpy.zeros((SAMPLE_SIZE, self.stateCount))
        y = numpy.zeros((SAMPLE_SIZE, self.actionCount))

        for i in range(samp_nums):
            r_sum = 0
            for o in sample_batch[(i*MAX_LEN):((i+1)*MAX_LEN)]:
                # o = batch[i]
                s = o[0]; a = o[1]; r = o[2]; s_ = o[3]
                r_sum += r

                if s_ is None:
                    t[a] = r
                else:
                    t[a] = r + GAMMA * numpy.amax(p_[i])

            t = p[i]



            # if s_ is None:
            #     t[a] = r
            # else:
            #     t[a] = r + GAMMA * numpy.amax(p_[i])

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


# -------------------- ENVIRONMENT ---------------------
class Environment:
    def __init__(self, actionCount):
        # load the tweet vectors data
        self.actionCount = actionCount
        self.train_list = self.load_data()

    def load_data(self):
        # acquire project root path
        PROJECT_PATH = os.path.dirname(os.getcwd())

        pkl_file = open(PROJECT_PATH + '/Practice/tweet_vec_list_train.pkl', 'rb')

        # The length of data is 56, the type is 'list'
        train_list = pickle.load(pkl_file)

        return train_list

    def run(self, agent):
        # reset, the type of 's' is <class 'numpy.ndarray'>
        for topic_item in self.train_list:
            # ['MB254', 'MB425', 'MB419', 'RTS19'], reward = 0
            if topic_item['topid'] == 'MB254' \
                    or topic_item['topid'] == 'MB425' \
                    or topic_item['topid'] == 'MB419' \
                    or topic_item['topid'] == 'RTS19':
                continue

            # each tweet of 'tweet_list_info' ordered by posted time
            tweet_list_len = len(topic_item['tweet_list_info'])

            # ----------------------------------------------
            # 初始化samples
            memory = Memory(MEMORY_CAPACITY)
            # randomAgent = RandomAgent(self.actionCount)
            while memory.isFull() == False:
                for num in range(tweet_list_len):
                    stem_lower_vec = ((topic_item['tweet_list_info'])[num])['stem_lower_vec']
                    # 计算出tweet中所有词的向量表示 的 和的平均，作为该tweet的向量表示
                    ii = np.zeros([300])
                    count_i = 0
                    for i in stem_lower_vec:
                        count_i += 1
                        ii += i
                    s = ii / float(count_i)
                    a = random.randint(0, self.actionCount - 1)
                    r = float(((topic_item['tweet_list_info'])[num])['relevance']) / 2
                    if num == (tweet_list_len - 1):
                        s_ = None
                    else:
                        stem_lower_vec_ = ((topic_item['tweet_list_info'])[num + 1])['stem_lower_vec']
                        jj = np.zeros([300])
                        count_j = 0
                        for j in stem_lower_vec_:
                            count_j += 1
                            jj += j
                        s_ = jj / float(count_j)
                    memory.add((s, a, r, s_))
            agent.memory.samples = memory.samples
            # ----------------------------------------------

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

                # 在状态为s时，决定采取什么动作a
                a = agent.act(s)

                r = float(((topic_item['tweet_list_info'])[num])['relevance']) / 2

                # The next state 's_'
                if num == (tweet_list_len - 1):
                    s_ = None
                else:
                    stem_lower_vec_ = ((topic_item['tweet_list_info'])[num + 1])['stem_lower_vec']
                    jj = np.zeros([300])
                    count_j = 0
                    for j in stem_lower_vec_:
                        count_j += 1
                        jj += j
                    s_ = jj / float(count_j)

                # 将样本(s, a, r, s_)添加到memory的samples数组中
                agent.observe((s, a, r, s_))
                # memory回放和改善
                agent.replay()


# -------------------- MAIN ----------------------------
stateCount = 300
actionCount = 2

env = Environment(actionCount)

agent = Agent(stateCount, actionCount)
# randomAgent = RandomAgent(actionCount)

# calculate cost time
start_time = time.time()

count = 0
try:
    # 初始化samples
    # while randomAgent.memory.isFull() == False:
    #     env.run(randomAgent)
    #
    # agent.memory.samples = randomAgent.memory.samples
    # randomAgent = None

    # while True:
    #     count += 1
    #     env.run(agent)
    #     print("count ======>", count)

    i = 0
    if i < 1:
        env.run(agent)
        i += 1
finally:
    print("count ======>", count)
    agent.brain.model.save("dqn_weights_sequence_lstm.h5")
    print("\nDone in %ds" % (time.time() - start_time))