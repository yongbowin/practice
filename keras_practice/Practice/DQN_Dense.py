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

    def _createModel(self):
        model = Sequential()

        model.add(Dense(units=256, activation='relu', input_dim=self.stateCount))
        model.add(Dense(units=256, activation='relu'))
        model.add(Dense(units=self.actionCount, activation='linear'))

        opt = RMSprop(lr=LEARNING_RATE)
        model.compile(loss=huber_loss, optimizer=opt)

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
BATCH_SIZE = 64

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

        # # debug the Q function in poin S
        # if self.steps % 100 == 0:
        #     S = numpy.array([-0.01335408, -0.04600273, -0.00677248, 0.01517507])
        #     pred = agent.brain.predictOne(S)
        #     print(pred[0])
        #     sys.stdout.flush()

        # slowly decrease Epsilon based on our eperience
        self.steps += 1
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)

    def replay(self):
        batch = self.memory.sample(BATCH_SIZE)
        batchLen = len(batch)

        no_state = numpy.zeros(self.stateCount)

        states = numpy.array([o[0] for o in batch])
        states_ = numpy.array([(no_state if o[3] is None else o[3]) for o in batch])

        p = self.brain.predict(states)
        p_ = self.brain.predict(states_, target=True)

        x = numpy.zeros((batchLen, self.stateCount))
        y = numpy.zeros((batchLen, self.actionCount))

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


# -------------------- ENVIRONMENT ---------------------
class Environment:
    def __init__(self):
        # load the tweet vectors data
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
                    s_ = ((topic_item['tweet_list_info'])[num + 1])['tweet_vec']

                # 将样本(s, a, r, s_)添加到memory的samples数组中
                agent.observe((s, a, r, s_))
                # memory回放和改善
                agent.replay()


# -------------------- MAIN ----------------------------
env = Environment()

stateCount = 300
actionCount = 2

agent = Agent(stateCount, actionCount)
randomAgent = RandomAgent(actionCount)

# calculate cost time
start_time = time.time()

count = 0
try:
    # 初始化samples
    while randomAgent.memory.isFull() == False:
        env.run(randomAgent)

    agent.memory.samples = randomAgent.memory.samples
    randomAgent = None

    # while True:
    #     count += 1
    #     env.run(agent)
    #     print("count ======>", count)

    i = 0
    if i < 10:
        env.run(agent)
        i += 1
finally:
    print("count ======>", count)
    agent.brain.model.save("dqn_weights_sequence.h5")
    print("\nDone in %ds" % (time.time() - start_time))