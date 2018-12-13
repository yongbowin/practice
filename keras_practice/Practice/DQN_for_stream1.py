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

'''
Architecture Graph:(4 classes)

1.Environment
    run()  # runs one episode

2.Agent
    act(s)  # decides what action to take in state s
    observe(sample)  # adds sample (s, a, r, s_) to memory
    replay()  # replays memories and improves

3.Brain
    predict(s)  # predicts the Q function values in state s
    train(batch)  # performs supervised training step with batch

4.Memory
    add(sample)  # adds sample to memory
    sample(n)  # returns random batch of n samples
'''


# ************************* Brain *************************
class Brain:
    def __init__(self, stateCount, actionCount):
        # state count
        self.stateCount = stateCount
        # action count
        self.actionCount = actionCount

        self.model = self._createModel()

        # load trained model (params)
        # self.model.load_weights("dqn_weights.h5f")

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
        # sgd = optimizers.SGD(lr=0.0000005)
        model.compile(loss=losses.mean_squared_error, optimizer=sgd)

        return model

    def train(self, x, y, epoch=1, verbose=2):
        # [array([]), array([]), array([])] ==> array([[], [], []]) as LSTM's input
        x = np.reshape(x, (len(x), 1, 300))
        self.model.fit(x, y, batch_size=64, nb_epoch=epoch, verbose=verbose)

    def predict(self, s):
        # s.shape == (1,300)
        s = numpy.reshape(s, (s.shape[0], 1, s.shape[1]))
        return self.model.predict(s)

    def predictOne(self, s):
        # stateCount = 300, Firstly, (300,)==>(1,300), i.e., [...]==>[[...]]
        # As input for LSTM, (1,300)
        return self.predict(s.reshape(1, self.stateCount)).flatten()


# ************************* Memory *************************
class Memory:
    '''
    store 'experience', ( s, a, r, s_ )
    '''

    samples = []

    def __init__(self, capacity):
        # init memory capacity
        self.capacity = capacity

    # store 'experience' to internal array, ensure that the capacity is enough
    def add(self, sample):
        self.samples.append(sample)

        # print("samples length =====>", len(self.samples))

        if len(self.samples) > self.capacity:
            # remove samples[0]
            self.samples.pop(0)

    # return n samples from memory
    def sample(self, n):
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)

    # return all samples
    def all_samples(self):
        a_samples = self.samples
        self.samples = []

        return a_samples


# ************************* Agent *************************
MEMORY_CAPACITY = 100000
BATCH_SIZE = 64
# discount factor
GAMMA = 1
MAX_EPSILON = 1
MIN_EPSILON = 0.01
# speed of decay
LAMBDA = 0.001


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
            return numpy.argmax(self.brain.predictOne(s))

    # add a sample to memory
    def observe(self, sample):  # in (s, a, r, s_) format
        self.memory.add(sample)

        # slowly decrease Epsilon based on our eperience
        self.steps += 1
        # decrease epsilon with time going, the decay rate is LAMBDA
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)

    # experience replay
    def replay(self):
        # obtain a batch of samples from memory
        batch = self.memory.sample(BATCH_SIZE)
        batchLen = len(batch)

        no_state = numpy.zeros(self.stateCount)

        # state array
        states = numpy.array([o[0] for o in batch])
        # the next state array
        states_ = numpy.array([(no_state if o[3] is None else o[3]) for o in batch])

        p = self.brain.predict(states)
        p_ = self.brain.predict(states_)

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

    # get all samples (mid-function)
    def get_all_samples(self):
        all_samples = self.memory.all_samples()
        return all_samples


# ************************* Environment *************************
class Environment:
    def __init__(self):
        # load the tweet vectors data
        self.train_list, self.test_list = self.load_data()

    def load_data(self):
        # acquire project root path
        PROJECT_PATH = os.path.dirname(os.getcwd())

        pkl_file = open(PROJECT_PATH + '/Practice/tweet_vec.pkl', 'rb')

        # The length of data is 56, the type is 'list'
        data_list = pickle.load(pkl_file)

        # spilt data, 85% for training, i.e. 48 topics
        train_list = random.sample(data_list, 48)
        # execute (data_list - train_list)
        test_list = [i for i in data_list if i not in train_list]

        # save train dataset
        fp = open(PROJECT_PATH + "/Practice/tweet_vec_train.pkl", "wb")
        pickle.dump(train_list, fp, 0)
        fp.close()

        # save test dataset
        f = open(PROJECT_PATH + "/Practice/tweet_vec_test.pkl", "wb")
        pickle.dump(test_list, f, 0)
        f.close()

        return train_list, test_list

    # run() function to execute a episode
    def run(self, agent):
        # reset, the type of 's' is <class 'numpy.ndarray'>
        reward_list = []
        for topic_item in self.train_list:
            # remove 'MB229'
            if topic_item['topid'] == 'MB229' \
                    or topic_item['topid'] == 'RTS13' \
                    or topic_item['topid'] == 'MB410' \
                    or topic_item['topid'] == 'MB265' \
                    or topic_item['topid'] == 'MB414' \
                    or topic_item['topid'] == 'MB239':
                continue

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
                print("topid ************>", topic_item['topid'])
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


# ************************* Environment *************************
env = Environment()

stateCount = 300
actionCount = 2

agent = Agent(stateCount, actionCount)

# calculate cost time
start_time = time.time()

# training, cost: 2433s
try:
    # while True:
    env.run(agent)
finally:
    agent.brain.model.save("dqn_weights.h5f")

print("\nDone in %ds" % (time.time() - start_time))


