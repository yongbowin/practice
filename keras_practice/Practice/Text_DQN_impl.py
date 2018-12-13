#!~/anaconda3/bin/python3.6
# encoding: utf-8

"""
@version: 0.0.1
@author: Yongbo Wang
@contact: yongbowin@outlook.com
@file: Keras-Practice - step_to_DQN_impl.py
@time: 3/14/18 1:00 AM
@description: 
"""
import random, numpy, math, gym

# -------------------- BRAIN ---------------------------
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *


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


# -------------------- Brain --------------------------
class Brain:
    def __init__(self, stateCnt, actionCnt):
        # 状态数量
        self.stateCnt = stateCnt
        # 动作数量
        self.actionCnt = actionCnt

        self.model = self._createModel()
        # self.model.load_weights("cartpole-basic.h5")

    # 创建神经网络
    def _createModel(self):
        model = Sequential()

        # 隐含层含有512个神经元，激活函数为ReLU
        model.add(LSTM(512, input_shape=(1, 300)))
        model.add(Dense(output_dim=64, activation='relu', input_dim=stateCnt))
        # 最后一层包含2个神经元（对应2个动作），激活函数为线性的linear
        model.add(Dense(output_dim=actionCnt, activation='linear'))

        # 使用RMSprop算法进行梯度下降
        opt = RMSprop(lr=0.00025)
        # 损失函数：计算均方误差
        model.compile(loss='mse', optimizer=opt)

        return model

    def train(self, x, y, epoch=1, verbose=0):
        x = numpy.reshape(x, (x.shape[0], 1, x.shape[1]))
        # y = numpy.reshape(y, (y.shape[0], 1, y.shape[1]))
        print("x =========>", x)
        print("x.shape =========>", x.shape)
        print("y =========>", y)
        print("y.shape =========>", y.shape)
        # 使用给定的batch大小进行梯度下降
        self.model.fit(x, y, batch_size=64, nb_epoch=epoch, verbose=verbose)

    def predict(self, s):
        # 返回给定状态下Q函数经过神经网络后的预测（数组）
        return self.model.predict(s)

    def predictOne(self, s):
        return self.predict(s.reshape(1, self.stateCnt)).flatten()


# -------------------- MEMORY --------------------------
class Memory:  # stored as ( s, a, r, s_ )
    '''
    Memory类的作用是存储“经验” ( s, a, r, s_ )
    '''

    samples = []

    def __init__(self, capacity):
        # 初始化memory的容量
        self.capacity = capacity

    # 存储“经验”到内部数组，确保不超过容量（限制）
    def add(self, sample):
        self.samples.append(sample)

        if len(self.samples) > self.capacity:
            # 删除list的第一项，即pop(0)
            self.samples.pop(0)

    # 从memory返回n个随机样本
    def sample(self, n):
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)

    # # return all samples
    # def all_samples(self):
    #     a_samples = self.samples
    #     self.samples = []
    #     return a_samples


# -------------------- AGENT ---------------------------
MEMORY_CAPACITY = 100000
BATCH_SIZE = 64

GAMMA = 0.99

MAX_EPSILON = 1
MIN_EPSILON = 0.01
LAMBDA = 0.001  # speed of decay


class Agent:
    steps = 0
    epsilon = MAX_EPSILON

    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.brain = Brain(stateCnt, actionCnt)
        self.memory = Memory(MEMORY_CAPACITY)

    # 实现ε - greedy策略
    def act(self, s):
        # 在概率epsilon下选择一个随机的动作，否则选择当前NN返回的最好的动作
        if random.random() < self.epsilon:
            return random.randint(0, self.actionCnt - 1)
        else:
            return numpy.argmax(self.brain.predictOne(s))

    # 添加一个样本到agent的memory
    def observe(self, sample):  # in (s, a, r, s_) format
        self.memory.add(sample)

        # slowly decrease Epsilon based on our eperience
        self.steps += 1
        # 随时间减小参数epsilon, 衰减速度LAMBDA
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)

    # 经验回放
    def replay(self):
        # 从memory中获取一个batch的样本
        batch = self.memory.sample(BATCH_SIZE)
        batchLen = len(batch)

        no_state = numpy.zeros(self.stateCnt)

        # 状态数组
        states = numpy.array([o[0] for o in batch])
        states_ = numpy.array([(no_state if o[3] is None else o[3]) for o in batch])

        # p中存储的是从开始到结束的所有的Q值的预测(数组)
        # states 是( s, a, r, s_ )中的s，输入神经网络之后，对改状态s进行Q值的估计，
        # 根据神经网络的两个输出[value1, value2]的大小来确定采取哪个动作[a0, a1]
        p = self.brain.predict(states)
        p_ = self.brain.predict(states_)

        x = numpy.zeros((batchLen, self.stateCnt))
        y = numpy.zeros((batchLen, self.actionCnt))

        # 迭代所有的样本，为每一个样本设置一个合适的target, 作为神经网络的训练数据x ,y
        for i in range(batchLen):
            o = batch[i]
            s = o[0]; a = o[1]; r = o[2]; s_ = o[3]

            t = p[i]
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + GAMMA * numpy.amax(p_[i])

            # (s, a, r, s_) ==
            #     (
            #         array([0.00836298, 0.02041548, 0.01961786, 0.01169834]),
            #         0,
            #         1.0,
            #         array([0.00877129, -0.17498225, 0.01985182, 0.3105058])
            #     )
            #
            # t = p[i] == [0.1397324 1.2220255]
            # t[a] == 1.2220255

            # print("(s, a, r, s_) ==>", (s, a, r, s_))
            # print("t=p[i] ==>", t)
            # print("t[a] ==>", t[a])

            x[i] = s
            y[i] = t

        self.brain.train(x, y)

    # # get all samples (mid-function)
    # def get_all_samples(self):
    #     all_samples = self.memory.all_samples()
    #     return all_samples


# -------------------- ENVIRONMENT ---------------------
class Environment:
    def __init__(self, problem):
        self.problem = problem
        # 创建打砖块游戏环境
        self.env = gym.make(problem)

    # run()执行一个 episode
    def run(self, agent):
        # 复位, type is <class 'numpy.ndarray'>
        s = self.env.reset()
        R = 0

        while True:
            # 在屏幕上显示画面
            self.env.render()

            # 在状态为s时，决定采取什么动作a
            a = agent.act(s)
            # print("a =====================>", a)

            # 在动作为a时，获取环境观测（下一时刻的状态信息）、reward、终止信号、当前时刻状态信息
            s_, r, done, info = self.env.step(a)
            # print("===================>", s_, r, done, info)

            if done:  # terminal state
                s_ = None

            # 将样本(s, a, r, s_)添加到memory中
            agent.observe((s, a, r, s_))
            # memory回放和改善
            agent.replay()

            # 将当前状态置于s_，即为下一时刻状态
            s = s_
            # 计算总的reward，即R
            R += r

            if done:
                # print("+++++++++++++++++++++++++++++++++++++++ start ++++++++++++++++++++++++++++++++++++++")
                # all_samp = agent.get_all_samples()
                # print(all_samp)
                # print("length:", len(all_samp))
                # print("+++++++++++++++++++++++++++++++++++++++ end +++++++++++++++++++++++++++++++++++++++")
                break

        print("Total reward:", R)


# -------------------- MAIN ----------------------------
PROBLEM = 'CartPole-v0'
env = Environment(PROBLEM)

stateCnt = env.env.observation_space.shape[0]
actionCnt = env.env.action_space.n

# 4, 2
# print("stateCnt, actionCnt", stateCnt, actionCnt)

agent = Agent(stateCnt, actionCnt)

try:
    while True:
        env.run(agent)
finally:
    agent.brain.model.save("cartpole-basic.h5")