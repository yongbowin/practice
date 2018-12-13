#!~/anaconda3/bin/python3.6
# encoding: utf-8

"""
@version: 0.0.1
@author: Yongbo Wang
@contact: yongbowin@outlook.com
@file: Keras-Practice - PG_cartpole.py
@time: 4/23/18 6:59 PM
@description: 
"""
import os
import pickle
import numpy as np
from keras.layers import Dense, LSTM
from keras.models import Sequential
from keras.optimizers import Adam

EPISODES = 1000
PROJECT_PATH = os.path.dirname(os.getcwd())


# This is Policy Gradient agent for the Cartpole
# In this example, we use REINFORCE algorithm which uses monte-carlo update rule
class REINFORCEAgent:
    def __init__(self, state_size, action_size):
        # if you want to see Cartpole learning, then change to True
        self.load_model = False
        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # These are hyper parameters for the Policy Gradient
        self.discount_factor = 0.99
        # self.learning_rate = 0.01
        self.learning_rate = 0.001
        # self.hidden0, self.hidden1, self.hidden2 = 128, 128, 64
        self.hidden0, self.hidden1, self.hidden2 = 256, 256, 128

        # create model for policy network
        self.model = self.build_model()

        # lists for the states, actions and rewards
        self.states, self.actions, self.rewards = [], [], []

        if self.load_model:
            self.model.load_weights(PROJECT_PATH + "/Practice/lstm_reinforce.h5")

    # approximate policy using Neural Network
    # state is input and probability of each action is output of network
    def build_model(self):
        model = Sequential()
        model.add(LSTM(self.hidden0, input_shape=(1, 300), activation='sigmoid'))
        model.add(Dense(self.hidden1, input_dim=self.state_size, activation='relu', kernel_initializer='glorot_uniform'))
        model.add(Dense(self.hidden2, activation='relu', kernel_initializer='glorot_uniform'))
        model.add(Dense(self.action_size, activation='softmax', kernel_initializer='glorot_uniform'))
        model.summary()
        model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=self.learning_rate))
        return model

    # using the output of policy network, pick action stochastically
    def get_action(self, state):
        policy = self.model.predict(state, batch_size=1).flatten()
        # 从np.arange(self.action_size)中产生一个非标准的size为1的随机采样
        return np.random.choice(self.action_size, 1, p=policy)[0]

    # In Policy Gradient, Q function is not available.
    # Instead agent uses sample returns for evaluating policy
    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    # save <s, a ,r> of each step
    def append_sample(self, state, action, reward):
        self.states.append(state)
        self.rewards.append(reward)
        self.actions.append(action)

    # update policy network every episode
    def train_model(self):
        episode_length = len(self.states)

        discounted_rewards = self.discount_rewards(self.rewards)
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)

        update_inputs = np.zeros((episode_length, self.state_size))
        advantages = np.zeros((episode_length, self.action_size))

        for i in range(episode_length):
            update_inputs[i] = self.states[i]
            advantages[i][self.actions[i]] = discounted_rewards[i]

        update_inputs = np.reshape(update_inputs, [len(update_inputs), 1, self.state_size])

        self.model.fit(update_inputs, advantages, epochs=1, verbose=0)
        self.states, self.actions, self.rewards = [], [], []

    def load_data(self):
        pkl_file = open(PROJECT_PATH + '/Practice/tweet_vec_list_train.pkl', 'rb')
        # The length of data is 56, the type is 'list'
        train_list = pickle.load(pkl_file)

        return train_list


if __name__ == "__main__":
    # get size of state and action from environment
    state_size = 300
    action_size = 2
    # make REINFORCE agent
    agent = REINFORCEAgent(state_size, action_size)
    # load data
    train_list = agent.load_data()

    count = 0
    ii = 0
    while ii < 10:
        ii += 1
        for topic_item in train_list:
            if topic_item['topid'] == 'MB254' \
                    or topic_item['topid'] == 'MB425' \
                    or topic_item['topid'] == 'MB419' \
                    or topic_item['topid'] == 'RTS19':
                continue

            count += 1
            print("epoch:", ii, "|| topic count ======>", count)
            tweet_list_len = len(topic_item['tweet_list_info'])
            for num in range(tweet_list_len):
                # The current state 's'
                s = ((topic_item['tweet_list_info'])[num])['tweet_vec']

                done = False
                score = 0
                state = np.reshape(s, [1, 1, state_size])  # state.shape=(1,1,300)

                # get action for the current state and go one step in environment
                action = agent.get_action(state)
                # next_state, reward, done, info = env.step(action)
                if num == (tweet_list_len - 1):
                    done = True

                reward = float(((topic_item['tweet_list_info'])[num])['relevance']) / 2

                # save the sample <s, a, r> to the memory
                agent.append_sample(state, action, reward)

                if done:
                    # every episode, agent learns from sample returns
                    agent.train_model()

    # save the model
    agent.model.save_weights(PROJECT_PATH + "/Practice/pg_lstm_reinforce.h5")





