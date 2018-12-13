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
import operator
import numpy as np
from keras.layers import Dense, LSTM
from keras.models import Sequential
from keras.optimizers import Adam

from config.QueryMongoDB import find_by_tweetid

EPISODES = 1000
PROJECT_PATH = os.path.dirname(os.getcwd())


# This is Policy Gradient agent for the Cartpole
# In this example, we use REINFORCE algorithm which uses monte-carlo update rule
class REINFORCEAgent:
    days_dict = {
        '20160802': [], '20160803': [], '20160804': [], '20160805': [], '20160806': [],
        '20160807': [], '20160808': [], '20160809': [], '20160810': [], '20160811': []
    }

    def __init__(self, state_size, action_size):
        # if you want to see Cartpole learning, then change to True
        self.load_model = True
        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # These are hyper parameters for the Policy Gradient
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.hidden0, self.hidden1, self.hidden2 = 256, 256, 128

        # create model for policy network
        self.model = self.build_model()

        # lists for the states, actions and rewards
        self.states, self.actions, self.rewards = [], [], []

        if self.load_model:
            self.model.load_weights(PROJECT_PATH + "/Practice/pg_lstm_reinforce.h5")

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

    # save <s, a ,r> of each step
    def append_sample(self, state):
        if len(self.states) != 0:
            self.states.pop(0)
        self.states.append(state)

    def test_model(self):
        episode_length = len(self.states)

        update_inputs = np.zeros((episode_length, self.state_size))

        for i in range(episode_length):
            update_inputs[i] = self.states[i]

        update_inputs = np.reshape(update_inputs, [len(update_inputs), 1, self.state_size])

        pred = self.model.predict(update_inputs, verbose=2)
        self.states = []

        return pred

    def load_data(self):
        pkl_file = open(PROJECT_PATH + '/Practice/tweet_vec_list_test.pkl', 'rb')
        # The length of data is 56, the type is 'list'
        train_list = pickle.load(pkl_file)

        return train_list

    def result_save(self, tweet_id, score, topid):
        push_item_dict = {}
        write_list = []
        tweet_dict = find_by_tweetid("FullTweetStem2016", tweet_id)

        raw_created_at = tweet_dict['created_at']
        created_at = ((tweet_dict['created_at'])[:10]).replace('-', '')

        sep = '\t'
        line_break = '\n'
        write_list.append(created_at + sep)
        write_list.append(raw_created_at + sep)
        write_list.append(topid + sep)
        write_list.append(tweet_id + sep)
        # print("---------->", type(score), "||", score)
        write_list.append(str(score) + sep)
        write_list.append('PG_A1' + line_break)

        push_item_dict['score'] = score
        push_item_dict['topid'] = topid
        push_item_dict['tweet_id'] = tweet_id
        push_item_dict['created_at'] = raw_created_at
        push_item_dict['write_list'] = write_list

        if created_at != '20160801':
            (self.days_dict[created_at]).append(push_item_dict)

    def write_res(self):
        DATA_EVALU_2016 = PROJECT_PATH + '/evaluation/PG_LSTM_TEST'
        print('start writing ...')
        f = open(DATA_EVALU_2016, 'w')

        for key in self.days_dict:
            topic_dict_temp = {
                'MB229': [], 'MB286': [], 'MB420': [], 'MB410': [], 'MB362': [], 'RTS14': [], 'MB320': [], 'MB256': [], 'MB436': [],
                'RTS13': [], 'MB365': [], 'MB265': [], 'RTS28': [], 'RTS43': [], 'MB361': [], 'MB419': [], 'RTS10': [], 'RTS6': [],
                'MB409': [], 'RTS36': [], 'RTS24': [], 'MB425': [], 'RTS35': [], 'MB414': [], 'MB440': [], 'MB392': [], 'RTS27': [],
                'RTS5': [], 'MB319': [], 'MB363': [], 'MB377': [], 'RTS32': [], 'RTS37': [], 'MB239': [], 'RTS2': [], 'RTS31': [], 'MB381': [],
                'RTS1': [], 'MB438': [], 'MB371': [], 'MB351': [], 'MB276': [], 'MB254': [], 'MB230': [], 'MB258': [], 'MB431': [], 'MB226': [],
                'RTS21': [], 'RTS19': [], 'MB382': [], 'MB267': [], 'MB358': [], 'MB332': [], 'MB391': [], 'RTS25': [], 'RTS4': []
            }

            print("day num:", key, "...")

            # 将某一天的tweet按照topid重新分组
            for i in self.days_dict[key]:
                (topic_dict_temp[i['topid']]).append(i)

            # 每一天，都需要对所有topid取出前10个最相关的tweet
            for j in topic_dict_temp:
                # 用来测试的topic只有8个
                if not topic_dict_temp[j]:
                    continue

                # TODO: 加上cluster过滤
                (topic_dict_temp[j]).sort(key=operator.itemgetter('score'), reverse=True)
                topic_dict_temp[j] = (topic_dict_temp[j])[:10]
                # (topic_dict_temp[j]).sort(key=operator.itemgetter('raw_created_at'))

                for write_line in topic_dict_temp[j]:
                    f.writelines(write_line['write_list'])

        f.close()


if __name__ == "__main__":
    # get size of state and action from environment
    state_size = 300
    action_size = 2
    # make REINFORCE agent
    agent = REINFORCEAgent(state_size, action_size)
    # load data
    train_list = agent.load_data()

    count = 0
    cou_12 = 0
    comm = 0
    argmax_cou = 0
    total_tweet_cou = 0
    for topic_item in train_list:
        count += 1
        print("count ======>", count)
        tweet_list_len = len(topic_item['tweet_list_info'])
        for num in range(tweet_list_len):
            total_tweet_cou += 1
            # The current state 's'
            s = ((topic_item['tweet_list_info'])[num])['tweet_vec']
            state = np.reshape(s, [1, 1, state_size])
            relevance = int(((topic_item['tweet_list_info'])[num])['relevance'])

            # save the sample <s, a, r> to the memory
            agent.append_sample(state)

            pred = agent.test_model()

            print("topid", topic_item['topid'], "==>", "pred:", pred, "|| action:", np.argmax(pred), "|| relevance:", relevance)

            # ===================================
            if np.argmax(pred)!=0:
                tweet_id = ((topic_item['tweet_list_info'])[num])['tweet_id']

                agent.result_save(tweet_id, pred[0][1], topic_item['topid'])

            # ===================================

            if int(relevance)!=0:
                cou_12 += 1

            if np.argmax(pred)!=0:
                argmax_cou += 1

            if int(relevance)!=0 and np.argmax(pred)!=0:
                comm += 1

    # 将结果生成提交格式进行评估
    agent.write_res()

    print("comm / cou_12:", comm, "/", cou_12)
    print("accuracy:", float(comm) / argmax_cou)
    print("total_tweet_cou:", total_tweet_cou, "|| push nums:", argmax_cou, "|| push rate:", float(argmax_cou) / total_tweet_cou)

# lr=0.01, hidden0=128, hidden1=128, hidden2=64
# 评估结果
# EG1     EG0     nCG1    nCG0
# 0.2146  0.0057  0.2269  0.0180
