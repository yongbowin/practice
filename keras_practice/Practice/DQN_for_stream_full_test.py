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
import numpy
import time
import os
import pickle

from keras.models import Sequential
from keras.layers import *
from keras import optimizers
from keras import losses
from keras import backend as K
import tensorflow as tf
import operator


# ----------
from config.QueryMongoDB import find_by_tweetid

HUBER_LOSS_DELTA = 1.0
LEARNING_RATE = 0.01

PROJECT_PATH = os.path.dirname(os.getcwd())


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

        # load trained model (params)
        self.model.load_weights("dqn_weights.h5")

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

    def predict(self, s):
        # s.shape == (1,300)
        s = numpy.reshape(s, (s.shape[0], 1, s.shape[1]))
        return self.model.predict(s)

    def predictOne(self, s):
        # stateCount = 300, Firstly, (300,)==>(1,300), i.e., [...]==>[[...]]
        # As input for LSTM, (1,300)
        return self.predict(s.reshape(1, self.stateCount)).flatten()


# ************************* Environment *************************
class Environment:
    days_dict = {
        '20160802': [], '20160803': [], '20160804': [], '20160805': [], '20160806': [],
        '20160807': [], '20160808': [], '20160809': [], '20160810': [], '20160811': []
    }

    def __init__(self):
        # load the tweet vectors data
        self.test_list = self.load_data()

    def load_data(self):
        # acquire project root path
        PROJECT_PATH = os.path.dirname(os.getcwd())

        pkl_file = open(PROJECT_PATH + '/Practice/tweet_vec_list_test.pkl', 'rb')

        # The length of data is 56, the type is 'list'
        test_list = pickle.load(pkl_file)

        return test_list

    def result_save(self, tweet_id, score, topid, relevance):
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
        push_item_dict['relevance'] = str(relevance)
        push_item_dict['write_list'] = write_list

        if created_at != '20160801':
            (self.days_dict[created_at]).append(push_item_dict)

    def write_res(self):
        DATA_EVALU_2016 = PROJECT_PATH + '/evaluation/DQN_LSTM_TEST'
        print('start writing ...')
        f = open(DATA_EVALU_2016, 'w')

        no_zero = 0
        all_cou = 0

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
                topic_dict_temp[j] = (topic_dict_temp[j])[:3]
                # (topic_dict_temp[j]).sort(key=operator.itemgetter('raw_created_at'))

                for write_line in topic_dict_temp[j]:
                    relevance = write_line['relevance']
                    all_cou += 1
                    if int(relevance)!=0:
                        no_zero += 1
                    f.writelines(write_line['write_list'])

        print("no_zero:", no_zero, "|| all_cou:", all_cou, "|| accuracy:", float(no_zero) / all_cou)

        f.close()

    # run() function to execute a episode
    def run(self, brain):
        # reset, the type of 's' is <class 'numpy.ndarray'>
        count_0 = 0
        count_1 = 0
        count_2 = 0

        cou_12 = 0
        argmax_cou = 0
        comm = 0

        for topic_item in self.test_list:
            # if count > 0:
            #     break
            # each tweet of 'tweet_list_info' ordered by posted time
            tweet_list_len = len(topic_item['tweet_list_info'])

            for num in range(tweet_list_len):
                # The current state 's'
                s = ((topic_item['tweet_list_info'])[num])['tweet_vec']

                pred = brain.predictOne(s)

                tweet_id = ((topic_item['tweet_list_info'])[num])['tweet_id']
                relevance = ((topic_item['tweet_list_info'])[num])['relevance']

                # ===================================
                if np.argmax(pred) != 0:
                    tweet_id = ((topic_item['tweet_list_info'])[num])['tweet_id']
                    # print(pred)
                    # print(relevance)
                    # print(pred[numpy.argmax(pred)])
                    self.result_save(tweet_id, pred[numpy.argmax(pred)], topic_item['topid'], relevance)

                # ===================================

                print(
                    "topid ******>", topic_item['topid'],
                    "|| tweet_id:", tweet_id,
                    "|| pred:", pred,
                    "|| max index:", numpy.argmax(pred),
                    "|| relevance:", relevance
                )

                if int(relevance) != 0:
                    cou_12 += 1

                if np.argmax(pred) != 0:
                    argmax_cou += 1

                if int(relevance) != 0 and np.argmax(pred) != 0:
                    comm += 1

                if int(relevance) == 0:
                    count_0 += 1
                elif int(relevance) == 1:
                    count_1 += 1
                elif int(relevance) == 2:
                    count_2 += 1

        self.write_res()

        print("comm / cou_12:", comm, "/", cou_12)
        print("accuracy:", float(comm) / argmax_cou)
        print(count_0, count_1, count_2)


# ************************* Main *************************
env = Environment()

stateCount = 300
actionCount = 2

brain = Brain(stateCount, actionCount)

# calculate cost time
start_time = time.time()

try:
    # while True:
    #     env.run(agent)
    env.run(brain)
finally:
    print("\nDone in %ds" % (time.time() - start_time))
