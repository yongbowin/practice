#!~/anaconda3/bin/python3.6
# encoding: utf-8

"""
@version: 0.0.1
@author: Yongbo Wang
@contact: yongbowin@outlook.com
@file: Keras-Practice - Sample_data.py
@time: 4/12/18 5:50 AM
@description: 对训练数据进行欠采样，保证训练集中各个类别的数目均衡
"""
import os
import pickle
import random


# 对训练数据进行欠采样，保证训练集中各个类别的数目均衡
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


# 经过清洗后的所有数据，各个相关度推文数量统计：
# '0' counts: 28321
# '1' counts: 1042
# '2' counts: 401
def load_data1():
    # acquire project root path
    PROJECT_PATH = os.path.dirname(os.getcwd())

    pkl_file = open(PROJECT_PATH + '/Practice/tweet_vec_stem_topic_list.pkl', 'rb')

    # The length of data is 56, the type is 'list'
    data_list = pickle.load(pkl_file)

    # -----------------------------------------------------------------------
    cou_0 = 0
    cou_1 = 0
    cou_2 = 0
    for i in data_list:
        for j in i['tweet_list_info']:
            if int(j['relevance']) == 0:
                cou_0 += 1
            elif int(j['relevance']) == 1:
                cou_1 += 1
            elif int(j['relevance']) == 2:
                cou_2 += 1

    print("'0' counts:", cou_0)
    print("'1' counts:", cou_1)
    print("'2' counts:", cou_2)


# load_data1()


# 将各个相关度推文数量的list存储到pkl文件中
def load_data2():
    # acquire project root path
    PROJECT_PATH = os.path.dirname(os.getcwd())

    pkl_file = open(PROJECT_PATH + '/Practice/tweet_vec_stem_topic_list.pkl', 'rb')

    # The length of data is 56, the type is 'list'
    data_list = pickle.load(pkl_file)

    # -----------------------------------------------------------------------
    cou_0 = 0
    cou_1 = 0
    cou_2 = 0
    rel_0_samp_list = []
    rel_1_samp_list = []
    rel_2_samp_list = []
    rel_012_list = []
    for i in data_list:
        for j in i['tweet_list_info']:
            if int(j['relevance']) == 0:
                cou_0 += 1
                rel_0_samp_list.append(j['tweet_id'])
            elif int(j['relevance']) == 1:
                cou_1 += 1
                rel_1_samp_list.append(j['tweet_id'])
            elif int(j['relevance']) == 2:
                cou_2 += 1
                rel_2_samp_list.append(j['tweet_id'])

    rel_0_samp_list = random.sample(rel_0_samp_list, 400)
    rel_1_samp_list = random.sample(rel_1_samp_list, 400)
    rel_012_list.append(list(set(rel_0_samp_list)))
    rel_012_list.append(list(set(rel_1_samp_list)))
    rel_012_list.append(list(set(rel_2_samp_list)))

    # acquire project root path
    PROJECT_PATH = os.path.dirname(os.getcwd())

    f = open(PROJECT_PATH + "/Practice/tweet_id_set_of_each_rel_samp.pkl", "wb")
    pickle.dump(rel_012_list, f, 0)
    f.close()


load_data2()




