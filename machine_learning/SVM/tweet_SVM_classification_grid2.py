#!~/anaconda3/bin/python3.6
# encoding: utf-8

"""
@version: 0.0.1
@author: Yongbo Wang
@contact: yongbowin@outlook.com
@file: Machine-Learning-Practice - Iris_SVM_classification.py
@time: 4/2/18 7:16 PM
@description: 
"""
import random

import numpy as np
import os
import pickle

from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def load_data(data_set):
    # acquire project root path
    PROJECT_PATH = os.path.dirname(os.getcwd())

    pkl_file = open(PROJECT_PATH + '/data/' + data_set, 'rb')

    # The length of data is 56, the type is 'list'
    data_list = pickle.load(pkl_file)

    return data_list


# for training
# train_list = load_data('tweet_vec_stem.pkl')
train_list = load_data('tweet_vec_stem_topic_list.pkl')
no_keywords_list = load_data('no_keywords_tweet_list.pkl')


total_list = []
for train_item in train_list:
    len_info = len(train_item['tweet_list_info'])

    tweet_list_info = train_item['tweet_list_info']

    for j in tweet_list_info:
        # 跳过没有关键词的tweet
        if j['tweet_id'] in no_keywords_list:
            continue

    for tweet_item in train_item['tweet_list_info']:
        item_list = []

        relevance = int(tweet_item['relevance'])

        N_title = int(tweet_item['N_title'])
        N_desc_narr = int(tweet_item['N_desc_narr'])
        N_word = int(tweet_item['N_word'])
        N_hashtag = int(tweet_item['N_hashtag'])
        time = int(tweet_item['time'])
        is_redundant = int(tweet_item['is_redundant'])
        # is_link = int(tweet_item['is_link'])
        is_link = int(tweet_item['tweet_id'])

        N_friends_cout = int(tweet_item['N_friends_cout'])
        N_followers_cout = int(tweet_item['N_followers_cout'])
        N_statuses_cout = int(tweet_item['N_statuses_cout'])

        # item_list.append(N_title)
        # item_list.append(N_desc_narr)
        # item_list.append(N_word)
        # item_list.append(N_hashtag)
        # item_list.append(time)
        # # item_list.append(is_redundant)
        # item_list.append(0)
        # item_list.append(is_link)
        # item_list.append(N_friends_cout)
        # item_list.append(N_followers_cout)
        # item_list.append(N_statuses_cout)

        item_list.append(N_title)
        item_list.append(N_desc_narr)
        item_list.append(N_word)
        item_list.append(N_hashtag)
        item_list.append(0)
        # item_list.append(is_redundant)
        item_list.append(0)
        item_list.append(is_link)
        item_list.append(N_friends_cout)
        item_list.append(N_followers_cout)
        item_list.append(N_statuses_cout)

        item_list.append(relevance)

        total_list.append(item_list)


# 对标签为0的数据进行欠采样，目的是让数据均衡
label_0_list = []
label_1_list = []
label_2_list = []
for item in total_list:
    if item[-1] == 0:
        label_0_list.append(item)
    elif item[-1] == 1:
        label_1_list.append(item)
    elif item[-1] == 2:
        label_2_list.append(item)

print(len(label_2_list))

# random.seed(1)
random.seed(1)
label_0_list = random.sample(label_0_list, 400)
label_1_list = random.sample(label_1_list, 400)
label_2_list.extend(label_0_list)
label_2_list.extend(label_1_list)
random.shuffle(label_2_list)

# ------------------------
total_list = np.array(label_2_list)

x, y = np.split(total_list, (10,), axis=1)
y = y.flatten()
# x.shape=(150,4)   y.shape=(150,)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.80)

# print("===>", total_list.shape)
# print("===>", y.shape)
print("x_train.shape ===>", x_train.shape)
print("y_train.shape ===>", y_train.shape)
print("===============================")

# # ======================== save data ==========================
# # 将训练数据和测试数的的id分别保存下来，在进行LSTM模型训练时作为输入，保证数据的一致性.
# train_tweet_id = []
# for i in x_train:
#     train_tweet_id.append(str(i[6]))
#
# test_tweet_id = []
# for i in x_test:
#     test_tweet_id.append(str(i[6]))
#
# train_tweet_id = list(set(train_tweet_id))
# test_tweet_id = list(set(test_tweet_id))
#
# print(len(train_tweet_id))
# print(len(test_tweet_id))
#
# PROJECT_PATH = os.path.dirname(os.getcwd())
# # save train dataset
# fp = open(PROJECT_PATH + "/data/tweet_vec_stem_topic_list_train1.pkl", "wb")
# pickle.dump(train_tweet_id, fp, 0)
# fp.close()
#
# # save test dataset
# f = open(PROJECT_PATH + "/data/tweet_vec_stem_topic_list_test1.pkl", "wb")
# pickle.dump(test_tweet_id, f, 0)
# f.close()
# print("save finished ...")
# # print(len(train_tweet_id), len(set(train_tweet_id)))
# # print(len(test_tweet_id), len(set(test_tweet_id)))
# # ======================== save data ==========================

# 标准化数据
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.fit_transform(x_test)

# clf = SVC(C=1, kernel='rbf', gamma=0.3, decision_function_shape='ovr')
# clf.fit(x_train, y_train.ravel())
#
# # 精度
# print(clf.score(x_train, y_train))
# y_pred = clf.predict(x_test)
# print(accuracy_score(y_pred, y_test))

pipeline = Pipeline([
    ('clf', SVC(kernel='rbf', gamma=0.01, C=100))
])
parameters = {
    'clf__gamma': (0.005, 0.01, 0.02, 0.03, 0.1, 0.3, 1),
    'clf__C': (0.05, 0.1, 0.3, 1, 3, 10, 30, 40),
    # 'clf__gamma': (0.0005, 0.005, 0.01),
    # 'clf__C': (0.01, 0.1, 0.2),
}

grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, scoring='accuracy', refit=True)
grid_search.fit(x_train, y_train)

print('最佳效果：%0.3f' % grid_search.best_score_)
print('最优参数集：')
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print('\t%s: %r' % (param_name, best_parameters[param_name]))

predictions = grid_search.predict(x_test)

print("accuracy_score ==>", accuracy_score(y_test, predictions))

# sum1 = 0
# for ii in y_pred:
#     sum1 += ii
# print("pred sum ==>", sum1, y_pred)

# # =================
# sum2 = 0
# for ii in y_train:
#     sum2 += ii
# print("y_train sum:", sum2)
#
# sum3 = 0
# for ii in y_test:
#     sum3 += ii
# print("y_test sum:", sum3)
#
# sum4 = 0
# for ii in y:
#     sum4 += ii
# print("y sum:", sum4)
#
# cou_0 = 0
# cou_1 = 0
# cou_2 = 0
# for ii in y:
#     if ii==0:
#         cou_0 += 1
#     elif ii==1:
#         cou_1 += 1
#     elif ii==2:
#         cou_2 += 1
#
# # 28325 | 1042 | 401
# print(cou_0, '|', cou_1, '|', cou_2)

comm_0 = 0
comm_1 = 0
comm_2 = 0
len_test = len(y_test)
comm_count = 0
no_zero_num = 0
y_test_no_zero = 0
for i in range(len_test):
    if y_test[i] != 0 and predictions[i] != 0:
        no_zero_num += 1

    if y_test[i] != 0:
        y_test_no_zero += 1

    if y_test[i] == predictions[i]:
        comm_count += 1
        if y_test[i] == 0:
            comm_0 += 1
        elif y_test[i] == 1:
            comm_1 += 1
        elif y_test[i] == 2:
            comm_2 += 1


print("rel:", comm_count, "/", len_test, "||", comm_0, "+", comm_1, "+", comm_2)

test_0 = 0
test_1 = 0
test_2 = 0
for ii in y_test:
    if ii==0:
        test_0 += 1
    elif ii==1:
        test_1 += 1
    elif ii==2:
        test_2 += 1

# pred_0 = 0
# pred_1 = 0
# pred_2 = 0
# for ii in y_pred:
#     if ii==0:
#         pred_0 += 1
#     elif ii==1:
#         pred_1 += 1
#     elif ii==2:
#         pred_2 += 1
#
print("y_test:", test_0, "+", test_1, "+", test_2)
# print("y_pred:", pred_0, "+", pred_1, "+", pred_2)

print("2 present:", float(comm_2) / test_2)

print("rel '0':", float(comm_0) / test_0, "|| rel '1':", float(comm_1) / test_1, "|| rel '2':", float(comm_2) / test_2)
print("no '0':", no_zero_num, "/", y_test_no_zero, "|| accur:", float(no_zero_num) / y_test_no_zero)