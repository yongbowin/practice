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
train_list = load_data('tweet_vec_stem.pkl')

xy_total = np.zeros((0, 11))

cc = 0
for train_item in train_list:
    len_info = len(train_item['tweet_list_info'])
    xy = np.zeros((len_info, 11))

    count = 0
    for tweet_item in train_item['tweet_list_info']:
        m = np.zeros(11)

        relevance = int(tweet_item['relevance'])

        N_title = int(tweet_item['N_title'])
        N_desc_narr = int(tweet_item['N_desc_narr'])
        N_word = int(tweet_item['N_word'])
        N_hashtag = int(tweet_item['N_hashtag'])
        time = int(tweet_item['time'])
        is_redundant = int(tweet_item['is_redundant'])
        is_link = int(tweet_item['is_link'])

        N_friends_cout = int(tweet_item['N_friends_cout'])
        N_followers_cout = int(tweet_item['N_followers_cout'])
        N_statuses_cout = int(tweet_item['N_statuses_cout'])

        m[0] = N_title
        m[1] = N_desc_narr
        m[2] = N_word
        m[3] = N_hashtag
        m[4] = time
        # m[5] = is_redundant
        m[5] = 0
        m[6] = is_link
        m[7] = N_friends_cout
        m[8] = N_followers_cout
        m[9] = N_statuses_cout

        m[10] = relevance

        cc += int(N_hashtag)

        xy[count] = m

        count += 1

    xy_total = np.append(xy_total, xy, axis=0)

print("<L<L<L<L<L<L<L<L>>>>>>>", cc)

print(xy_total[0])
print(xy_total[1])
print(xy_total[251])

x, y = np.split(xy_total, (10,), axis=1)
y = y.flatten()
# x.shape=(150,4)   y.shape=(150,)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.6)

print("===>", xy_total.shape)
print("===>", y.shape)

# 标准化数据
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.fit_transform(x_test)

pipeline = Pipeline([
    ('clf', SVC(kernel='rbf', gamma=0.01, C=100))
])
parameters = {
    # 'clf__gamma': (0.005, 0.01, 0.03, 0.1, 0.3, 1),
    # 'clf__C': (0.05, 0.1, 0.3, 1, 3, 10, 30),
    'clf__gamma': (0.0005, 0.005, 0.01),
    'clf__C': (0.01, 0.1, 0.2),
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

sum1 = 0
for ii in predictions:
    sum1 += ii
print("pred sum ==>", sum1)

sum2 = 0
for ii in y_train:
    sum2 += ii
print("y_train sum:", sum2)

sum3 = 0
for ii in y_test:
    sum3 += ii
print("y_test sum:", sum3)

sum4 = 0
for ii in y:
    sum4 += ii
print("y sum:", sum4)
