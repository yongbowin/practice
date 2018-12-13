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

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


def iris_type(s):
    # bytes to str
    s = str(s, encoding="utf8")

    it = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    return it[s]


PROJECT_PATH = os.path.dirname(os.getcwd())
path = PROJECT_PATH + '/data/iris.txt'
data = np.loadtxt(path, dtype=float, delimiter=',', converters={4: iris_type})

x, y = np.split(data, (4,), axis=1)
y = y.flatten()
# x.shape=(150,4)   y.shape=(150,)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.6)

# clf = SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
clf = SVC(C=1, kernel='rbf', gamma=0.3, decision_function_shape='ovr')
clf.fit(x_train, y_train.ravel())

# 精度
print(clf.score(x_train, y_train))
y_pred = clf.predict(x_train)
print(accuracy_score(y_pred, y_train))

print("-------------------")

print(clf.score(x_test, y_test))
y_pred = clf.predict(x_test)
print(accuracy_score(y_pred, y_test))


