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

from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
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


pipeline = Pipeline([
    ('clf', SVC(kernel='rbf', gamma=0.01, C=100))
])
parameters = {
    'clf__gamma': (0.01, 0.03, 0.1, 0.3, 1),
    'clf__C': (0.1, 0.3, 1, 3, 10, 30),
}

grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, scoring='accuracy', refit=True)
grid_search.fit(x_train, y_train)

print('最佳效果：%0.3f' % grid_search.best_score_)
print('最优参数集：')
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print('\t%s: %r' % (param_name, best_parameters[param_name]))

predictions = grid_search.predict(x_test)
print("-------------------")
print(classification_report(y_test, predictions))

print(accuracy_score(y_test, predictions))


