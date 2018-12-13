# encoding: utf-8

import numpy as np
import os

from sklearn.model_selection import GridSearchCV
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
path = 'dataset/iris.txt'
data = np.loadtxt(path, dtype=float, delimiter=',', converters={4: iris_type})

x, y = np.split(data, (4,), axis=1)
y = y.flatten()
# x.shape=(150,4)   y.shape=(150,)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.6)


pipeline = Pipeline([
    ('clf', SVC(kernel='rbf', gamma=0.01, C=100))
])
parameters = {
    'clf__gamma': (0.01, 0.03, 0.1, 0.3, 1, 1.3),
    'clf__C': (0.1, 0.3, 1, 3, 10, 30, 40),
}

grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, scoring='accuracy', refit=True)
grid_search.fit(x_train, y_train)

print('best test score:%0.3f' % grid_search.best_score_)
print('best params set:')
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print('\t%s: %r' % (param_name, best_parameters[param_name]))

predictions = grid_search.predict(x_test)
print("index of each category:", classification_report(y_test, predictions))
print("accuracy of all:", accuracy_score(y_test, predictions))


