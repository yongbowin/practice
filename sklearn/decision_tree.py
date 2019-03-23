"""
For classification and Regression. (unsupervised)
"""
from sklearn import tree


def dt_classifier():
    X = [[0, 0], [1, 1]]
    Y = [0, 1]
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, Y)

    print(clf.predict([[2., 2.]]))


dt_classifier()
