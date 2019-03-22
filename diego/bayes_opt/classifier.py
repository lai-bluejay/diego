#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
/Users/charleslai/PycharmProjects/diego/diego/bayes_opt.classifier.py was created on 2019/03/20.
file in :relativeFile
Author: Charles_Lai
Email: lai.bluejay@gmail.com
"""

from diego.classifier import DiegoClassifier

class BayesOptClassifier(DiegoClassifier):
    
    def __init__(self, *args, **kwargs):
        super(BayesOptClassifier, self).__init__(*args, **kwargs)
    

if __name__ == '__main__':
    import sklearn
    import autosklearn.classification
    
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,
                                                        train_size=0.75, test_size=0.25)

    clf = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=30,
    )
    clf.fit(X_train, y_train)

    # Print prediction score and statistics.
    y_pred = clf.predict(X_test)
    print("accracy: ", sklearn.metrics.accuracy_score(y_pred, y_test))
    print(clf.sprint_statistics())
    print('='*50, 'models', "="*50)
    print(clf.show_models())

    clf.fit_ensemble(y_train, ensemble_size=50)
    print('='*50, 'ensemble', "="*50)

    print(clf.show_models())
    predictions = clf.predict(X_test)
    print(clf.sprint_statistics())
    print("Accuracy score", sklearn.metrics.accuracy_score(y_test, predictions))