#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   test_sk.py
@Time    :   2020/12/24 16:03:12
@Author  :   Charles Lai 
@Version :   1.0
@Desc    :   None
'''
# here put the import lib
import autosklearn.classification
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
if __name__ == "__main__":
    X, y = sklearn.datasets.load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = \
            sklearn.model_selection.train_test_split(X, y, random_state=1)
    automl = autosklearn.classification.AutoSklearnClassifier(memory_limit=8192, n_jobs=-1)
    automl.fit(X_train, y_train)
    y_hat = automl.predict(X_test)
    print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_hat))