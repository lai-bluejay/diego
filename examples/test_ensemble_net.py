#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
examples/test_simlple_study.py was created on 2019/03/22.
file in :relativeFile
Author: Charles_Lai
Email: lai.bluejay@gmail.com
"""
import os
import sys
root = os.path.dirname(os.path.abspath(__file__))
sys.path.append("%s/../.." % root)
sys.path.append("%s/.." % root)
sys.path.append("%s/../diego" % root)
sys.path.append("%s/../../.." % root)
sys.path.append(u"{0:s}".format(root))
import numpy as np
from diego.study import create_study
from autosklearn.classification import AutoSklearnClassifier
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression, SGDClassifier
from diego.ensemble_net import Ensemble, EnsembleStack, EnsembleStackClassifier, Combiner

if __name__ == "__main__":
    import sklearn.datasets
    digits = sklearn.datasets.load_breast_cancer()
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(digits.data, digits.target,
                                                                                train_size=0.8, test_size=0.2)
    clf1 = LogisticRegression()
    clf2 = SGDClassifier(loss='log')
    clf3 = SGDClassifier(loss='log')
    clf1.fit(X_train, y_train)
    clf2.fit(X_train, y_train)
    clf3.fit(X_train, y_train)

    ens = Ensemble([clf1, clf2, clf3])
    stack = EnsembleStack(cv=3)
    stack.add_layer(ens)

    sclf = EnsembleStackClassifier(stack, combiner=Combiner('mean'))
    sclf.refit(X_train, y_train)
    y_pred = sclf.predict(X_test)
    print(accuracy_score(y_pred, y_test))

