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
print(sys.path)
from diego.study import create_study

if __name__ == "__main__":
    import sklearn.datasets
    digits = sklearn.datasets.load_digits()
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(digits.data, digits.target,
                                                                                train_size=0.75, test_size=0.25)

    s = create_study(X_train, y_train)
    s.generate_autosk_trial(mode='fast')
    s.optimize(X_test, y_test)
    for t in s.trials_list:
        print(t.__dict__)
        print(t.clf.score(X_test, y_test))