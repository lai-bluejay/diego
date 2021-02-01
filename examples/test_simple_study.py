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
sys.path.append("%s/../../.." % root)
sys.path.append("%s/../diego" % root)


# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)



import numpy as np
from diego.study import create_study
from autosklearn.classification import AutoSklearnClassifier
import sklearn
import sklearn.datasets
import sklearn.metrics


if __name__ == "__main__":
    X, y = sklearn.datasets.load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = \
            sklearn.model_selection.train_test_split(X, y, random_state=2047, train_size=0.8, test_size=0.2)
    s = create_study(X_train, y_train,is_autobin=False, metrics='acc', sample_method=None, precision=np.float32)
    # s.generate_autosk_trial(mode='fast', n_jobs=1)
    s.optimize(X_test, y_test)
    s.show_models()
