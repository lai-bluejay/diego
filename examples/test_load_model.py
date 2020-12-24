#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
examples/test_load_model.py was created on 2019/04/01.
file in :relativeFile
Author: Charles_Lai
Email: lai.bluejay@gmail.com
"""

import joblib

pipeline = joblib.load('./hehediego_model_no-name-00000000-0000-0000-0000-000000000000.joblib')
import sklearn.datasets
digits = sklearn.datasets.load_breast_cancer()
X, y = digits.data, digits.target
y_pred = pipeline.predict(X)
print(y_pred)