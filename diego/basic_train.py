#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
diego/basic_train.py was created on 2019/03/21.
file in :relativeFile
Author: Charles_Lai
Email: lai.bluejay@gmail.com
"""
import io
import json
import os
from typing import Optional, List
import unittest.mock
import warnings

from ConfigSpace.read_and_write import pcs
import numpy as np
import numpy.ma as ma
import scipy.stats
from sklearn.base import BaseEstimator
from sklearn.model_selection._split import _RepeatedSplits, \
    BaseShuffleSplit, BaseCrossValidator
from sklearn.externals import joblib
import sklearn.utils
import scipy.sparse
from sklearn.metrics.classification import type_of_target

from autosklearn.metrics import Scorer
from autosklearn.data.abstract_data_manager import AbstractDataManager
from autosklearn.data.competition_data_manager import CompetitionDataManager
from autosklearn.data.xy_data_manager import XYDataManager
from autosklearn.evaluation import ExecuteTaFuncWithQueue
from autosklearn.metrics import calculate_score
from autosklearn.util import StopWatch, get_logger, setup_logger, \
    pipeline
from autosklearn.ensemble_builder import EnsembleBuilder
from autosklearn.smbo import AutoMLSMBO
from autosklearn.util.hash import hash_array_or_matrix
from autosklearn.metrics import f1_macro, accuracy, r2


def _model_predict(self, X, batch_size, identifier):
    def send_warnings_to_log(
            message, category, filename, lineno, file=None, line=None):
        self._logger.debug('%s:%s: %s:%s' %
                       (filename, lineno, category.__name__, message))
        return
    model = self.models_[identifier]
    X_ = X.copy()
    with warnings.catch_warnings():
        warnings.showwarning = send_warnings_to_log
        if self._task in REGRESSION_TASKS:
            prediction = model.predict(X_, batch_size=batch_size)
        else:
            prediction = model.predict_proba(X_, batch_size=batch_size)
    if len(prediction.shape) < 1 or len(X_.shape) < 1 or \
            X_.shape[0] < 1 or prediction.shape[0] != X_.shape[0]:
        self._logger.warning("Prediction shape for model %s is %s "
                             "while X_.shape is %s" %
                             (model, str(prediction.shape),
                              str(X_.shape)))
    return prediction






