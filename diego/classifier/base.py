#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
/Users/charleslai/PycharmProjects/diego/.vscode/diego.classifier.py was created on 2019/03/18.
file in :relativeFile
Author: Charles_Lai
Email: lai.bluejay@gmail.com
"""
from typing import Optional, List
import warnings
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_X_y, check_array
import sklearn.utils
from sklearn.metrics.classification import type_of_target

from ConfigSpace import ConfigurationSpace, CategoricalHyperparameter

from autosklearn.automl import BaseAutoML
from diego.basic import *
from diego.metrics import Scorer


class DiegoClassifier(BaseAutoML):
    # C3 BFS search class methods
    # TODO Rewrite Ensemble methods
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 暂时是二分类或者多分类
        self._task_mapping = {'multilabel-indicator': MULTILABEL_CLASSIFICATION,
                              'multiclass': MULTICLASS_CLASSIFICATION,
                              'binary': BINARY_CLASSIFICATION}

    def _check_y(self, y):
        y = check_array(y, ensure_2d=False)

        y = np.atleast_1d(y)
        if y.ndim == 2 and y.shape[1] == 1:
            warnings.warn("A column-vector y was passed when a 1d array was"
                          " expected. Will change shape via np.ravel().",
                          sklearn.utils.DataConversionWarning, stacklevel=2)
            y = np.ravel(y)

        return y

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        metric: Optional[Scorer] = None,
        feat_type: Optional[List[bool]] = None,
        dataset_name: Optional[str] = None,
        only_return_configuration_space: bool = False,
        load_models: bool = True,
    ):
        # important, 都要做
        X, y = check_X_y(X, y, accept_sparse="csr")
        check_classification_targets(y)

        if X_test is not None:
            X_test, y_test = check_X_y(X_test, y_test)
            if len(y.shape) != len(y_test.shape):
                raise ValueError('Target value shapes do not match: %s vs %s'
                                 % (y.shape, y_test.shape))

        y_task = type_of_target(y)

        # 7 category
        task = self._task_mapping.get(y_task)
        if task is None:
            raise ValueError('Cannot work on data of type %s' % y_task)

        if metric is None:
            if task == MULTILABEL_CLASSIFICATION:
                metric = f1_macro
            else:
                metric = accuracy

        if y_test is not None:
            # Map test values to actual values - TODO: copy to all kinds of
            # other parts in this code and test it!!!
            y_test_new = []
            for output_idx in range(len(self._classes)):
                mapping = {self._classes[output_idx][idx]: idx
                           for idx in range(len(self._classes[output_idx]))}
                enumeration = y_test if len(
                    self._classes) == 1 else y_test[output_idx]
                y_test_new.append(
                    np.array([mapping[value] for value in enumeration])
                )
            y_test = np.array(y_test_new)
            if self._n_outputs == 1:
                y_test = y_test.flatten()

        return super().fit(
            X, y,
            X_test=X_test,
            y_test=y_test,
            task=task,
            metric=metric,
            feat_type=feat_type,
            dataset_name=dataset_name,
            only_return_configuration_space=only_return_configuration_space,
            load_models=load_models,
        )

    def fit_ensemble(self, y, task=None, metric=None, precision='32',
                     dataset_name=None, ensemble_nbest=None,
                     ensemble_size=None):
        y = self._process_target_classes(y)

        return super().fit_ensemble(y, task, metric, precision, dataset_name,
                                    ensemble_nbest, ensemble_size)

    def _process_target_classes(self, y):
        y = self._check_y(y)
        self._n_outputs = 1 if len(y.shape) == 1 else y.shape[1]

        y = np.copy(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        self.n_classes = n_classes
        classes_ = self.classes_
        return y

    def predict(self, X, batch_size=None, n_jobs=1):
        return super().predict(X, batch_size=batch_size, n_jobs=n_jobs)

    def predict_proba(self, X, batch_size=None, n_jobs=1):
        return super().predict(X, batch_size=batch_size, n_jobs=n_jobs)

    def predict_label(self, X, batch_size=None, n_jobs=1):
        predicted_probabilities = super().predict(X, batch_size=batch_size,
                                                  n_jobs=n_jobs)
        if self._n_outputs == 1:
            predicted_indexes = np.argmax(predicted_probabilities, axis=1)
            predicted_classes = self.classes_[0].take(predicted_indexes)

            return predicted_classes
        else:
            predicted_indices = (predicted_probabilities > 0.5).astype(int)
            n_samples = predicted_probabilities.shape[0]
            predicted_classes = np.zeros((n_samples, self._n_outputs))

            for k in range(self._n_outputs):
                output_predicted_indexes = predicted_indices[:, k].reshape(-1)
                predicted_classes[:, k] = self.classes_[k].take(
                    output_predicted_indexes)

            return predicted_classes

    def extend_classifier(self):
        pass

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'AutoBinning',
                'name': 'Auto Binning for linear model',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': False,
                'handles_multilabel': False,
                'is_deterministic': True,
                'input': (DENSE, UNSIGNED_DATA, SIGNED_DATA),
                'output': (DENSE, UNSIGNED_DATA, SIGNED_DATA)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        # TODO add hyperparameter to gbdt binning
        cs = ConfigurationSpace()
        binning_method = CategoricalHyperparameter(
        )
        # shrinkage = UniformFloatHyperparameter(
        #     name="shrinkage", lower=0.0, upper=1.0, default_value=0.5
        # )
        # n_components = UniformIntegerHyperparameter(
        #     name="n_components", lower=1, upper=29, default_value=10
        # )
        # tol = UniformFloatHyperparameter(
        #     name="tol", lower=0.0001, upper=1, default_value=0.0001
        # )
        cs.add_hyperparameters([binning_method])
        return cs
