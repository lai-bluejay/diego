#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
classifier/logistic_regression.py was created on 2019/04/24.
file in :relativeFile
Author: Charles_Lai
Email: lai.bluejay@gmail.com
"""
import numpy as np

# from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model.base import LinearClassifierMixin, SparseCoefMixin, BaseEstimator
from sklearn.linear_model import LogisticRegression

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformIntegerHyperparameter, UniformFloatHyperparameter

import sklearn.metrics
import autosklearn.classification
import autosklearn.pipeline.components.classification
from autosklearn.pipeline.components.base import AutoSklearnClassificationAlgorithm
from autosklearn.pipeline.constants import DENSE, SIGNED_DATA, UNSIGNED_DATA, \
    PREDICTIONS

from sklearn.linear_model import LogisticRegression


class LogisticRegressionSK(AutoSklearnClassificationAlgorithm):
    def __init__(self, penalty='l2', dual=False, tol=1e-4, C=1.0,
                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None, solver='warn', max_iter=100,
                 multi_class='warn', verbose=0, warm_start=False, n_jobs=-1):

        self.penalty = penalty
        self.dual = dual
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.random_state = random_state
        self.solver = solver
        self.max_iter = max_iter
        self.multi_class = multi_class
        self.verbose = verbose
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        super(LogisticRegressionSK, self).__init__()

    def fit(self, X, y):
        self.estimator = LogisticRegression(
           penalty=self.penalty, C=self.C, tol=self.tol, solver=self.solver,
           class_weight=self.class_weight, verbose=self.verbose, n_jobs=self.n_jobs,
           max_iter=self.max_iter, warm_start=self.warm_start
           )
    
        self.estimator.fit(X, y)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict(X)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict_proba(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'LogisticRegressionSK',
                'name': 'LogisticRegressionSK',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': False,
                'is_deterministic': False,
                # Both input and output must be tuple(iterable)
                'input': [DENSE, SIGNED_DATA, UNSIGNED_DATA],
                'output': [PREDICTIONS]
                }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()

        # weight of penalty
        C = UniformFloatHyperparameter(
            name='C', lower=0.01, upper=10.0, default_value=1.0
        )

        class_weight = CategoricalHyperparameter(
            name='class_weight', choices=['balanced', None], default_value='balanced'
        )
        penalty = CategoricalHyperparameter(
            name='penalty', choices=['l1', 'l2'], default_value='l2'
        )

        tol = CategoricalHyperparameter(
            name='tol', choices=[1e-5, 1e-4, 1e-3], default_value=1e-4
        )

        max_iter = 2048

        solver = CategoricalHyperparameter(
            name="solver", choices=['sag', 'saga'], default_value='sag'
        )
        cs.add_hyperparameters(
            [C, class_weight, penalty, tol, solver])
        return cs


if __name__ == '__main__':
    # Add MLP classifier component to auto-sklearn.
    autosklearn.pipeline.components.classification.add_classifier(
        LogisticRegressionSK)
    cs = LogisticRegressionSK.get_hyperparameter_search_space()
    print(cs)

    # Generate data.
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # Fit MLP classifier to the data.
    clf = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=30,
        per_run_time_limit=10,
        include_estimators=['LogisticRegressionSK'],
    )
    clf.fit(X_train, y_train)

    # Print test accuracy and statistics.
    y_pred = clf.predict(X_test)
    print("accuracy: ", sklearn.metrics.accuracy_score(y_pred, y_test))
    print(clf.sprint_statistics())
    print(clf.show_models())
