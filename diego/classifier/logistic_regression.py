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
import torch.nn as nn
import torch

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

from lightning.classification import SGDClassifier, SVRGClassifier, SAGAClassifier, AdaGradClassifier
print(SGDClassifier)

class LogisticRegressionSMAC(AutoSklearnClassificationAlgorithm):
    def __init__(self, solver='svrg', loss="hinge", penalty="l2",
                 multiclass=False, alpha=0.01, gamma=1.0, tol=1e-4,
                 learning_rate="pegasos", eta=0.03, power_t=0.5,
                 epsilon=0.01, fit_intercept=True, intercept_decay=1.0,
                 max_iter=2048, shuffle=True, random_state=None,
                 callback=None, n_calls=100, verbose=0):
        self.solver = solver
    
        self.tol = tol
        self.loss = loss
        self.penalty = penalty
        self.multiclass = multiclass
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.eta = eta
        self.gamma = gamma
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = verbose
        self.estimator=None
        super(LogisticRegressionSMAC, self).__init__()

    def fit(self, X, y):
        
        try:
            #TODO remove svrg, performance no good with same parameter space with sgd
            if self.solver == 'svrg':
            
                    self.estimator = SVRGClassifier(loss=self.loss, alpha=self.alpha, tol=self.tol,
                                                eta=self.eta, gamma=self.gamma, max_iter=self.max_iter, verbose=self.verbose)                
            elif self.solver == 'sgd':
                self.estimator = SGDClassifier(loss=self.loss, alpha=self.alpha, penalty=self.penalty,
                                            eta0=self.eta, max_iter=self.max_iter, multiclass=self.multiclass,
                                            learning_rate=self.learning_rate, verbose=self.verbose)
            else:
                raise NotImplementedError
                print('No impelement solver of {}'.format(self.solver))
        except Exception as e:
                print(self.solver, e)
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
        return {'shortname': 'LogisticRegressionSMAC',
                'name': 'LogisticRegressionSMAC',
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

        loss = CategoricalHyperparameter(
            name='loss', choices=['modified_huber', 'smooth_hinge', 'log', 'squared', 'squared_hinge'], 
            default_value='smooth_hinge'
        )

        # weight of penalty
        alpha = UniformFloatHyperparameter(
            name='alpha', lower=1e-4, upper=1.0, default_value=0.01
        )

        # step size
        eta = UniformFloatHyperparameter(
            name='eta', lower=1e-4, upper=0.1, default_value=0.03
        )

        # gamma = UniformFloatHyperparameter(
        #     name='gamma', lower=0.01, upper=10.0, default_value=1.0
        # )

        learning_rate = CategoricalHyperparameter(
            name='learning_rate', choices=['pegasos', 'constant', 'invscaling'], default_value='pegasos'
        )

        penalty = CategoricalHyperparameter(
            name='penalty', choices=['l1', 'l2', 'l1/l2'], default_value='l2'
        )

        tol = CategoricalHyperparameter(
            name='tol', choices=[1e-5, 1e-4, 1e-3], default_value=1e-4
        )

        max_iter = 2048

        solver = CategoricalHyperparameter(
            name="solver", choices=['sgd'], default_value='sgd'
        )
        cs.add_hyperparameters(
            [loss, alpha, eta, learning_rate, penalty, tol, solver])
        return cs


if __name__ == '__main__':
    # Add MLP classifier component to auto-sklearn.
    autosklearn.pipeline.components.classification.add_classifier(
        LogisticRegressionSMAC)
    cs = LogisticRegressionSMAC.get_hyperparameter_search_space()
    # print(cs)

    # Generate data.
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # Fit MLP classifier to the data.
    clf = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=30,
        per_run_time_limit=10,
        include_estimators=['LogisticRegressionSMAC'],
    )
    clf.fit(X_train, y_train)

    # Print test accuracy and statistics.
    y_pred = clf.predict(X_test)
    print("accuracy: ", sklearn.metrics.accuracy_score(y_pred, y_test))
    print(clf.sprint_statistics())
    print(clf.show_models())
