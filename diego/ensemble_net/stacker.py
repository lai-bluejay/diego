#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
conan.stacker was created on 2017/10/17.
Author: Charles_Lai
Email: lai.bluejay@gmail.com
"""
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from diego.ensemble_net.base import Ensemble
from diego.ensemble_net.combination import Combiner
from diego.classifier.logistic_regression_sk import LogisticRegressionSK

class EnsembleStack(object):
    def __init__(self, mode='probs', cv=5):
        self.mode = mode
        self.layers = []
        self.cv = cv

    def add_layer(self, ensemble):
        if isinstance(ensemble, Ensemble):
            self.layers.append(ensemble)
        else:
            raise Exception('not an Ensemble object')

    def fit_layer(self, layer_idx, X, y):
        if layer_idx >= len(self.layers):
            return
        elif layer_idx == len(self.layers) - 1:
            self.layers[layer_idx].fit(X, y)
        else:
            n_classes = len(set(y)) - 1
            n_classifiers = len(self.layers[layer_idx])
            output = np.zeros((X.shape[0], n_classes * n_classifiers))
            skf = list(StratifiedKFold(self.cv).split(X, y))
            for tra, tst in skf:
                self.layers[layer_idx].fit(X[tra], y[tra])
                out = self.layers[layer_idx].output(X[tst], mode=self.mode)
                output[tst, :] = out[:, 1:, :].reshape(
                    out.shape[0], (out.shape[1] - 1) * out.shape[2])

            self.layers[layer_idx].fit(X, y)
            self.fit_layer(layer_idx + 1, output, y)

    def fit(self, X, y):
        if self.cv > 1:
            self.fit_layer(0, X, y)
        else:
            X_ = X
            for layer in self.layers:
                layer.fit(X_, y)
                out = layer.output(X_, mode=self.mode)
                X_ = out[:, 1:, :].reshape(
                    out.shape[0], (out.shape[1] - 1) * out.shape[2])

        return self

    def output(self, X):
        input_ = X

        for layer in self.layers:
            out = layer.output(input_, mode=self.mode)
            input_ = out[:, 1:, :].reshape(
                out.shape[0], (out.shape[1] - 1) * out.shape[2])
        return input_


class EnsembleStackClassifier(object):
    def __init__(self, stack, combiner=None):
        self.stack = stack
        if combiner is None:
            self.combiner = Combiner(rule='mean')
        elif isinstance(combiner, str):
            if combiner == 'majority_vote':
                raise ValueError('EnsembleStackClassifier '
                                 'do not support majority_vote')
            self.combiner = Combiner(rule=combiner)
        elif isinstance(combiner, Combiner):
            self.combiner = combiner
        else:
            raise ValueError('Invalid combiner!')

        self.clf = self._make_clf()

    @staticmethod
    def _make_clf():
        import autosklearn.classification
        import autosklearn.pipeline.components.classification
        autosklearn.pipeline.components.classification.add_classifier(
        LogisticRegressionSK)
        clf = autosklearn.classification.AutoSklearnClassifier(
            time_left_for_this_task=30,
            per_run_time_limit=10,
            include_estimators=['LogisticRegressionSK'],
        )
        return clf


    def fit(self, X, y):
        self.stack.fit(X, y)
        return self

    def refit(self, X, y):
        out = self.stack.output(X)
    
        self.clf.fit(out, y)

    def predict(self, X):
        out = self.stack.output(X)
        try:
            y_pred = self.clf.predict(out)
        except:
            raise Exception('You must refit ensemble stacker')
        return y_pred

    def output(self, X):
        out = self.stack.output(X)
        return self.combiner.combine(out)

    def output_proba(self, X):
        out = self.stack.output(X)
        return np.mean(out, axis=2)