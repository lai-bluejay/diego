#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
/Users/charleslai/PycharmProjects/diego/.vscode/diego.classifier.py was created on 2019/03/18.
file in :relativeFile
Author: Charles_Lai
Email: lai.bluejay@gmail.com
"""
from autosklearn.automl import BaseAutoML
from basic import *


class DiegoClassifier(BaseAutoML):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._task_mapping = {'multilabel-indicator': MULTILABEL_CLASSIFICATION,
                              'multiclass': MULTICLASS_CLASSIFICATION,
                              'binary': BINARY_CLASSIFICATION}

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
        X, y = self._perform_input_checks(X, y)
        if X_test is not None:
            X_test, y_test = self._perform_input_checks(X_test, y_test)
            if len(y.shape) != len(y_test.shape):
                raise ValueError('Target value shapes do not match: %s vs %s'
                                 % (y.shape, y_test.shape))

        y_task = type_of_target(y)
        task = self._task_mapping.get(y_task)
        if task is None:
            raise ValueError('Cannot work on data of type %s' % y_task)

        if metric is None:
            if task == MULTILABEL_CLASSIFICATION:
                metric = f1_macro
            else:
                metric = accuracy

        y, self._classes, self._n_classes = self._process_target_classes(y)
        if y_test is not None:
            # Map test values to actual values - TODO: copy to all kinds of
            # other parts in this code and test it!!!
            y_test_new = []
            for output_idx in range(len(self._classes)):
                mapping = {self._classes[output_idx][idx]: idx
                           for idx in range(len(self._classes[output_idx]))}
                enumeration = y_test if len(self._classes) == 1 else y_test[output_idx]
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
        y, _classes, _n_classes = self._process_target_classes(y)
        if not hasattr(self, '_classes'):
            self._classes = _classes
        if not hasattr(self, '_n_classes'):
            self._n_classes = _n_classes

        return super().fit_ensemble(y, task, metric, precision, dataset_name,
                                    ensemble_nbest, ensemble_size)

    def _process_target_classes(self, y):
        y = super()._check_y(y)
        self._n_outputs = 1 if len(y.shape) == 1 else y.shape[1]

        y = np.copy(y)

        _classes = []
        _n_classes = []

        if self._n_outputs == 1:
            classes_k, y = np.unique(y, return_inverse=True)
            _classes.append(classes_k)
            _n_classes.append(classes_k.shape[0])
        else:
            for k in range(self._n_outputs):
                classes_k, y[:, k] = np.unique(y[:, k], return_inverse=True)
                _classes.append(classes_k)
                _n_classes.append(classes_k.shape[0])

        _n_classes = np.array(_n_classes, dtype=np.int)

        return y, _classes, _n_classes

    def predict(self, X, batch_size=None, n_jobs=1):
        predicted_probabilities = super().predict(X, batch_size=batch_size,
                                                  n_jobs=n_jobs)

        if self._n_outputs == 1:
            predicted_indexes = np.argmax(predicted_probabilities, axis=1)
            predicted_classes = self._classes[0].take(predicted_indexes)

            return predicted_classes
        else:
            predicted_indices = (predicted_probabilities > 0.5).astype(int)
            n_samples = predicted_probabilities.shape[0]
            predicted_classes = np.zeros((n_samples, self._n_outputs))

            for k in range(self._n_outputs):
                output_predicted_indexes = predicted_indices[:, k].reshape(-1)
                predicted_classes[:, k] = self._classes[k].take(
                    output_predicted_indexes)

            return predicted_classes

    def predict_proba(self, X, batch_size=None, n_jobs=1):
        return super().predict(X, batch_size=batch_size, n_jobs=n_jobs)