#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
/Users/charleslai/PycharmProjects/diego/.vscode/diego.regressor.py was created on 2019/03/18.
file in :relativeFile
Author: Charles_Lai
Email: lai.bluejay@gmail.com
"""
from autosklearn.automl import BaseAutoML
from basic import *

class DiegoRegressor(BaseAutoML):
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
        X, y = super()._perform_input_checks(X, y)
        _n_outputs = 1 if len(y.shape) == 1 else y.shape[1]
        if _n_outputs > 1:
            raise NotImplementedError(
                'Multi-output regression is not implemented.')
        if metric is None:
            metric = r2
        return super().fit(
            X, y,
            X_test=X_test,
            y_test=y_test,
            task=REGRESSION,
            metric=metric,
            feat_type=feat_type,
            dataset_name=dataset_name,
            only_return_configuration_space=only_return_configuration_space,
            load_models=load_models,
        )

    def fit_ensemble(self, y, task=None, metric=None, precision='32',
                     dataset_name=None, ensemble_nbest=None,
                     ensemble_size=None):
        y = super()._check_y(y)
        return super().fit_ensemble(y, task, metric, precision, dataset_name,
                                    ensemble_nbest, ensemble_size)