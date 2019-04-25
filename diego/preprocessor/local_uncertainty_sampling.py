#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BoeingML.local_uncertainty_sampling was created on 2017/8/9.
Author: Charles_Lai
Email: lai.bluejay@gmail.com
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.linear_model import LogisticRegression
import autosklearn.pipeline.components.feature_preprocessing

from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import DENSE, SIGNED_DATA, \
    UNSIGNED_DATA
from diego.depens import logging

class LocalUncertaintySampling(BaseEstimator, TransformerMixin):

    """Local Uncertainty Sampling

    This class implements Local Uncertainty Sampling active learning algorithm [1]_.

    Parameters
    ----------
    model: :py:class:`libact.base.interfaces.ContinuousModel` or :py:class:`libact.base.interfaces.ProbabilisticModel` object instance
        The base model used for training.

    method: {'lc', 'sm', 'entropy'}, optional (default='lc')
        least confidence (lc), it queries the instance whose posterior
        probability of being positive is nearest 0.5 (for binary
        classification);
        smallest margin (sm), it queries the instance whose posterior
        probability gap between the most and the second probable labels is
        minimal;
        entropy, requires :py:class:`libact.base.interfaces.ProbabilisticModel`
        to be passed in as model parameter;

    Attributes
    ----------
    model: :py:class:`libact.base.interfaces.ContinuousModel` or :py:class:`libact.base.interfaces.ProbabilisticModel` object instance
        The model trained in last query.

    Examples
    --------
    Here is an example of declaring a UncertaintySampling query_strategy
    object:

    .. code-block:: python

       from libact.query_strategies import UncertaintySampling
       from libact.models import LogisticRegression

       qs = UncertaintySampling(
                dataset, # Dataset object
                model=LogisticRegression(C=0.1)
            )

    Note that the model given in the :code:`model` parameter must be a
    :py:class:`ContinuousModel` which supports predict_real method.


    References
    ----------

    @article{Han:2016aa,
	Author = {{Han}, L. and {Yang}, T. and {Zhang}, T.},
	Journal = {ArXiv e-prints},
	Month = apr,
	Title = {{Local Uncertainty Sampling for Large-Scale Multi-Class Logistic Regression}},
	Year = 2016}
    """

    def __init__(self, method='lus', model='lr', gamma=1.1):
        self.method = method
        self.logger = logging.get_logger(__name__)
        if model == 'lr':
            self.model = LogisticRegression(solver='sag',max_iter=400, verbose=0, class_weight='balanced', n_jobs=-1)
        self.logger.info('Local Uncertainty Sampling with {0} method, model is {1}'.format(method, self.model))
        self.gamma = gamma
        if self.method not in ['lus', 'sm', 'entropy']:
            raise TypeError(
                "supported methods are ['lc', 'sm', 'entropy'], the given one "
                "is: " + self.method
            )
        # if self.method=='entropy' and \
        #         not isinstance(self.model, ProbabilisticModel):
        #     raise TypeError(
        #         "method 'entropy' requires model to be a ProbabilisticModel"
        #     )
        if self.model is None:
            raise TypeError(
                "__init__() missing required keyword-only argument: 'model'"
            )
        # if not isinstance(self.model, ContinuousModel) and \
        #         not isinstance(self.model, ProbabilisticModel):
        #     raise TypeError(
        #         "model has to be a ContinuousModel or ProbabilisticModel"
        #     )

    def make_query(self, X, y, return_score=False):
        """Return the index of the sample to be queried and labeled and
        selection score of each sample. Read-only.

        No modification to the internal states.

        Returns
        -------
        ask_id : int
            The index of the next unlabeled sample to be queried and labeled.

        score : list of (index, score) tuple
            Selection score of unlabled entries, the larger the better.

        """
        #
        # unlabeled_entry_ids, X_pool = zip(*dataset.get_unlabeled_entries())
        X_pool = X
        y_train = y
        dvalue = self.model.predict_proba(X_pool)
        # if isinstance(self.model, ProbabilisticModel):
        #     dvalue = self.model.predict_proba(X_pool)
        # elif isinstance(self.model, ContinuousModel):
        #     dvalue = self.model.predict_real(X_pool)
        if self.method == 'lus':  # local_uncertainty sampling
            tmp_afunc = []
            sample_idx = []
            # 给 all q_i
            score = np.max(dvalue, axis=1)
            for idx, y in enumerate(y_train):
                # 获得qi
                v = score[idx]
                qi = max(0.5, v)
                pred = dvalue[idx][y]
                if pred == qi:
                    tmp = (1 - qi)/(self.gamma - max(qi, self.gamma * 0.5))
                else:
                    qi = 1 - v
                    tmp = min(1, 2 * qi/self.gamma)
                tmp_afunc.append(tmp)
                # 服从a(x, c)的bernoulli分布
                candidator = np.random.binomial(1, tmp)
                if candidator == 1:
                    sample_idx.append(idx)
                else:
                    continue

            """
            ===> 抽样完成
            """
        # elif self.method == 'sm':  # smallest margin
        #     if np.shape(dvalue)[1] > 2:
        #         # Find 2 largest decision values
        #         dvalue = -(np.partition(-dvalue, 2, axis=1)[:, :2])
        #     score = -np.abs(dvalue[:, 0] - dvalue[:, 1])
        #
        # elif self.method == 'entropy':
        #     score = np.sum(-dvalue * np.log(dvalue), axis=1)

        ask_id = np.argmax(score)

        if return_score:
            return sample_idx, score
        else:
            return sample_idx

    def fit(self, X, y):
            self.model.fit(X, y)
            return self

    def transform(self, X, y):
        sample_idx = self.make_query(X, y)
        sample_train = X[sample_idx, :]
        sample_y = y[sample_idx]
        return sample_train, sample_y

    def fit_transform(self, X, y, **fit_params):
        return self.fit(X, y, **fit_params).transform(X, y)

class LusSample(AutoSklearnPreprocessingAlgorithm):
    def __init__(self, model='lr', method='lus', gamma=1.1,  random_state=None):
        self.model = model
        self.gamma = gamma
        self.method = method
        self.random_state = random_state
        self.preprocessor = None

    def fit(self, X, y=None):
    
        self.preprocessor = LocalUncertaintySampling(
                model=self.model,
                method=self.method,
                gamma=self.gamma
            )
        self.preprocessor.fit(X, y)
        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)

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
        method = CategoricalHyperparameter(
            name="binning_method", choices=['ef', 'ew', 'xgb', 'modl'], default_value='ef'
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


if __name__ == '__main__':
    # Add LDA component to auto-sklearn.
    autosklearn.pipeline.components.feature_preprocessing.add_preprocessor(LusSample)

    # Create dataset.
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    print(X_train.shape)
    from sklearn.linear_model import RidgeClassifier
    clf = RidgeClassifier().fit(X_train, y_train)
    print(clf.score(X_test, y_test))

    print('lus')
    pp = LocalUncertaintySampling()
    X_train, y_train = pp.fit_transform(X_train, y_train)
    print(X_train.shape)
    # X_test = pp.transform(X_test)
    clf = RidgeClassifier().fit(X_train, y_train)
    print(clf.score(X_test, y_test))
    print('='*100)
    # Configuration space.
    cs = AutoBinning.get_hyperparameter_search_space()
    print(cs)

    # Fit the model using LDA as preprocessor.
    clf = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=30,
        include_preprocessors=['AutoBinning'],
    )
    clf.fit(X_train, y_train)

    # Print prediction score and statistics.
    y_pred = clf.predict(X_test)
    print("accracy: ", sklearn.metrics.accuracy_score(y_pred, y_test))
    print(clf.sprint_statistics())
    print('='*50, 'models', "="*50)
    print(clf.show_models())

    clf.fit_ensemble(y_train, ensemble_size=50)
    print('='*50, 'ensemble', "="*50)

    print(clf.show_models())
    predictions = clf.predict(X_test)
    print(clf.sprint_statistics())
    print("Accuracy score", sklearn.metrics.accuracy_score(y_test, predictions))