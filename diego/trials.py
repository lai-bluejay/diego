#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
diego/trials.py was created on 2019/03/21.
file in :relativeFile
Author: Charles_Lai
Email: lai.bluejay@gmail.com
"""

import abc
import math
import six
import warnings

from diego.depens import logging


from typing import Any  # NOQA
from typing import Dict  # NOQA
from typing import Optional  # NOQA
from typing import Sequence  # NOQA
from typing import TypeVar  # NOQA

T = TypeVar('T', float, str)


@six.add_metaclass(abc.ABCMeta)
class BaseTrial(object):
    """Base class for trials.

    Note that this class is not supposed to be directly accessed by library users.
    """

    def suggest_uniform(self, name, low, high):
        # type: (str, float, float) -> float

        raise NotImplementedError

    def suggest_loguniform(self, name, low, high):
        # type: (str, float, float) -> float

        raise NotImplementedError

    def suggest_discrete_uniform(self, name, low, high, q):
        # type: (str, float, float, float) -> float

        raise NotImplementedError

    def suggest_int(self, name, low, high):
        # type: (str, int, int) -> int

        raise NotImplementedError

    def suggest_categorical(self, name, choices):
        # type: (str, Sequence[T]) -> T

        raise NotImplementedError

    def report(self, value, step=None):
        # type: (float, Optional[int]) -> None

        raise NotImplementedError

    def should_prune(self, step):
        # type: (int) -> bool

        raise NotImplementedError

    def set_user_attr(self, key, value):
        # type: (str, Any) -> None

        raise NotImplementedError

    def set_system_attr(self, key, value):
        # type: (str, Any) -> None

        raise NotImplementedError

    @property
    def user_attrs(self):
        # type: () -> Dict[str, Any]

        raise NotImplementedError

    @property
    def system_attrs(self):
        # type: () -> Dict[str, Any]

        raise NotImplementedError

    @property
    def clf(self):
        return self._clf
    
    @clf.setter
    def clf(self, clf):
        self._clf = clf
    
    @property
    def clf_params(self):
        return self._clf_params
    
    @clf_params.setter
    def clf_params(self, params):
        self._clf_params = params

class Trial(BaseTrial):
    """A trial is a process of evaluating an objective function.

    This object is passed to an objective function and provides interfaces to get parameter
    suggestion, manage the trial's state, and set/get user-defined attributes of the trial.

    Note that this object is seamlessly instantiated and passed to the objective function behind
    :func:`diego.study.Study.optimize()` method (as well as optimize function); hence, in typical
    use cases, library users do not care about instantiation of this object.

    Args:
        study:
            A :class:`~diego.study.Study` object.
        trial_id:
            A trial ID that is automatically generated.

    """

    def __init__(self, study, trial_id):
        # type: (Study, int) -> None

        self.study = study
        self._trial_id = trial_id

        self.study_id = self.study.study_id
        self.storage = self.study.storage
        self._clf = None
        self._clf_params = {}
        self.logger = logging.get_logger(__name__)

    def report(self, value, step=None):
        # type: (float, Optional[int]) -> None
        """Report an objective function value.

        If step is set to :obj:`None`, the value is stored as a final value of the trial.
        Otherwise, it is saved as an intermediate value.

        Example:

            Report intermediate scores of `SGDClassifier <https://scikit-learn.org/stable/modules/
            generated/sklearn.linear_model.SGDClassifier.html>`_ training

            .. code::

                >>> def objective(trial):
                >>>     ...
                >>>     clf = sklearn.linear_model.SGDClassifier()
                >>>     for step in range(100):
                >>>         clf.partial_fit(x_train , y_train , classes)
                >>>         intermediate_value = clf.score(x_val , y_val)
                >>>         trial.report(intermediate_value , step=step)
                >>>         if trial.should_prune(step):
                >>>             raise TrialPruned()
                >>>     ...

        Args:
            value:
                A value returned from the objective function.
            step:
                Step of the trial (e.g., Epoch of neural network training).
        """

        self.storage.set_trial_value(self._trial_id, value)
        self.storage.set_trial_clf(self._trial_id, self.clf)
        if step is not None:
            self.storage.set_trial_intermediate_value(self._trial_id, step, value)

    def set_user_attr(self, key, value):
        # type: (str, Any) -> None
        """Set user attributes to the trial.

        The user attributes in the trial can be access via :func:`diego.trial.Trial.user_attrs`.

        Example:

            Save fixed hyperparameters of neural network training:

            .. code::

                >>> def objective(trial):
                >>>     ...
                >>>     trial.set_user_attr('BATCHSIZE', 128)
                >>>
                >>> study.best_trial.user_attrs
                {'BATCHSIZE': 128}


        Args:
            key:
                A key string of the attribute.
            value:
                A value of the attribute. The value should be JSON serializable.
        """

        self.storage.study_user_attrs(key, value)

    def set_system_attr(self, key, value):
        # type: (str, Any) -> None
        """Set system attributes to the trial.

        Note that diego internally uses this method to save system messages such as failure
        reason of trials. Please use :func:`~diego.trial.Trial.set_user_attr` to set users'
        attributes. 

        Args:
            key:
                A key string of the attribute.
            value:
                A value of the attribute. The value should be JSON serializable.
        """

        self.storage.study_system_attrs(key, value)

    @property
    def number(self):
        # type: () -> int
        """Return trial's number which is consecutive and unique in a study.

        Returns:
            A trial number.
        """

        return self.storage.get_trial_number_from_id(self._trial_id)

    @property
    def trial_id(self):
        # type: () -> int
        """Return trial ID.

        Note that the use of this is deprecated.
        Please use :attr:`~diego.trial.Trial.number` instead.

        Returns:
            A trial ID.
        """

        warnings.warn(
            'The use of `Trial.trial_id` is deprecated. '
            'Please use `Trial.number` instead.', DeprecationWarning)

        self.logger.warning('The use of `Trial.trial_id` is deprecated. '
                            'Please use `Trial.number` instead.')

        return self._trial_id

    @property
    def user_attrs(self):
        # type: () -> Dict[str, Any]
        """Return user attributes.

        Returns:
            A dictionary containing all user attributes.
        """

        return self.storage.study_user_attrs

    @property
    def system_attrs(self):
        # type: () -> Dict[str, Any]
        """Return system attributes.

        Returns:
            A dictionary containing all system attributes.
        """

        return self.storage.study_system_attrs


class FixedTrial(BaseTrial):
    """A trial class which suggests a fixed value for each parameter.

    This object has the same methods as :class:`~diego.trial.Trial`, and it suggests pre-defined
    parameter values. The parameter values can be determined at the construction of the
    :class:`~diego.trial.FixedTrial` object. In contrast to :class:`~diego.trial.Trial`,
    :class:`~diego.trial.FixedTrial` does not depend on :class:`~diego.study.Study`, and it is
    useful for deploying optimization results.

    Example:

        Evaluate an objective function with parameter values given by a user:

        .. code::

            >>> def objective(trial):
            >>>     x = trial.suggest_uniform('x', -100, 100)
            >>>     y = trial.suggest_categorical('y', [-1, 0, 1])
            >>>     return x ** 2 + y
            >>>
            >>> objective(FixedTrial({'x': 1, 'y': 0}))
            1

    .. note::
        Please refer to :class:`~diego.trial.Trial` for details of methods and properties.

    Args:
        params:
            A dictionary containing all parameters.

    """

    def __init__(self, params):
        # type: (Dict[str, Any]) -> None

        self._params = params
        self._user_attrs = {}  # type: Dict[str, Any]
        self._system_attrs = {}  # type: Dict[str, Any]

    def report(self, value, step=None):
        # type: (float, Optional[int]) -> None

        pass

    def should_prune(self, step):
        # type: (int) -> bool

        return False

    def set_user_attr(self, key, value):
        # type: (str, Any) -> None

        self._user_attrs[key] = value

    def set_system_attr(self, key, value):
        # type: (str, Any) -> None

        self._system_attrs[key] = value

    @property
    def params(self):
        # type: () -> Dict[str, Any]

        return self._params

    @property
    def user_attrs(self):
        # type: () -> Dict[str, Any]

        return self._user_attrs

    @property
    def system_attrs(self):
        # type: () -> Dict[str, Any]

        return self._system_attrs

