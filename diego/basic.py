#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
/Users/charleslai/PycharmProjects/diego/.vscode/diego.basic.py was created on 2019/03/18.
file in :relativeFile
Author: Charles_Lai
Email: lai.bluejay@gmail.com
"""
# python 3.7
# from dataclasses import dataclass
from typing import Optional
from typing import NamedTuple
from typing import Dict
from typing import Any
import enum
from datetime import datetime

BINARY_CLASSIFICATION = 1
MULTICLASS_CLASSIFICATION = 2
MULTILABEL_CLASSIFICATION = 3
REGRESSION = 4

REGRESSION_TASKS = [REGRESSION]
CLASSIFICATION_TASKS = [BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION,
                        MULTILABEL_CLASSIFICATION]

TASK_TYPES = REGRESSION_TASKS + CLASSIFICATION_TASKS

TASK_TYPES_TO_STRING = \
    {BINARY_CLASSIFICATION: 'binary.classification',
     MULTICLASS_CLASSIFICATION: 'multiclass.classification',
     MULTILABEL_CLASSIFICATION: 'multilabel.classification',
     REGRESSION: 'regression'}

STRING_TO_TASK_TYPES = \
    {'binary.classification': BINARY_CLASSIFICATION,
     'multiclass.classification': MULTICLASS_CLASSIFICATION,
     'multilabel.classification': MULTILABEL_CLASSIFICATION,
     'regression': REGRESSION}


class TrialState(enum.Enum):
    """State of a :class:`~diego.trial.Trial`.

    Attributes:
        RUNNING:
            The :class:`~diego.trial.Trial` is running.
        COMPLETE:
            The :class:`~diego.trial.Trial` has been finished without any error.
        PRUNED:
            The :class:`~diego.trial.Trial` has been pruned with :class:`TrialPruned`.
        FAIL:
            The :class:`~diego.trial.Trial` has failed due to an uncaught error.
    """

    RUNNING = 0
    COMPLETE = 1
    PRUNED = 2
    FAIL = 3

    def is_finished(self):
        # type: () -> bool

        return self == TrialState.COMPLETE or self == TrialState.PRUNED


class StudyDirection(enum.Enum):
    """Direction of a :class:`~diego.study.Study`.

    Attributes:
        NOT_SET:
            Direction has not been set.
        MNIMIZE:
            :class:`~diego.study.Study` minimizes the objective function.
        MAXIMIZE:
            :class:`~diego.study.Study` maximizes the objective function.
    """

    NOT_SET = 0
    MINIMIZE = 1
    MAXIMIZE = 2

# python 3.7
# @dataclass
class BaseFrozenTrial(NamedTuple):
    """Status and results of a :class:`~diego.trial.Trial`.

    Attributes:
        number:
            Unique and consecutive number of :class:`~diego.trial.Trial` for each
            :class:`~diego.study.Study`. Note that this field uses zero-based numbering.
        state:
            :class:`TrialState` of the :class:`~diego.trial.Trial`.
        value:
            Objective value of the :class:`~diego.trial.Trial`.
        datetime_start:
            Datetime where the :class:`~diego.trial.Trial` started.
        datetime_complete:
            Datetime where the :class:`~diego.trial.Trial` finished.
        params:
            Dictionary that contains suggested parameters.
        user_attrs:
            Dictionary that contains the attributes of the :class:`~diego.trial.Trial` set with
            :func:`diego.trial.Trial.set_user_attr`.
        system_attrs:
            Dictionary that contains the attributes of the :class:`~diego.trial.Trial` internally
            set by diego.
        intermediate_values:
            Intermediate objective values set with :func:`diego.trial.Trial.report`.
        params_in_internal_repr:
            diego's internal representation of :attr:`params`. Note that this field is not
            supposed to be used by library users.
        trial_id:
            diego's internal identifier of the :class:`~diego.trial.Trial`. Note that this field
            is not supposed to be used by library users. Instead, please use :attr:`number` and
            :class:`~diego.study.Study.study_id` to identify a :class:`~diego.trial.Trial`.
    """
    number: int
    state: TrialState
    value: Optional[float]
    params: Dict[str, Any]
    datetime_start: Optional[datetime]
    datetime_complete: Optional[datetime]
    user_attrs: Dict[str, Any]
    system_attrs: Dict[str, Any]
    intermediate_values: Dict[int, float]
    params_in_internal_repr: Dict[str, float]
    trial_id: int
    clf: Any
    clf_params: Dict[str, Any]

class FrozenTrial(BaseFrozenTrial):
    internal_fields = ['params_in_internal_repr', 'trial_id']

class StudySummary(
        NamedTuple('StudySummary', [('study_id', int), ('study_name', str),
                                    ('direction', StudyDirection),
                                    ('best_trial', Optional[FrozenTrial]),
                                    ('user_attrs', Dict[str, Any]),
                                    ('system_attrs',
                                     Dict[str, Any]), ('n_trials', int),
                                    ('datetime_start', Optional[datetime])])):
    """Basic attributes and aggregated results of a :class:`~diego.study.Study`.

    See also :func:`diego.study.get_all_study_summaries`.

    Attributes:
        study_id:
            Identifier of the :class:`~diego.study.Study`.
        study_name:
            Name of the :class:`~diego.study.Study`.
        direction:
            :class:`StudyDirection` of the :class:`~diego.study.Study`.
        best_trial:
            :class:`FrozenTrial` with best objective value in the :class:`~diego.study.Study`.
        user_attrs:
            Dictionary that contains the attributes of the :class:`~diego.study.Study` set with
            :func:`diego.study.Study.set_user_attr`.
        system_attrs:
            Dictionary that contains the attributes of the :class:`~diego.study.Study` internally
            set by diego.
        n_trials:
            The number of trials ran in the :class:`~diego.study.Study`.
        datetime_start:
            Datetime where the :class:`~diego.study.Study` started.
    """

def get_approp_n_jobs(n_jobs=-1):
    """
    if core > 2, return core/2+2；
    if core
    """

    import multiprocessing
    max_jobs = multiprocessing.cpu_count()
    if n_jobs == -1:
        n_jobs = max_jobs
    if max_jobs >= 4:
        max_jobs = int(max_jobs/2) + 2
    n_jobs = min(n_jobs, max_jobs)
    return n_jobs