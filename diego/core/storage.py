#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
core/storage.py was created on 2019/03/21.
file in :relativeFile
Author: Charles_Lai
Email: lai.bluejay@gmail.com
"""

import copy
from datetime import datetime
import threading
import uuid
import numpy as np
from tpot import TPOTClassifier

from diego import basic

from typing import Any  # NOQA
from typing import Dict  # NOQA
from typing import List  # NOQA
from typing import Optional  # NOQA

DEFAULT_STUDY_NAME_PREFIX = 'no-name-'

def generate_uuid():
    import time
    random_uuid = uuid.uuid4()
    st = str(time.time())
    nid = uuid.uuid5(random_uuid, str(st))
    return nid

class InMemoryStorage(object):
    """Storage class that stores data in memory of the Python process.
    
    This class is not supposed to be directly accessed by library users.

    get trails : 1.id, 2. clf, 3. best_thins={}  4.state etc.
    """

    def __init__(self):
        # type: () -> None
        self.trials = []  # type: List[basic.FrozenTrial]
        self._direction = basic.StudyDirection.NOT_SET
        self._study_user_attrs = {}  # type: Dict[str, Any]
        self._study_system_attrs = {}  # type: Dict[str, Any]
        self.study_uuid = generate_uuid()
        self.study_name = DEFAULT_STUDY_NAME_PREFIX + str(self.study_uuid)  # type: str
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self._lock = threading.Lock()

    def __getstate__(self):
        # type: () -> Dict[Any, Any]
        state = self.__dict__.copy()
        del state['_lock']
        return state

    def __setstate__(self, state):
        # type: (Dict[Any, Any]) -> None
        self.__dict__.update(state)
        self._lock = threading.Lock()

    def create_new_study_id(self, study_name=None):
        # type: (Optional[str]) -> int

        if study_name is not None:
            self.study_name = study_name
            
        return self.study_uuid

    @property
    def direction(self):
        return self._direction

    @direction.setter
    def direction(self, direction):
        with self._lock:
            if self._direction != basic.StudyDirection.NOT_SET and self._direction != direction:
                raise ValueError('Cannot overwrite study direction from {} to {}.'.format(
                    self._direction, direction))
            self._direction = direction
    
    @property
    def study_user_attrs(self):
        with self._lock:
            return copy.deepcopy(self._study_user_attrs)
    
    @study_user_attrs.setter
    def study_user_attrs(self, key, value):
        with self._lock:
            self._study_user_attrs[key] = value
    
    @property
    def study_system_attrs(self):
        with self._lock:
            return copy.deepcopy(self._study_system_attrs)

    @study_system_attrs.setter
    def study_system_attrs(self, key, value):
        with self._lock:
            self._study_system_attrs[key] = value


    def set_train_storage(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def set_test_storage(self, X, y):
        self.X_test = X
        self.y_test = y

    def get_study_id_from_name(self, study_name):
        # type: (str) -> int

        if study_name != self.study_name:
            raise ValueError("No such study {}.".format(study_name))

        return self.study_uuid

    def get_study_id_from_trial_id(self, trial_id):
        # type: (int) -> int

        return self.study_uuid

    def get_study_name_from_id(self, study_id):
        # type: (int) -> str

        self._check_study_id(study_id)
        return self.study_name

    def get_all_study_summaries(self):
        # type: () -> List[basic.StudySummary]

        best_trial = None
        n_complete_trials = len([t for t in self.trials if t.state == basic.TrialState.COMPLETE])
        if n_complete_trials > 0:
            best_trial = self.get_best_trial(self.study_uuid)

        datetime_start = None
        if len(self.trials) > 0:
            datetime_start = min([t.datetime_start for t in self.trials])

        return [
            basic.StudySummary(
                study_id=self.study_uuid,
                study_name=self.study_name,
                direction=self._direction,
                best_trial=best_trial,
                user_attrs=copy.deepcopy(self._study_user_attrs),
                system_attrs=copy.deepcopy(self._study_system_attrs),
                n_trials=len(self.trials),
                datetime_start=datetime_start)
        ]

    def get_best_trial(self, study_id):
        # type: (int) -> basic.FrozenTrial

        all_trials = self.get_all_trials(study_id)
        all_trials = [t for t in all_trials if t.state is basic.TrialState.COMPLETE]
        if len(all_trials) == 0:
            raise ValueError('No trials are completed yet.')

        if self.direction == basic.StudyDirection.MAXIMIZE:
            return max(all_trials, key=lambda t: t.value)
        return min(all_trials, key=lambda t: t.value)

    def create_new_trial_id(self, study_id):
        # type: (int) -> int

        self._check_study_id(study_id)
        with self._lock:
            trial_id = len(self.trials)
            self.trials.append(
                basic.FrozenTrial(
                    number=trial_id,
                    state=basic.TrialState.RUNNING,
                    params={},
                    user_attrs={},
                    system_attrs={'_number': trial_id},
                    value=None,
                    intermediate_values={},
                    params_in_internal_repr={},
                    datetime_start=datetime.now(),
                    datetime_complete=None,
                    trial_id=trial_id,
                    clf=None,
                    clf_params=None))
        return trial_id

    def set_trial_state(self, trial_id, state):
        # type: (int, basic.TrialState) -> None

        with self._lock:
            self.trials[trial_id] = self.trials[trial_id]._replace(state=state)
            if state.is_finished():
                self.trials[trial_id] = \
                    self.trials[trial_id]._replace(datetime_complete=datetime.now())

    def get_trial_number_from_id(self, trial_id):
        # type: (int) -> int

        return trial_id

    def get_trial_param(self, trial_id, param_name):
        # type: (int, str) -> float

        return self.trials[trial_id].params_in_internal_repr[param_name]

    def set_trial_value(self, trial_id, value):
        # type: (int, float) -> None

        with self._lock:
            self.trials[trial_id] = self.trials[trial_id]._replace(value=value)

    def set_trial_clf(self, trial_id, clf):
        with self._lock:
            if isinstance(clf, TPOTClassifier):
                clf = clf.fitted_pipeline_
            self.trials[trial_id] = self.trials[trial_id]._replace(clf=clf)

    def set_trial_intermediate_value(self, trial_id, step, intermediate_value):
        # type: (int, int, float) -> bool

        with self._lock:
            values = self.trials[trial_id].intermediate_values
            if step in values:
                return False

            values[step] = intermediate_value

            return True

    def set_trial_user_attr(self, trial_id, key, value):
        # type: (int, str, Any) -> None

        with self._lock:
            self.trials[trial_id].user_attrs[key] = value

    def set_trial_system_attr(self, trial_id, key, value):
        # type: (int, str, Any) -> None

        with self._lock:
            self.trials[trial_id].system_attrs[key] = value

    def get_trial(self, trial_id):
        # type: (int) -> basic.FrozenTrial

        with self._lock:
            return copy.deepcopy(self.trials[trial_id])

    def get_all_trials(self, study_id):
        # type: (int) -> List[basic.FrozenTrial]

        self._check_study_id(study_id)
        with self._lock:
            return copy.deepcopy(self.trials)

    def get_n_trials(self, study_id, state=None):
        # type: (int, Optional[basic.]TrialState]) -> int

        self._check_study_id(study_id)
        if state is None:
            return len(self.trials)

        return len([t for t in self.trials if t.state == state])

    def _check_study_id(self, study_id):
        # type: (int) -> None

        if study_id != self.study_uuid:
            raise ValueError('study_id is supposed to be {} in {}.'.format(
                self.study_uuid, self.__class__.__name__))
    
    def get_trial_user_attrs(self, trial_id):
        # type: (int) -> Dict[str, Any]

        return self.get_trial(trial_id).user_attrs

    def get_trial_system_attrs(self, trial_id):
        # type: (int) -> Dict[str, Any]

        return self.get_trial(trial_id).system_attrs

    # Methods for the TPE sampler

    def get_trial_param_result_pairs(self, study_id, param_name):
        # type: (int, str) -> List[Tuple[float, float]]

        # Be careful: this method returns param values in internal representation
        all_trials = self.get_all_trials(study_id)

        return [(t.params_in_internal_repr[param_name], t.value) for t in all_trials
                if (t.value is not None and param_name in t.params
                    and t.state is basic.TrialState.COMPLETE)
                ]

    # Methods for the median pruner

    def get_best_intermediate_result_over_steps(self, trial_id):
        # type: (int) -> float

        values = np.array(list(self.get_trial(trial_id).intermediate_values.values()), np.float)

        study_id = self.get_study_id_from_trial_id(trial_id)
        if self.get_study_direction(study_id) == basic.StudyDirection.MAXIMIZE:
            return np.nanmax(values)
        return np.nanmin(values)

    def get_median_intermediate_result_over_trials(self, study_id, step):
        # type: (int, int) -> float

        all_trials = [
            t for t in self.get_all_trials(study_id) if t.state == basic.TrialState.COMPLETE
        ]

        if len(all_trials) == 0:
            raise ValueError("No trials have been completed.")

        return float(
            np.nanmedian(
                np.array([
                    t.intermediate_values[step]
                    for t in all_trials if step in t.intermediate_values
                ], np.float)))

    def remove_session(self):
        # type: () -> None

        pass