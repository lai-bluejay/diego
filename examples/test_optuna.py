#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
examples/test_optuna.py was created on 2019/03/21.
file in :relativeFile
Author: Charles_Lai
Email: lai.bluejay@gmail.com
"""

import sklearn.datasets
import sklearn.ensemble
import sklearn.model_selection
import sklearn.svm


# FYI: Objective functions can take additional arguments
# (https://optuna.readthedocs.io/en/stable/faq.html#objective-func-additional-args).
def objective(trial):
    digits = sklearn.datasets.load_digits()
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(digits.data, digits.target,
                                                        train_size=0.75, test_size=0.25)

    classifier_name = trial.suggest_categorical('classifier', ['SVC', 'RandomForest'])
    if classifier_name == 'SVC':
        svc_c = trial.suggest_loguniform('svc_c', 1e-10, 1e10)
        classifier_obj = sklearn.svm.SVC(C=svc_c)
    else:
        rf_max_depth = int(trial.suggest_loguniform('rf_max_depth', 2, 32))
        classifier_obj = sklearn.ensemble.RandomForestClassifier(max_depth=rf_max_depth)

    score = sklearn.model_selection.cross_val_score(classifier_obj, x, y, n_jobs=-1, cv=3)
    accuracy = score.mean()
    return 1.0 - accuracy


if __name__ == '__main__':
    import optuna
    study = optuna.create_study()
    study.optimize(objective, n_trials=10)
    print(study.best_trial)