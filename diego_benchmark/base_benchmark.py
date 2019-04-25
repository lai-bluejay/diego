#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
diego_benchmark/base_benchmark.py was created on 2019/04/08.
file in :
Author: Charles_Lai
Email: lai.bluejay@gmail.com
"""

from diego.study import create_study
def simple_diego(X, Y, metrics='acc'):
    from sklearn.model_selection import train_test_split

    # Split dataset for model testing
    X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.2, random_state=42)

    ts_autobin = create_study(X_train, y_train)
    ts_autobin.generate_trial(n_jobs=10, mode='cus', time_left_for_this_task=3600,
    per_run_time_limit=360,
    initial_configurations_via_metalearning=25,
    ensemble_size=10,
    ensemble_nbest=3,
    ensemble_memory_limit=1024,
    seed=1,
    ml_memory_limit=10240,
    include_estimators=["adaboost", "extra_trees", "k_nearest_neighbors",
                "libsvm_svc", "random_forest", "gaussian_nb","xgradient_boosting"])
    ts_autobin.optimize(X_valid, y_valid, n_jobs=-1, metrics=metrics)
    ts_autobin.show_models()
    return ts_autobin.pipeline