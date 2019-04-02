
# Diego

Diego: Data in,  IntElliGence Out.

[简体中文](README_zh_CN.md)

A fast framework that supports the rapid construction of automated learning tasks. Simply create an automated learning study (`Study`) and generate correlated trials (`Trial`). Then run the code and get a machine learning model. Implemented using Scikit-learn API [glossary](https://scikit-learn.org/stable/glossary.html), using Bayesian optimization and genetic algorithms for automated machine learning.

Inspired by [Fast.ai](https://github.com/fastai/fastai) and [MicroSoft nni](https://github.com/Microsoft/nni).

[![Build Status](https://travis-ci.org/lai-bluejay/diego.svg?branch=master)](https://travis-ci.org/lai-bluejay/diego)
![PyPI](https://img.shields.io/pypi/v/diego.svg?style=flat)
![GitHub](https://img.shields.io/github/license/lai-bluejay/diego.svg)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/lai-bluejay/diego.svg)

- [x] the classifier trained by a Study.
- [x] AutoML classifier with support for scikit-learn api. Support for exporting models and use them directly.
- [x] Hyperparametric optimization using Bayesian optimization and genetic algorithms
- [x] Supports bucketing/binning algorithm and LUS sampling method for preprocessing
- [ ] Supports scikit-learn api classifier custom classifier for parameter search and super parameter optimization


## Installation

You need to install swig first, and some rely on C/C++ interface compilation. Recommended to use conda installation

```shell
conda install --yes pip gcc swig libgcc=5.2.0
pip install diego
```

After installation, start with 6 lines of code to solve a machine learning classification problem.

## Usage

Each task is considered to be a `Study`, and each Study consists of multiple `Trial`.
It is recommended to create a Study first and then generate a Trial from the Study:

```python
from diego.study import create_study
import sklearn.datasets
digits = sklearn.datasets.load_digits()
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(digits.data, digits.target,train_size=0.75, test_size=0.25)

s = create_study(X_train, y_train)
# can use default trials in Study

# or generate one
# s.generate_trials(mode='fast')
s.optimize(X_test, y_test)
# all_trials = s.get_all_trials()
# for t in all_trials:
#     print(t.__dict__)
#     print(t.clf.score(X_test, y_test))

```

## RoadMap
ideas for releases in the future
- [ ] 回归。
- [ ] add documents.
- [ ] 不同类型的Trial。TPE， BayesOpt， RandomSearch
- [ ] 自定义的Trial。Trials by custom Classifier (like sklearn, xgboost)
- [ ] 模型保存。model persistence
- [ ] 模型输出。model output
- [ ] basic Classifier
- [ ] fix mac os hanged in optimize pipeline
- [ ] add preprocessor
- [ ] add FeatureTools for automated feature engineering


## 

## Project Structure

### study, trials
Study: 

Trial:

### 如果在OS X或者Linux多进程被 hang/crash/freeze

Since n_jobs>1 may get stuck during parallelization. Similar problems may occur in [scikit-learn] (https://scikit-learn.org/stable/faq.html#why-do-i-sometime-get-a-crash-freeze-with-n -jobs-1-under-osx-or-linux)

In Python 3.4+, one solution is to directly configure `multiprocessing` to use `forkserver` or `spawn` to start process pool management (instead of the default `fork`). For example, the `forkserver` mode is enabled globally directly in the code.

```python
import multiprocessing
# other imports, custom code, load data, define model...
if __name__ == '__main__':
    multiprocessing.set_start_method('forkserver')

    # call scikit-learn utils with n_jobs > 1 here
```

more info :[multiprocessing document](https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods)

### core

#### storage

For each study, the data storage and parameters, and the model is additionally stored in the `Storage` object, which ensures that Study only controls trials, and each Trial updates the results in the storage after updating, and updates the best results.

#### update result

When creating `Study`, you need to specify the direction of optimization `maximize` or `minimize`. Also specify the metrics for optimization when creating `Trials`. The default is `maximize accuracy`.

## auto ml 补完计划

[overview](https://hackernoon.com/a-brief-overview-of-automatic-machine-learning-solutions-automl-2826c7807a2a)

### bayes opt

1. [fmfn/bayes](https://github.com/fmfn/BayesianOptimization)
2. [auto-sklearn](https://github.com/automl/auto-sklearn)

### grid search

1. H2O.ai

### tree parzen

1. hyperopt
2. mlbox

### metaheuristics grid search

1. pybrain

### generation

1.tpot

### dl

1. ms nni

## issues

## updates

### TODO 文档更新。

