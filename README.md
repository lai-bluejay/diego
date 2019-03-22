# Diego

Diego: Data in, IntElliGence Out. Sprite Come from [Fast.ai](https://github.com/fastai/fastai) and [MicroSoft nni](https://github.com/Microsoft/nni).

## Installation

需要先安装swig，部分依赖C/C++的接口编译。推荐使用 conda 安装

```shell
conda install swig
pip install diego
```

安装好之后，开始6行代码解决一个分类问题吧。

## 模块结构

### study, trials

参考[MicroSoft nni](https://github.com/Microsoft/nni)，定义`Study`和`Trial`。
每次的任务认为是一个`Study`，每个 Study 由多个`Trial`构成。
建议先创建 Study，再从 Study 中生成 Trial:

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

### 如果在OS X或者Linux多进程被 hang/crash/freeze

由于在并行化的时候，n_jobs>1可能会卡住。在[scikit-learn中，同样可能出现类似的问题](https://scikit-learn.org/stable/faq.html#why-do-i-sometime-get-a-crash-freeze-with-n-jobs-1-under-osx-or-linux)
在Python3.4+中，一种解决方案是，直接配置`multiprocessing`使用`forkserver` 或 `spawn`来启动进程池管理 (而不是默认的`fork`)。例如直接在代码中全局启用`forkserver`模式。

```python
import multiprocessing
# other imports, custom code, load data, define model...
if __name__ == '__main__':
    multiprocessing.set_start_method('forkserver')

    # call scikit-learn utils with n_jobs > 1 here
```

更多设置可以参考[multiprocessing document](https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods)

### core

#### storage

对于每次的Study，数据的存储和参数，以及模型是额外存在`Storage`对象的，保证了Study只控制trials，同时每个Trial完成后更新在storage中的结果，同时更新最好的结果。

#### 结果的更新

在创建`Study`的时候，需要指定优化的方向 `maximize` 或者 `minimize`。同时在创建`Trials`的时候，指定优化的指标。默认是 `maximize accuracy`。

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

### features TODO

- [ ] 回归。
- [ ] 不同类型的Trial。TPE， BayesOpt， RandomSearch
- [ ] 自定义的Trial。Trials by custom Classifier (like sklearn, xgboost)
- [ ] 模型保存。model persistence
- [ ] 模型输出。model output
- [ ] basic Classifier
- [ ] fix mac os hanged in optimize pipeline
- [ ] add preprocessor
- [ ] add FeatureTools for automated feature engineering
