# Conan Reborn

Diego: Data IntElliGence Out. Sprite Come from [Fast.ai](https://github.com/fastai/fastai) and [MicroSoft nni](https://github.com/Microsoft/nni).

# 模块结构

## study, trials

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
s.generate_trials(mode='fast')
s.optimize(X_test, y_test)
all_trials = s.get_all_trials()
for t in all_trials:
    print(t.__dict__)
    print(t.clf.score(X_test, y_test))

```




## core

### storage
对于每次的Study，数据的存储和参数，以及模型是额外存在`Storage`对象的，保证了Study只控制trials，同时每个Trial完成后更新在storage中的结果，同时更新最好的结果。

#### 结果的更新
在创建`Study`的时候，需要指定优化的方向 `maximize` 或者 `minimize`。同时在创建`Trials`的时候，指定优化的指标。默认是 `maximize accuracy`。

### TODO 文档更新。

### features TODO

- [ ] 不同类型的Trial。TPE， BayesOpt， RandomSearch
- [ ] 自定义的Trial。Trials by custom Classifier (like sklearn, xgboost)
- [ ] 模型保存。model persistence
- [ ] 模型输出。model output
- [ ] basic Classifier
- [ ] fix mac os hanged in optimize pipeline
- [ ] add preprocessor
- [ ] add FeatureTools for automated feature engineering


## train

用于组装数据、模型、损失函数、优化方法等。
定义：

```python
class Learner():
    "Trainer for `model` using `data` to minimize `loss_func` with optimizer `opt_func`."

class Recorder(LearnerCallback):
    "A `LearnerCallback` that records epoch, loss, opt and metric data during training."

class Estimator():
    "A estimator to evaulate learner.  "

```

# auto ml 补完计划

[overview](https://hackernoon.com/a-brief-overview-of-automatic-machine-learning-solutions-automl-2826c7807a2a)

## bayes opt

1. [fmfn/bayes](https://github.com/fmfn/BayesianOptimization)
2. [auto-sklearn](https://github.com/automl/auto-sklearn)

## grid search

1. H2O.ai

## tree parzen

1. hyperopt
2. mlbox

## metaheuristics grid search

1. pybrain

## generation

1.tpot

# dl

1. ms nni
2.

# installation

## install swig

推荐使用 conda 安装

```shell
conda install swig
```

## 其他 dep

```
pip install pyrfr
pip install smac
pip install autosklearn
```

# issues

# updates
