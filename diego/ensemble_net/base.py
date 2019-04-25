#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
conan.base was created on 2017/10/17.
Author: Charles_Lai
Email: lai.bluejay@gmail.com

参考资料：
Zhou, Zhi-Hua. Ensemble methods: foundations and algorithms. CRC Press, 2012.

设计思路：
进行model ensemble时，会有不同的ensemble策略；会有初级学习器和次级学习器。
因此会有如下层级结构：
            D
    C1,  C2,  C3, ...
    y1,  y2,  y3, ...
--------融合策略P或[p1, p2]---------
           res
即， 把模型融合看成一个层级结构。由数据开始，每个数据/全量数据，使用每层不同的分类器进行学习和分类。最后一个每层的分类器会输出自己对应的结果；
采用不同的融合策略，比如平均、投票、学习，得到下一层的模型。
和Conan本身的layer不同，这里每个layer理论上都是可以并行的。
对于基本的ensemble，实际上只有一层+融合策略。如下：
ensemble = Ensemble([clf1, clf2, clf3])
eclf = EnsembleClassifier(ensemble=ensemble, combiner='mean')

对于stacking等，可能会有多个次级学习器，但太过复杂，只要两层一般够。再深不如直接神经网络。

# Creating Stacking
layer_1 = Ensemble([clf1, clf2, clf3])
layer_2 = Ensemble([sklearn.clone(clf1)])

stack = EnsembleStack(cv=3)

stack.add_layer(layer_1)
stack.add_layer(layer_2)

sclf = EnsembleStackClassifier(stack, combiner=Combiner('mean'))



"""

import numpy as np

from sklearn.metrics import accuracy_score, roc_auc_score
# 后续参考sklearn API的写法。
from sklearn.ensemble import BaseEnsemble, GradientBoostingClassifier
from xgboost import Booster
from diego.ensemble_net.combination import Combiner

def transform2votes(output, n_classes):

    n_samples = output.shape[0]

    votes = np.zeros((n_samples, n_classes), dtype=int)
    # uses the predicted label as index for the vote matrix
    # for i in range(n_samples):
    #    idx = int(output[i])
    #    votes[i, idx] = 1
    votes[np.arange(n_samples), output.astype(int)] = 1
    # assert np.equal(votes2.astype(int), votes.astype(int)).all()

    return votes.astype(int)


class Transformer(object):

    def __init__(self):
        pass

    def apply(self, X):
        pass


class FeatureSubsamplingTransformer(Transformer):

    def __init__(self, features=None):
        self.features = features

    def apply(self, X):
        # if is only one sample (1D)
        if X.ndim == 1:
            return X[self.features]
        # if X has more than one sample (2D)
        else:
            return X[:, self.features]


class BrewClassifier(object):
    """
    # TODO
    参考另一个项目的写法，暂时没用。
    """

    def __init__(self, classifier=None, transformer=None):
        self.transformer = transformer
        self.classifier = classifier
        self.classes_ = []

    def fit(self, X, y):
        X = self.transformer.apply(X)
        self.classifier.fit(X, y)
        self.classes_ = self.classifier.classes_

    def predict(self, X):
        X = self.transformer.apply(X)
        y = self.classifier.predict(X)
        return y

    def predict_proba(self, X):
        X = self.transformer.apply(X)
        y = self.classifier.predict_proba(X)
        return y


class Ensemble(object):
    """
    用于承载一堆分类器。基本是个list。
    当前要求每个classifier都有predict方法，在使用不同的合并条件的时候，可能需要predict_proba方法。
    """

    def __init__(self, classifiers=None, classes=None):

        if classifiers is None:
            self.classifiers = []
        else:
            self.classifiers = classifiers
        if classes is None:
            self.classes_ = np.array([0, 1])
        else:
            self.classes_ = classes

    def add(self, classifier):
        self.classifiers.append(classifier)

    def add_classifiers(self, classifiers):
        self.classifiers = self.classifiers + classifiers

    def add_ensemble(self, ensemble):
        self.classifiers = self.add_classifiers(ensemble.classifiers)

    def get_classes(self):
        return self.classes_

    def output(self, X, mode='vote'):
        """
        输出所有分类器的结果。

        (1) 'labels': 返回label，2-d array, （n_samples, n_classifiers)
        (2) 'probs': 返回预测的概率值，每个类都会返回，因此相当于一个3-d array，（n_samples, n_class, n_classifiers).
        #TODO 这边可能做一些优化，包括对分类器的适配。 在不做多分类的时候，可以考虑只输出同一类概率。
        输出的概率值，可以通过融合策略进行融合。
        (3) 'votes': 使用投票法融合的时候会用到。
        3d-array， (n_samples, n_classes, n_classifiers),
        Parameters
        ----------
        X: array-like, shape = [n_samples, n_features]
        mode: string, optional(default='labels')
                'labels' | 'probs' | 'votes'
        """

        if mode == 'label':
            out = np.zeros((X.shape[0], len(self.classifiers)))
            for i, clf in enumerate(self.classifiers):
                out[:, i] = clf.predict(X)

        else:
            # assumes that all classifiers were
            # trained with the same number of classes\
            try:
                classes__ = self.get_classes()
                n_classes = len(classes__)
            except:
                classes__ = np.array([0, 1])
                n_classes = 2
            out = np.zeros((X.shape[0], n_classes, len(self.classifiers)))

            for i, c in enumerate(self.classifiers):
                # TODO 当前都是基于sklearn API写的，对XGB的原生API还没做支持。
                if mode == 'probs':
                    probas = np.zeros((X.shape[0], n_classes))
                    if isinstance(c, Booster):
                        probas[:, list(self.classes_)] = c.predict(X)
                    else:
                        probas[:, list(self.classes_)] = c.predict_proba(X)
                    out[:, :, i] = probas

                elif mode == 'vote':
                    tmp = c.predict(X)  # (n_samples,)
                    # (n_samples, n_classes)
                    votes = transform2votes(tmp, n_classes)
                    out[:, :, i] = votes

        return out

    def output_simple(self, X):
        """
        每个分类器的预测的label，votes可用。
        :param X:
        :return:
        """
        out = np.zeros((X.shape[0], len(self.classifiers)))
        for i, clf in enumerate(self.classifiers):
            out[:, i] = clf.predict(X)
        return out

    def in_agreement(self, x):
        prev = None
        for clf in self.classifiers:
            [tmp] = clf.predict(x)
            if tmp != prev:
                return False
            prev = tmp

        return True

    def __len__(self):
        return len(self.classifiers)

    def fit(self, X, y):
        '''
        warning: this fit overrides previous generated base classifiers!
        '''
        for clf in self.classifiers:
            clf.fit(X, y)

        return self


class EnsembleClassifier(object):
    """
    参考sklearn API
    """

    def __init__(self, ensemble=None, selector=None, combiner=None):
        self.ensemble = ensemble

        # self.selector = selector

        if combiner is None:
            self.combiner = Combiner(rule='majority_vote')
        elif isinstance(combiner, str):
            self.combiner = Combiner(rule=combiner)
        elif isinstance(combiner, Combiner):
            self.combiner = combiner
        else:
            raise ValueError('Invalid parameter combiner')

    def fit(self, X, y):
        """
        暂不支持 sample_weight. 难难难。
        :param X:
        :param y:
        :return:
        """
        self.ensemble.fit(X, y)

    def predict(self, X):

        # TODO dynamic selection
        # Zhou, Zhi-Hua. Ensemble methods: foundations and algorithms. CRC Press, 2012.  第4.6 章
        out = self.ensemble.output(X)
        y = self.combiner.combine(out)
        return np.asarray(y)

    def predict_proba(self, X):
        out = self.ensemble.output(X, mode='probs')
        return np.mean(out, axis=2)

    def score(self, X, y, sample_weight=None):
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)
