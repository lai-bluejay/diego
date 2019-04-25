#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
conan.combination was created on 2017/10/17.
Author: Charles_Lai
Email: lai.bluejay@gmail.com
"""
import numpy as np

"""
[1] Kittler, J.; Hatef, M.; Duin, R.P.W.; Matas, J., "On combining
    classifiers," Pattern Analysis and Machine Intelligence,
[2] Zhou, Zhi-Hua. Ensemble methods: foundations and algorithms. CRC Press, 2012.
"""

import numpy as np

"""
probs:  (n_class, classifiers_proba).
行：样本所属类别的概率
列：一个分类器的分类结果。
每列和为1，
"""

def max_rule(probs):
    # 最大预测值
    return probs.max(axis=1).argmax()


def min_rule(probs):

    return probs.min(axis=1).argmax()


def mean_rule(probs):
    return probs.mean(axis=1).argmax()


def median_rule(probs):

    # numpy array has no median method
    return np.median(probs, axis=1).argmax()


def majority_vote_rule(votes):
    """
    每列只能有一个1，即每个分类器只能投一票。
    :param votes:
    :return:
    """
    return votes.sum(axis=1).argmax()


class Combiner(object):

    def __init__(self, rule='majority_vote'):
        self.combination_rule = rule

        if rule == 'majority_vote':
            self.rule = majority_vote_rule

        elif rule == 'max':
            self.rule = max_rule

        elif rule == 'min':
            self.rule = min_rule

        elif rule == 'mean':
            self.rule = mean_rule

        elif rule == 'median':
            self.rule = median_rule

        else:
            raise Exception('invalid argument rule for Combiner class')

    def combine(self, results):

        n_samples = results.shape[0]

        out = np.zeros((n_samples,))

        for i in range(n_samples):
            out[i] = self.rule(results[i, :, :])

        return out