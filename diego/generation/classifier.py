#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
/Users/charleslai/PycharmProjects/diego/diego/generation.classifier.py was created on 2019/03/20.
file in :relativeFile
Author: Charles_Lai
Email: lai.bluejay@gmail.com
"""
from diego.classifier import DiegoClassifier

class GenerationClassifier(DiegoClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    
if __name__ == "__main__":
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,
                                                        train_size=0.75, test_size=0.25)
