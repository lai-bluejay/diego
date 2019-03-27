#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
/Users/charleslai/PycharmProjects/diego/diego/preprocessor.auto_binning.py was created on 2019/03/18.
file in :relativeFile
Author: Charles_Lai
Email: lai.bluejay@gmail.com
"""

import copy

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin


import sklearn.metrics
import autosklearn.classification
import autosklearn.metrics
import autosklearn.pipeline.components.feature_preprocessing
from autosklearn.pipeline.components.base \
    import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import DENSE, SIGNED_DATA, \
    UNSIGNED_DATA


import os
import math
import numpy as np
from collections import defaultdict
# import lightgbm as lgb

from diego.preprocessor.base_binning import EqualFreqBinning, EqualWidthBinning, XGBBinning, ModlBinning

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))



def normalize_feature_by_col(feature_col_origin, auto_binning, feature_num, binning_value_type="woe"):
    """
    按列进行woe映射
    param feature_col:特征列
    param auto_binning:分桶模型
    param feature_num:特征名
    param binning_value_type:返回特征值类型
    return:
    """
    bins_detail = auto_binning['detail']
    bins_box = bins_detail[feature_num]
    feature_col = copy.deepcopy(feature_col_origin)
    for i, cell in bins_box.items():
        if i == "iv":
            continue
        if i == "feature_num":
            continue
        left = float(cell['left_edges'])
        right = float(cell['right_edges']) + 1e-7
        f_idx1 = np.where(feature_col_origin > left)[0]
        tmp_f = feature_col_origin[f_idx1]
        f_idx2 = np.where(tmp_f <= right)[0]
        f_idx = f_idx1[f_idx2]
        if binning_value_type in ["woe", "left_edges", "right_edges"]:
            feature_col[f_idx] = np.float64(cell[binning_value_type])
        elif binning_value_type in ["bins_NO"]:
            feature_col[f_idx] = cell[binning_value_type]
        else:
            # print("[warning]:", feature_num, binning_value_type, "not found, using woe")
            feature_col[f_idx] = np.float64(cell["woe"])
    return feature_col

class AutobinningTransform(BaseEstimator, TransformerMixin):
    """[define the preprocessor]
    
    Arguments:
        BaseEstimator {[type]} -- [description]
        TransformerMixin {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """

    def __init__(self, param_dic={},**kwargs):
        '''
        binning_method: ‘default’,使用等宽分桶
                        ’equalfrequency’,使用等频分桶
                        ’lgbm_dt’，决策树(lgbm)分桶
                        ’xgb_dt’，决策树(xgb)分桶
                        ’modl’，modl分桶
        binning_value_type: ‘woe’,woe值
                            ‘bins_NO’,桶号
                            ‘left_edges’,左边界值,可能是-inf
                            ‘right_edges’,左边界值,可能是inf
        not_outlier_file:  配置文件名，默认为None，配置文件里包括不做特征值处理的特征编号
        outlier_threshold: int/float，默认为10
                           当类型为float时，outlier_threshold*=样本总数
                           当特征值的数量小于outlier_threshold时，不进行异常值处理
        outlier_num_rate: float，默认0.95
                          当进行异常值处理后，保留的特征值所对应的样本数量 <  样本总量*outlier_num_rate时，不进行异常之处理

        not_bins_file：配置文件名，默认为None，配置文件里的特征在进行分桶时，单个特征为一桶
        bins_threshold：int/float，默认为10
                        当类型为float时，bins_threshold*=样本总数
                        当特征值的数量小于bins_threshold时，单个特征为一桶

        keep_origin：set，集合中的特征不进行任何处理，保留原值
        bins_conf_file：配置文件名，默认为’binning_dict.conf’，指定具体特征分多少桶
        '''
        kwargs.update(param_dic)
        self.cut_tt = kwargs.get('cut_tt',0.9)
        self.autobinning_dump_file = "_np_autobinnig_normal"
        self.binning_method = kwargs.get('binning_method', 'default')
        self.binning_value_type = kwargs.get('binning_value_type', "woe")
        self.dump_file = kwargs.get('dump_file', False)
        self.not_outlier_file = kwargs.get('not_outlier_file',None)
        self.not_outlier_set = set([])
        self.outlier_threshold = kwargs.get('outlier_threshold',10)
        self.outlier_num_rate = kwargs.get('outlier_num_rate',0.95)
        self.not_bins_file = kwargs.get('not_bins_file',None)
        self.not_bins_set = set([])
        self.bins_threshold = kwargs.get('bins_threshold',10)
        self.keep_origin = kwargs.get('keep_origin',set([]))
        self.bins_conf_file = kwargs.get('bins_conf_file','binning_dict.conf')
        self.type_conf_file = kwargs.get('type_conf_file','feature_select.conf.first')

        self.bins_conf = {}
        # self.type_conf = {}

        self.binning_method = param_dic.get('binning_method', self.binning_method)
        self.binning_value_type = param_dic.get('binning_value_type',self.binning_value_type)
        self.not_outlier_file = param_dic.get('not_outlier_file',self.not_outlier_file)
        self.outlier_threshold = param_dic.get('outlier_threshold',self.outlier_threshold)
        self.outlier_num_rate = param_dic.get('outlier_num_rate',self.outlier_num_rate)
        self.not_bins_file = param_dic.get('not_bins_file',self.not_bins_file)
        self.bins_threshold = param_dic.get('bins_threshold',self.bins_threshold)
        self.keep_origin = param_dic.get('keep_origin',self.keep_origin)
        # self.bins_conf_file = param_dic.get('bins_conf_file',self.bins_conf_file)
        # self.type_conf_file = param_dic.get('type_conf_file',self.type_conf_file)

        if isinstance(self.outlier_threshold,float):
            self.outlier_threshold *= len(self.input_node.feature)
        if isinstance(self.bins_threshold,float):
            self.bins_threshold *= len(self.input_node.feature)

        # fit_model
        self.bins_model_save = None
        self.featurename = list()
        self.delete_list = []

    
    def fit(self,X,y,featurename=None):
        """
        @note:单进程版本autobinning
        @param featuresmatrix: 特征矩阵
        @param featuresvector: 特征名向量
        @param labelvector: 标签向量
        @return : normal_features 归一化后的feature
                  delete_list 需要删除的feature列表
        """
        featuresmatrix = X
        labelvector = y
        feature_cols = featuresmatrix.shape[1]
        if not featurename:
            featurename = [i for i in range(feature_cols+1)]
        self.feature_name = featurename
        delete_list = []
        bins_model_save = {}
        bins_model_dict = dict()
        sample_len = featuresmatrix.shape[0]
        # create modl binning model
        if isinstance(self.binning_method, ModlBinning):
            modl_handler = Modl(sample_len)
        else:
            modl_handler = None
        # 做一次基础数据的适配
        for i in range(0, feature_cols):
            feature_vector = featuresmatrix[:, i]
            feature_vector_bin = feature_vector[:]
            labelvector_bin = labelvector[:]
            feature_num = featurename[i]

            if feature_num in self.keep_origin:
                #if feature_vector.min() < 0 or feature_vector.max() > 1:
                #   print 'origin feature value should be in range(0,1)'
                #   delete_list.append(i)
                continue
            # 判断数据离散还是连续，离散不分桶
            # feature_type = self.type_conf.get(feature_num, 1)
            # if feature_type == 0 or  len(np.unique(feature_vector_bin))< self.bins_threshold:
            #     self.not_bins_set.add(featurename[i])
            '''
            if int(feature_num) in [10030030,10050005,10030009,10030011,10030005]:#or str(feature_num).f
                bins_tt = 100
            else:
                bins_tt = 10
            '''
            bins_tt = self.bins_conf.get(feature_num,10)
            if self.binning_method == 'default' or self.binning_method == 'ef':
                Auto_Binning = EqualFreqBinning
            elif self.binning_method == 'ew':
                Auto_Binning = EqualWidthBinning
            elif self.binning_method == 'xgb':
                Auto_Binning = XGBBinning
            elif self.binning_method == 'modl':
                Auto_Binning = ModlBinning
            auto_binning = Auto_Binning(feature_vector=feature_vector_bin, \
                                        label_vector=labelvector_bin, \
                                        bins=bins_tt, feature_num=feature_num, \
                                        not_outlier_set=self.not_outlier_set, \
                                        outlier_threshold=self.outlier_threshold, \
                                        outlier_num_rate=self.outlier_num_rate, \
                                        not_bins_set=self.not_bins_set, \
                                        bins_threshold=self.bins_threshold, \
                                        modl_handler=modl_handler)
            #auto_binning.init_all()
            result_bins_box = auto_binning.binning()
            #print result_bins_box, feature_num
            """
            result_bin_box
+            {0: {0: 171390,
+                 1: 62796,
+                 'left_edges': -inf,
+                 'repay_rate': 0.73185416720043039,
+                 'right_edges': inf,
+                 'tmp_iv': 6.5617517881995298e-08,
+                 'woe': -0.0002602236216015671},
+             'iv': 6.5617517881995298e-08}
            """
            #break
            if len(result_bins_box) <= 2:
                delete_list.append(i)
                #print i, feature_num, result_bins_box
                continue
            # 保存模型
            result_bins_box['feature_num'] = feature_num
            bins_model_dict[feature_num] = result_bins_box
        bins_model_save['delete'] = delete_list
        #np.save(training_name+NP_DELETE_FEATURES, delete_list)
        bins_model_save['detail'] = bins_model_dict
        self.delete_list = delete_list
        self.bins_model_save = bins_model_save
        return self

    def transform(self, X):
        featuresmatrix = X
        bins_model_save = self.bins_model_save
        feature_name = self.feature_name
        featuresmatrix = np.delete(featuresmatrix, bins_model_save['delete'], axis=1)
        featuresmatrix = featuresmatrix.astype('float64', copy=False)
        feature_name = np.delete(feature_name, bins_model_save['delete'], axis=0)
        # feature_name = bins_model_save['detail'].keys()
        # print feature_name
        for i in range(0, featuresmatrix.shape[1]):
            feature_np = featuresmatrix[:, i]
            feature_num = feature_name[i]
            if feature_num in self.keep_origin:
                continue

            binning_value_type = self.binning_value_type
            feature_np = normalize_feature_by_col(feature_np, bins_model_save, feature_num, binning_value_type=binning_value_type)
            featuresmatrix[:, i] = feature_np
        return featuresmatrix


class AutoBinning(AutoSklearnPreprocessingAlgorithm):
    def __init__(self, binning_method, random_state=None):
        self.binning_method = binning_method
        self.random_state = random_state
        self.preprocessor = None

    def fit(self, X, y=None):
    
        self.preprocessor = AutobinningTransform(
                binning_method=self.binning_method
            )
        self.preprocessor.fit(X, y)
        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'AutoBinning',
                'name': 'Auto Binning for linear model',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': False,
                'handles_multilabel': False,
                'is_deterministic': True,
                'input': (DENSE, UNSIGNED_DATA, SIGNED_DATA),
                'output': (DENSE, UNSIGNED_DATA, SIGNED_DATA)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        # TODO add hyperparameter to gbdt binning
        cs = ConfigurationSpace()
        binning_method = CategoricalHyperparameter(
            name="binning_method", choices=['ef', 'ew', 'xgb', 'modl'], default_value='ef'
        )
        # shrinkage = UniformFloatHyperparameter(
        #     name="shrinkage", lower=0.0, upper=1.0, default_value=0.5
        # )
        # n_components = UniformIntegerHyperparameter(
        #     name="n_components", lower=1, upper=29, default_value=10
        # )
        # tol = UniformFloatHyperparameter(
        #     name="tol", lower=0.0001, upper=1, default_value=0.0001
        # )
        cs.add_hyperparameters([binning_method])
        return cs


if __name__ == '__main__':
    # Add LDA component to auto-sklearn.
    autosklearn.pipeline.components.feature_preprocessing.add_preprocessor(AutoBinning)

    # Create dataset.
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    from sklearn.linear_model import RidgeClassifier
    clf = RidgeClassifier().fit(X_train, y_train)
    print(clf.score(X_test, y_test))

    print('binning')
    pp = AutobinningTransform(binning_method='xgb')
    X_train = pp.fit_transform(X_train, y_train)
    X_test = pp.transform(X_test)
    clf = RidgeClassifier().fit(X_train, y_train)
    print(clf.score(X_test, y_test))
    print('='*100)
    # Configuration space.
    cs = AutoBinning.get_hyperparameter_search_space()
    print(cs)

    # Fit the model using LDA as preprocessor.
    clf = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=30,
        include_preprocessors=['AutoBinning'],
    )
    clf.fit(X_train, y_train)

    # Print prediction score and statistics.
    y_pred = clf.predict(X_test)
    print("accracy: ", sklearn.metrics.accuracy_score(y_pred, y_test))
    print(clf.sprint_statistics())
    print('='*50, 'models', "="*50)
    print(clf.show_models())

    clf.fit_ensemble(y_train, ensemble_size=50)
    print('='*50, 'ensemble', "="*50)

    print(clf.show_models())
    predictions = clf.predict(X_test)
    print(clf.sprint_statistics())
    print("Accuracy score", sklearn.metrics.accuracy_score(y_test, predictions))