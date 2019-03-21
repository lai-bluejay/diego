#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
/Users/charleslai/PycharmProjects/diego/diego/preprocessor.base_binning.py was created on 2019/03/18.
file in :relativeFile
Author: Charles_Lai
Email: lai.bluejay@gmail.com
"""
import math
from collections import defaultdict
import re
import numpy as np
# TODO numpy to numba
import os
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))


def cal_woe(good_counts, bad_counts, all_good, all_bad):
    """
       计算weight of evidence
    """
    try:
        py1 = bad_counts * 1.0 / all_bad
        py2 = good_counts * 1.0 / all_good
        woe = math.log(py1 / py2)
    except:
        woe = 0
    return woe


def cal_woe_iv(good_counts, bad_counts, all_good, all_bad):
    """
       计算weight of evidence
    """
    try:
        py1 = bad_counts * 1.0 / all_bad
        py2 = good_counts * 1.0 / all_good
        woe = math.log(py1 / py2)
        iv = (py1 - py2) * woe
    except Exception as e:
        woe = 0
        iv = 0
    return woe, iv


class BaseBinning(object):
    """
       特征分桶模块，输入数据：特征向量，特征标注
                     输出数据：分桶结果，分桶方法
                     save_binning 保存为文件
    """

    def __init__(self, feature_vector=None, label_vector=None, param_dic={}, **kwargs):
        """
            基本参数预处理
        """
        self.feature_vector = feature_vector.astype('float', copy=False)
        self.feature_vector = np.nan_to_num(self.feature_vector, copy=False)
        self.feature_num = kwargs.get('feature_num', '0')
        self.not_outlier_set = kwargs.get('not_outlier_set', set([]))
        self.outlier_threshold = kwargs.get('outlier_threshold', 10)
        self.outlier_num_rate = kwargs.get('outlier_num_rate', 0.95)
        self.bins = kwargs.get('bins', 10)
        self.not_bins_set = kwargs.get('not_bins_set', set([]))
        self.label_vector = label_vector.astype('int', copy=False)
        self.modl_handler = kwargs.get('modl_handler', None)
        self.rm_outlier = True
        MAX = 1e100
        idx1 = np.where(self.feature_vector < MAX)[0]
        fx = self.feature_vector[idx1]
        idx2 = np.where(fx > -MAX)[0]
        del fx
        idx = idx1[idx2]
        self.feature_vector = self.feature_vector[idx]
        self.label_vector = self.label_vector[idx]

        self.good = 0
        self.bad = 1
        self.all_bad = self.label_vector.sum()
        self.all_good = self.label_vector.shape[0] - self.all_bad
        self.mapping = {}
        # print self.bins,self.feature_num
        self.left_out = 0
        self.right_out = 0
        self.bins_model_save = None

        self.verbose = 0

    def mapping_outlier(self, unique_feature):
        """
           异常值处理，目前算法采用的是拉依达法则，公式 miu+3*lmuda
        和之前相比，现在判断异常值之后，只保留正常值参与计算。===> 注意feature和label的index的对应。
        """
        # 参数初始化
        miu = self.feature_vector.mean()
        lmuda = self.feature_vector.std()
        self.left_out = miu - 3 * lmuda
        self.right_out = miu + 3 * lmuda
        x = self.feature_vector

        f_idx1 = np.where(x >= miu - 3 * lmuda)[0]
        x1 = x[f_idx1]
        f_idx2 = np.where(x1 <= miu + 3 * lmuda)[0]
        f_idx = f_idx1[f_idx2]

        new_unique_feature = np.unique(self.feature_vector[f_idx])

        # 剩余特征数得超过指定数量 & 对应的样本数得超过指定的比例，才进行删除
        if len(f_idx) > self.outlier_num_rate * self.feature_vector.shape[0]:
            self.feature_vector = self.feature_vector[f_idx]
            self.label_vector = self.label_vector[f_idx]
            unique_feature = np.unique(self.feature_vector)
            return new_unique_feature
        return unique_feature

    def discrete_method(self, feature_value_list):
        """
        分桶方法的模板
        :param feature_value_list: 特征值的列表，相当于特征向量的set。
        :return: init_map, {k:v}, key是桶编号，v是该桶内特征的值。为方便计算woe等
        """
        init_map = defaultdict(list)
        for i in self.bins:
            init_map[i].append(feature_value_list[i])
        return init_map

    def discrete_method_default(self, feature_value_list):
        """
           对特征值进行等宽分桶，原有方法
        """
        feature_value_list.sort()
        lens = len(feature_value_list)
        # print feature_value_list
        # print self.bins,'ew'
        step = int(lens / self.bins)
        if lens < self.bins:
            self.bins = lens
            step = 1

        init_map = {}
        for i in range(0, self.bins):
            init_map[i] = []
            for j in range(0, step):
                init_map[i].append(feature_value_list[i * step + j])
        # 最后一些多余分桶处理
        lasts = lens % self.bins
        if lasts > step:
            normal_tong = int(lasts / step)
            for i in range(0, normal_tong):
                init_map[self.bins] = []
                for j in range(0, step):
                    init_map[self.bins].append(feature_value_list[self.bins * step + j])
                self.bins += 1
        lasts = lens % self.bins
        for i in range(0, lasts):
            init_map[self.bins - 1].append(feature_value_list[-1 - i])
        return init_map

    def map_split_point_list_2_init_map(self, split_point_list, feature_value_list):
        """
        当只有分割点的时候，如何把特征值映射到init_map中，供后续使用
        :param split_point_list: 切点
        :param feature_value_list: 特征值列表
        :return:
        """
        init_map = defaultdict(list)
        """ sort asc"""
        split_point_list.sort()
        feature_value_list.sort()
        sp_len = len(split_point_list)
        if len(feature_value_list) == len(split_point_list):
            for i, fv in enumerate(feature_value_list):
                init_map[i].append(fv)
        else:

            cut_index = 0
            point = split_point_list[0]
            for i, fv in enumerate(feature_value_list):
                if fv <= point:
                    init_map[cut_index].append(fv)
                else:
                    if fv == point:
                        # 说明两者 相等或当前fv比分割点高，跳出循环
                        if point in self.mapping:
                            pass
                        else:
                            self.mapping[point] = dict()
                            self.mapping[point][self.good] = 0
                            self.mapping[point][self.bad] = 0
                        init_map[cut_index].append(point)
                    else:
                        init_map[cut_index + 1].append(fv)
                    if cut_index < sp_len - 1:
                        cut_index += 1
                        point = split_point_list[cut_index]
                    else:
                        init_map[cut_index + 1].extend(feature_value_list[i + 1:])
                        break

        return init_map

    def mapping_func(self, rm_outlier=True):
        """
           按照bins进行分桶，map所有数据，Monoton算法实现
           :param rm_outlier True 去除异常值；False 不去异常值
        """
        feature_value_list = np.unique(self.feature_vector)
        # 特征值的个数满足条件才进行异常值处理
        if rm_outlier and len(feature_value_list) > self.outlier_threshold:
            feature_value_list = self.mapping_outlier(feature_value_list)

        # flen = len(feature_value_list)
        # if flen < self.bins and self.discrete_method_name not in ["lgbm_dt", "xgb_dt", "modl"]:
        #     # 如果特征的取值数小于分桶数，使用默认分桶算法（默认分桶算法支持此类>处理)
        #     self.discrete_method_name = 'default'
        # try:
        #     init_map = self.discrete_method_map[self.discrete_method_name](feature_value_list)
        # except Exception as e:
        #     print self.bins, 'except'
        #     print self.feature_num
        #     print e
        #     init_map = self.discrete_method_map["default"](feature_value_list)
        """划重点，在这里全部抽样成基础方法。不同的方法在不同地方实例化"""
        try:
            init_map = self.discrete_method(feature_value_list)
        except:
            # 对分桶返回为空的特征，用默认方法重新分桶
            if self.verbose:
                print("[warning]: feature", self.feature_num, "binning failed, using default binning method redo" )
            init_map = self.discrete_method_default(feature_value_list)
        """
        #TODO
        假设这个位置只给出桶的切分点。
        """

        # init_map 按照分布映射表，统计到对应分桶中，注意给出分桶边界
        bins_box = {}
        for i in init_map.keys():
            count = {}
            edges = sorted(init_map[i])
            count['edges'] = edges[-1]
            bins_box[i] = count
        return bins_box

    # @exe_time
    def binning(self):
        """
            整个autobinning过程
        """
        # mapping and statistic
        rm_outlier = self.rm_outlier
        if self.feature_num in self.not_outlier_set:
            rm_outlier = False
        if self.feature_num in self.not_bins_set:
            rm_outlier = False
            self.bins = self.feature_vector.shape[0] + 1
        init_bins_box = self.mapping_func(rm_outlier)
        # print init_bins_box
        sorted_bins_box = sorted(init_bins_box.items(), key=lambda init_bins_box: init_bins_box[0], reverse=False)
        # repay rate check
        result_bins_box = {}
        repay_rate = []
        left_index = 0
        cut_point_list = [i for i in range(0, len(sorted_bins_box))]
        # cut_point_list = np.append(float('-inf'), cut_point_list)
        clen = len(cut_point_list)
        if clen == 0:
            """
            当不做任何切分的时候
            """
            clen = 1
        # repay rate check
        result_bins_box = {}
        repay_rate = []
        left_index = 0
        left_edges = float('-inf')
        iv = 0
        for i in range(0, clen):
            bins_cell = dict()
            bins_cell[self.good] = 0
            bins_cell[self.bad] = 0
            bins_cell['repay_rate'] = 0
            # 取边界
            for j in range(left_index, cut_point_list[i] + 1):
                right_edges = sorted_bins_box[j][1]['edges']
            left_index = cut_point_list[i] + 1

            bins_cell['left_edges'] = left_edges
            bins_cell['right_edges'] = right_edges
            # 分桶值填充
            f_idx1 = np.where(self.feature_vector > left_edges)[0]
            tmp_f = self.feature_vector[f_idx1]
            f_idx2 = np.where(tmp_f <= right_edges)[0]
            f_idx = f_idx1[f_idx2]
            # cur_feature = self.feature_vector[f_idx]
            cur_label = self.label_vector[f_idx]

            # 下一个桶的左边界
            left_edges = bins_cell['right_edges']
            bins_good = (cur_label == self.good).sum()
            bins_bad = (cur_label == self.bad).sum()
            bins_cell[self.good] = bins_good
            bins_cell[self.bad] = bins_bad

            bins_cell['repay_rate'] = bins_cell[self.good] * 1.0 / (bins_cell[self.good] + bins_cell[self.bad])
            # if bins_cell[self.good] == 0:
            #     print bins_cell
            #     print self.feature_num
            #     return {}
            woe, tmp_iv = cal_woe_iv(bins_cell[self.good] + 1, bins_cell[self.bad] + 1, self.all_good + clen,
                                     self.all_bad + clen)
            # print woe
            # print woe
            if woe == 0 and i >= 2:
                # np line
                X = [i - 1, i - 2]
                Y = [result_bins_box[i - 1]['woe'], result_bins_box[i - 2]['woe']]
                iv_y = [result_bins_box[i - 1]['tmp_iv'], result_bins_box[i - 2]['tmp_iv']]
                zx = np.polyfit(X, Y, 1)
                func = np.poly1d(zx)
                woe = func(i)
                zx2 = np.polyfit(X, iv_y, 1)
                iv_func = np.poly1d(zx2)
                tmp_iv = iv_func(i)
            bins_cell['woe'] = woe
            bins_cell['tmp_iv'] = tmp_iv
            bins_cell['bins_NO'] = i
            iv += tmp_iv
            result_bins_box[i] = bins_cell
        # 最后一个赋值无穷大
        result_bins_box[clen - 1]['right_edges'] = float('inf')
        result_bins_box['iv'] = iv
        return result_bins_box

class EqualWidthBinning(BaseBinning):
    def __init__(self, feature_vector=None, label_vector=None, param_dic={}, **kwargs):
        super(EqualWidthBinning, self).__init__(feature_vector=feature_vector, label_vector=label_vector,
                                                param_dic=param_dic,
                                                **kwargs)

    def discrete_method(self, feature_value_list):
        """
           对特征值进行等宽分桶，原有方法
        """
        feature_value_list.sort()
        lens = len(feature_value_list)
        # print feature_value_list
        # print self.bins,'ew'
        step = int(lens / self.bins)
        if lens < self.bins:
            self.bins = lens
            step = 1

        init_map = {}
        for i in range(0, self.bins):
            init_map[i] = []
            for j in range(0, step):
                init_map[i].append(feature_value_list[i * step + j])
        # 最后一些多余分桶处理
        lasts = lens % self.bins
        if lasts > step:
            normal_tong = int(lasts / step)
            for i in range(0, normal_tong):
                init_map[self.bins] = []
                for j in range(0, step):
                    init_map[self.bins].append(feature_value_list[self.bins * step + j])
                self.bins += 1
        lasts = lens % self.bins
        for i in range(0, lasts):
            init_map[self.bins - 1].append(feature_value_list[-1 - i])
        return init_map


class EqualFreqBinning(BaseBinning):
    def __init__(self, feature_vector=None, label_vector=None, param_dic={}, **kwargs):
        super(EqualFreqBinning, self).__init__(feature_vector=feature_vector, label_vector=label_vector,
                                               param_dic=param_dic,
                                               **kwargs)

    def discrete_method(self, feature_value_list):
        """
           对特征值进行近似等频分桶，遍历各值，达到剩余值/剩余份数则分桶，遇到某个值占比足够大，则将前面累积的值分一桶，该值分一桶。略有提升。
        """
        # print 'ef is running'
        init_map = defaultdict(list)
        # remain_bins = self.bins
        feature_value_nums = len(feature_value_list)
        sample_nums = len(self.label_vector)
        value_counts = {}
        for sample_num in range(sample_nums):
            cur_value = self.feature_vector[sample_num]
            if cur_value not in value_counts:
                value_counts[cur_value] = 0
            value_counts[cur_value] += 1
        sorted_value_counts = sorted(value_counts.items(), key=lambda d: d[0])
        bin_length = sample_nums * 1.0 / self.bins
        cur_bin_num = 0
        cur_samples = 0
        for value, count in sorted_value_counts:
            if count >= bin_length:
                cur_bin_num += 1
                init_map[cur_bin_num] = [value]
                cur_bin_num += 1
                cur_samples = 0
            else:
                cur_samples += count
                init_map[cur_bin_num].append(value)
                if cur_samples >= bin_length:
                    cur_bin_num += 1
                    cur_samples = 0
        return init_map


class XGBBinning(BaseBinning):
    def __init__(self, feature_vector=None, label_vector=None, param_dic={}, **kwargs):
        super(XGBBinning, self).__init__(feature_vector=feature_vector, label_vector=label_vector, param_dic=param_dic,
                                         **kwargs)
        self.rm_outlier = False

    def discrete_method(self, feature_value_list):
        """
        利用树对单特征进行分桶.
        自适应桶的数量: #TODO
        每个桶的最小样本数量：样本数/叶子节点数/2
        :param feature_value_list:  特征列

        :return: init_map, 记录每一个分割点的位置
        """
        import xgboost as xgb
        # TODO 自适应桶的数量
        
        self.bins = 15
        label = self.label_vector
        feature = self.feature_vector.reshape(self.feature_vector.shape[0], 1)
        # 用于判断是连续值还是离散值
        value_num = len(feature_value_list)
        if value_num < self.bins:
            self.bins = value_num
        if value_num < 5:
            split_point_list = feature_value_list
            self.bins = len(split_point_list)
        else:
            num_leaves = self.bins
            min_child_weight = 10
            bt = "gbtree"
            xgb_params = {
                'eta': 1.0,
                'booster': 'dart',
                'max_depth': 3,
                'gamma': 1,
                'objective': 'binary:logistic',
                'eval_metric': ['auc', 'logloss'],
                'nthread': 16,
                'tree_method': 'hist',
                'grow_policy': 'lossguide',
                'max_leaves': num_leaves,
                'min_child_weight': min_child_weight,
                'max_bins': 1024,
                'silent':True
            }

            dtr = xgb.DMatrix(feature, label=label)
            clf = xgb.train(xgb_params, dtr, num_boost_round=1)
            first_tree = clf.get_dump()[0]
            split_str_list = re.findall(r'\[(.*)\]', first_tree)
            split_point_list = [float(tmp.split('<')[1]) for tmp in split_str_list]

        split_point_list.sort()
        # feature_value_list.sort()
        init_map = self.map_split_point_list_2_init_map(split_point_list, feature_value_list)
        ###
        return init_map


class LGBBinning(BaseBinning):
    def __init__(self, feature_vector=None, label_vector=None, param_dic={}, **kwargs):
        super(LGBBinning, self).__init__(feature_vector=feature_vector, label_vector=label_vector, param_dic=param_dic,
                                         **kwargs)
        self.rm_outlier = False

    def discrete_method(self, feature_value_list):
        """
        利用lightgbm对单特征进行分桶.
        自适应桶的数量: #TODO
        每个桶的最小样本数量：样本数/叶子节点数/2
        :param feature_value_list:  占位符

        :return: init_map, 记录每一个分割点的位置
        """
        # TODO 自适应桶的数量
        self.bins = 15
        label = self.label_vector
        feature = self.feature_vector.reshape(self.feature_vector.shape[0], 1)
        # 用于判断是连续值还是离散值
        value_num = len(feature_value_list)
        if value_num < self.bins:
            self.bins = value_num
        if value_num < 5:
            split_point_list = feature_value_list
            self.bins = len(split_point_list)
        else:
            num_leaves = self.bins
            ins_len = len(feature)
            min_data_in_leaf = int(ins_len / num_leaves / 2)
            # TODO  min_sum_hessian_in_leaf
            bt = "gbdt"
            lgb_params = {
                'learning_rate': 1.0,
                'boosting': bt,
                'objective': 'binary',
                'metric': ['auc', 'binary_logloss'],
                'nthread': 12,
                'num_trees': 1,
                'min_data_in_leaf': min_data_in_leaf,
                'tree_method': 'hist',
                'num_leaves': num_leaves,
                'max_bin': 255,
            }
            dtr = lgb.Dataset(feature, label=label)
            clf = lgb.train(lgb_params, dtr, num_boost_round=1)

            tree_info = clf.dump_model(1)["tree_info"]
            tree_structure = tree_info[1]['tree_structure']
            split_point_list = list()

            def _read_tree_split_value(tree_structure, split_point_list):
                if isinstance(tree_structure, dict):
                    for k, v in tree_structure.items():
                        if k == 'threshold':
                            split_point_list.append(v)
                        if isinstance(v, dict):
                            _read_tree_split_value(v, split_point_list)

            _read_tree_split_value(tree_structure, split_point_list)
            """split point list, sort asc"""

            self.bins = len(split_point_list) + 1
        split_point_list.sort()
        feature_value_list.sort()
        init_map = self.map_split_point_list_2_init_map(split_point_list, feature_value_list)
        return init_map


class ModlBinning(BaseBinning):
    def __init__(self, feature_vector=None, label_vector=None, param_dic={}, **kwargs):
        super(ModlBinning, self).__init__(feature_vector=feature_vector, label_vector=label_vector, param_dic=param_dic,
                                          **kwargs)
        self.rm_outlier = False

    def discrete_method(self, feature_value_list):
        """
        对特征值进行modl分桶
        :return: init_map, 记录每一个分割点的位置
        """

        """feature_vector & label_vector sort needed, if use modl"""
        idx = np.lexsort((self.label_vector, self.feature_vector))
        sorted_feature_vector = [self.feature_vector[i] for i in idx]
        sorted_label_vector = [self.label_vector[i] for i in idx]
        """TODO: ctypes遇到性能问题，目前以系统调用方式接入,ugly~"""
        # split_index = self.modl_handler.binning(sorted_feature_vector, sorted_label_vector)
        # split_point_list = [(sorted_feature_vector[i]+sorted_feature_vector[i+1])/2 for i in split_index]
        split_point_list = self.modl_binning(sorted_feature_vector, sorted_label_vector)
        split_point_list.sort()
        init_map = self.map_split_point_list_2_init_map(split_point_list, feature_value_list)

        # print "[debug] modl feature_num:", self.feature_num
        # print "[debug] modl split_point_list: ", split_point_list
        # for i in range(0, len(init_map)):
        #    print "[debug] modl init_map: ", i, init_map[i]

        return init_map

    def modl_binning(self, sorted_feature_vector, sorted_label_vector):
        """
        系统调用方式进行modl分桶
        :return: split_point_list, 记录每一个分割特征值
        """
        modl_path = CURRENT_DIR + '/lib/modl'
        tmp_file = CURRENT_DIR + '/../../data/modl_binning_tmp_lbl_fea'
        cmd = "%s %s" % (modl_path, tmp_file)
        fh = open(tmp_file, 'w')
        for i in range(len(sorted_feature_vector)):
            fh.write("%s\t%s\n" % (sorted_label_vector[i], sorted_feature_vector[i]))
        fh.close()

        split_point_list = [float(x.strip()) for x in os.popen(cmd).readlines()]
        return split_point_list

if __name__ == '__main__':
    fv = np.array([i for i in range(10)])
    lv = np.array([i for i in range(10)])
    x = XGBBinning(fv, lv)
    if isinstance(x, XGBBinning):
        print(type(x))