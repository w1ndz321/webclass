"""
feature env
interactive with the actor critic for the state and state after action
"""
#'r'是只读，'w'是只能写,'rb'是只读二进制文件,'wb'是只能写二进制文件
import os
from collections import namedtuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from record import RecordList
from utils.logger import error, info
from utils.tools import test_task_new, downstream_task_new, downstream_task_by_method_std


base_path = r'C:\Users\Jupiter\Desktop\IJCAI-GAINS-code'
print('basepath:',base_path)
#这里就出现了之前看到的ap_omentum_ovary
TASK_DICT = {'airfoil': 'reg', 'amazon_employee': 'cls', 'ap_omentum_ovary': 'cls',
             'bike_share': 'reg', 'german_credit': 'cls', 'higgs': 'cls',
             'housing_boston': 'reg', 'ionosphere': 'cls', 'lymphography': 'cls',
             'messidor_features': 'cls', 'openml_620': 'reg', 'pima_indian': 'cls',
             'spam_base': 'cls', 'spectf': 'cls', 'svmguide3': 'cls',
             'uci_credit_card': 'cls', 'wine_red': 'cls', 'wine_white': 'cls',
             'openml_586': 'reg', 'openml_589': 'reg', 'openml_607': 'reg',
             'openml_616': 'reg', 'openml_618': 'reg', 'openml_637': 'reg',
             'smtp': 'det', 'thyroid': 'det', 'yeast': 'det', 'wbc': 'det', 'mammography': 'det', 'arrhythmia': 'cls',
             'nomao': 'cls', 'megawatt1': 'cls', 'activity':'mcls', 'mice_protein':'mcls', 'coil-20':'mcls', 'isolet':'mcls', 'minist':'mcls',
             'minist_fashion':'mcls',
             'new_df':'mcls',
             'df500':'mcls',
             'sample_df3600':'mcls'
             }

#reg是regression, cls是classification, det是detection, mcls是multi-classification

MEASUREMENT = {
    'cls': ['precision', 'recall', 'f1_score', 'roc_auc'],
    'reg': ['mae', 'mse', 'rae', 'rmse'],
    'det': ['map', 'f1_score', 'ras', 'recall'],
    'mcls' : ['precision', 'recall', 'mif1', 'maf1']
}

model_performance = {
    'mcls':namedtuple('ModelPerformance', MEASUREMENT['mcls']),
    'cls': namedtuple('ModelPerformance', MEASUREMENT['cls']),
    'reg': namedtuple('ModelPerformance', MEASUREMENT['reg']),
    'det': namedtuple('ModelPerformance', MEASUREMENT['det'])
}
#创建了名为‘modelperformance’的类，包含了不同任务的性能指标，包括precision, recall, f1_score, roc_auc, mae, mse, rae, rmse, map, f1_score, ras, recall, precision, recall, mif1, maf1

#传入当前任务名，任务类型，数据地址，他会去读取哪个hdf文件，用pddataframe把训练集和测试集分割好，最后创建一个任务名文件夹
class Evaluator(object):
    def __init__(self, task, task_type=None, dataset=None):
        self.original_report = None
        self.records = RecordList()
        self.task_name = task#当前任务名称
        if task_type is None:
            self.task_type = TASK_DICT[self.task_name]
        else:
            self.task_type = task_type

        if dataset is None:
            data_path = os.path.join(base_path, 'data', 'history',self.task_name + '.hdf')
            print('data_path:',data_path)
            # if self.task_name =='openml_618':
            #     original =pd.read_hdf(data_path, key='wdj')
            # else:
            original = pd.read_hdf(data_path,key='wdj')#这里的key的设定要看一下,看hdf文件里面 有哪些key
        else:
            original = dataset

        col = np.arange(original.shape[1])#生成从0到original的列数-1的数组，original的列应该是各个特征外加最后一列的label
        self.col_names = original.columns
        original.columns = col
        y = original.iloc[:, -1]#取最后一列作为标签
        x = original.iloc[:, :-1]#取除最后一列以外的列作为特征
        #对于任务'ap_omentum_ovary'（卵巢？腹膜？），对他的标签做转换
        if task == 'ap_omentum_ovary':
            y[y == 'Ovary'] = 1
            y[y == 'Omentum'] = 0
            y = y.astype(float)
            original = pd.concat([pd.DataFrame(x), pd.DataFrame(y)], axis=1)
        self.original = original.fillna(value=0)#缺失值替换成0

        y = self.original.iloc[:, -1]
        x = self.original.iloc[:, :-1]
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                            random_state=0, shuffle=True)
        #拼接训练集的特征选取情况和对应标签
        self.train = pd.concat([pd.DataFrame(X_train), pd.DataFrame(y_train)], axis=1)
        ##拼接测试集的特征选取情况和对应标签
        self.test = pd.concat([pd.DataFrame(X_test), pd.DataFrame(y_test)], axis=1)
        #concat合并两个dataframe，axis=1表示按列合并，生成一个新的dataframe
        # dataframe是一个表格，每一列是一个特征，每一行是一个样本，最后一列是标签

        #info是从logger.py里引入的，
        info('initialize the train and test dataset')
        #如果没有history下的任务名文件夹，就创建一个
        self._check_path()

    def __len__(self):
        return len(self.records)

    def generate_data(self, operation, flag):
        pass

    #传入data，自动返回down函数的结果是评估值
    def get_performance(self, data=None):
        if data is None:
            data = self.original
        return downstream_task_new(data, self.task_type)

    def report_ds(self):
        pass

    def _store_history(self, choice, performance):
        self.records.append(choice, performance)

#传入choices是字符串，主要是去保存pt文件
    def _flush_history(self, choices, performances, is_permuted, num, padding):
        if is_permuted:
            flag_1 = 'augmented'
        else:
            flag_1 = 'original'
        if padding:
            flag_2 = 'padded'
        else:
            flag_2 = 'not_padded'
        torch.save(choices, f'{base_path}/data/history/{self.task_name}/choice.{flag_1}.{flag_2}.{num}.pt')
        info(f'save the choice to {base_path}/data/history/{self.task_name}/choice.pt')
        torch.save(performances, f'{base_path}/data/history/{self.task_name}/performance.{flag_1}.{flag_2}.{num}.pt')
        info(f'save the performance to {base_path}/data/history/{self.task_name}/performance.pt')

    def _check_path(self):
        if not os.path.exists(f'{base_path}/data/history/{self.task_name}'):
            os.mkdir(f'{base_path}/data/history/{self.task_name}')

    def save(self, num=25, padding=True, padding_value=-1):
        if num > 0:
            is_permuted = True
        else:
            is_permuted = False
        info('save the records...')
        choices, performances = \
            self.records.generate(num=num, padding=padding, padding_value=padding_value)
        self._flush_history(choices, performances, is_permuted, num, padding)

    def get_record(self, num=0, eos=-1):
        results = []
        labels = []
        for record in self.records.r_list:
            result, label = record.get_permutated(num, True, eos)
            results.append(result)
            labels.append(label)
        return torch.cat(results, 0), torch.cat(labels, 0)

    def get_triple_record(self, num=0, eos=-1, mode='ht'):
        h_results = []
        labels = []
        t_results = []
        h_seed = []
        labels_seed = []
        for record in self.records.r_list:
            if mode.__contains__('h'):
                h, label = record.get_permutated(num, True, eos)
            else:
                h, label = record.repeat(num, True, eos)
            if mode.__contains__('t'):
                t, _ = record.get_permutated(num, True, eos)
            else:
                t, _ = record.repeat(num, True, eos)
            h_results.append(h)
            t_results.append(t)
            labels.append(label)
            h_seed.append(h_results[0])
            labels_seed.append(labels[0])
        return torch.cat(h_results, 0), torch.cat(labels, 0), torch.cat(t_results), \
               torch.cat(h_seed), torch.cat(labels_seed),

    def report_performance(self, choice, store=True, rp=True, flag=''):
        opt_ds = self.generate_data(choice, flag)
        a, b, c, d = test_task_new(opt_ds, task=self.task_type)
        report = model_performance[self.task_type](a, b, c, d)
        if flag == 'test':
            store = False
        if self.original_report is None:
            a, b, c, d = test_task_new(self.test, task=self.task_type)
            self.original_report = (a, b, c, d)
        else:
            a, b, c, d = self.original_report
        original_report = model_performance[self.task_type](a, b, c, d)

        if self.task_type == 'reg':
            final_result = report.rae
            if rp:
                info('1-MAE on original is: {:.4f}, 1-MAE on generated is: {:.4f}'.
                     format(original_report.mae, report.mae))
                info('1-MSE on original is: {:.4f}, 1-MSE on generated is: {:.4f}'.
                     format(original_report.mse, report.mse))
                info('1-RAE on original is: {:.4f}, 1-RAE on generated is: {:.4f}'.
                     format(original_report.rae, report.rae))
                info('1-RMSE on original is: {:.4f}, 1-RMSE on generated is: {:.4f}'.
                     format(original_report.rmse, report.rmse))
        elif self.task_type == 'cls':
            final_result = report.f1_score
            if rp:
                info('Pre on original is: {:.4f}, Pre on generated is: {:.4f}'.
                     format(original_report.precision, report.precision))
                info('Rec on original is: {:.4f}, Rec on generated is: {:.4f}'.
                     format(original_report.recall, report.recall))
                info('F-1 on original is: {:.4f}, F-1 on generated is: {:.4f}'.
                     format(original_report.f1_score, report.f1_score))
                info('ROC/AUC on original is: {:.4f}, ROC/AUC on generated is: {:.4f}'.
                     format(original_report.roc_auc, report.roc_auc))
        elif self.task_type == 'det':
            final_result = report.ras
            if rp:
                info(
                    'Average Precision Score on original is: {:.4f}, Average Precision Score on generated is: {:.4f}'
                    .format(original_report.map, report.map))
                info(
                    'F1 Score on original is: {:.4f}, F1 Score on generated is: {:.4f}'
                    .format(original_report.f1_score, report.f1_score))
                info(
                    'ROC AUC Score on original is: {:.4f}, ROC AUC Score on generated is: {:.4f}'
                    .format(original_report.ras, report.ras))
                info(
                    'Recall on original is: {:.4f}, Recall Score on generated is: {:.4f}'
                    .format(original_report.recall, report.recall))
        elif self.task_type == 'mcls':
            final_result = report.mif1
            if rp:
                info('Pre on original is: {:.4f}, Pre on generated is: {:.4f}'.
                     format(original_report.precision, report.precision))
                info('Rec on original is: {:.4f}, Rec on generated is: {:.4f}'.
                     format(original_report.recall, report.recall))
                info('Micro-F1 on original is: {:.4f}, Micro-F1 on generated is: {:.4f}'.
                     format(original_report.mif1, report.mif1))
                info('Macro-F1 on original is: {:.4f}, Macro-F1 on generated is: {:.4f}'.
                     format(original_report.maf1, report.maf1))
        else:
            error('wrong task name!!!!!')
            assert False
        if store:
            self._store_history(choice, final_result)
        return final_result


class FeatureEvaluator(Evaluator):
    def __init__(self, task, task_type=None, dataset=None):
        super(FeatureEvaluator, self).__init__(task, task_type, dataset)
        self.ds_size = self.original.shape[1] - 1
        #ds_size是特征的数量
    def generate_data(self, choice, flag=''):
        if choice.shape[0] != self.ds_size:
            error('wrong shape of choice')
            assert False
        if flag == 'test':
            ds = self.test
        elif flag == 'train':
            ds = self.train
        else:
            ds = self.original
        X = ds.iloc[:, :-1]
        indice = torch.arange(0, self.ds_size)[choice == 1]
        #包含所有被选择的特征列的索引
        X = X.iloc[:, indice].astype(np.float64)
        y = ds.iloc[:, -1].astype(np.float64)
        #这里看一下要不要改成32位
        Dg = pd.concat([pd.DataFrame(X), pd.DataFrame(y)], axis=1)
        #把被选择的特征和标签合并成一个dataframe
        return Dg

    def _full_mask(self):
        return torch.FloatTensor([1] * self.ds_size)
    #这个张量表示选择了所有特征
    def report_ds(self):
        per = self.get_performance()
        info(f'current dataset : {self.task_name}')
        info(f'the size of shape is : {self.original.shape[1]}')
        info(f'original performance is : {per}')
        self._store_history(self._full_mask(), per)
        #    def _store_history(self, choice, performance):
        #        self.records.append(choice, performance)

import time
if __name__ == '__main__':
    task_name = 'spectf'
    fe = FeatureEvaluator(task_name)
    start_time = time.time()
    fe.report_ds()
    end_time = time.time()
    info(f'training on overall eval cost : {end_time - start_time}s')
    for method in ['RF', 'XGB', 'SVM', 'KNN', 'Ridge', 'DT', 'LASSO']:
        info(method)
        start_time = time.time()
        p, std = downstream_task_by_method_std(fe.original, fe.task_type, method)
        end_time = time.time()
        info(f'training on {method} eval cost : {end_time - start_time}s')
    # fe.report_performance(torch.FloatTensor([0, 1, 1, 0, 0]))
    # fe.save()
    # print(1)
