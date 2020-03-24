# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 17:33:13 2020

@author: domis
"""

import numpy as np
import pandas as pd

from datetime import datetime

from wrappers import KNNWrapper, IFWrapper, LOFWrapper#, COFWrapper

class LogParser():
    
    def __init__(self, filename, verbose=0):
        self._readfile(filename)
        self._verbose = verbose
        
    def _readfile(self, filename):
        self._dataset = pd.read_csv('reduced_log.csv')
        pd.options.display.max_columns = self._dataset.shape[1]
        
    def _parse(self):
        if self._verbose > 0:
            print('Pre-parsed dataset\n')
            print(self._dataset.describe(include='all'))
            
        
        self._dataset = self._dataset.dropna(axis=1, how='all')
        self._dataset = self._dataset.select_dtypes(exclude=['object'])
        
        fill = dict()
        for col in self._dataset:
            val = self._dataset[col].quantile(0.5)
            if self._dataset[col].mean() < self._dataset[col].quantile(0.8) and self._dataset[col].mean() > self._dataset[col].quantile(0.2):
                val = self._dataset[col].mean()
            fill[col] = val
        
        self._dataset = self._dataset.fillna(value=fill)
        
        if self._verbose > 1:
            print('Mean values for each column')
            for k, v in fill.items():
                print(k, ':', v)
        
        if self._verbose > 0:
            print('Pre-parsed dataset\n')
            print(self._dataset.describe(include='all'))
        
        return self._dataset
    
    def _testAlgorithms(self):
        algorithms = [KNNWrapper(range(20, 30), self._dataset, self._verbose),
                IFWrapper([1], self._dataset, self._verbose),
                LOFWrapper(range(20, 30), self._dataset, self._verbose)]
        results = []
        
        for a in algorithms:
            results.append(self._singleTest(a))
                           
        overlap = results[0]
        for r in range(1, len(results)):
            overlap = compareIndexes(overlap, results[r])
        
        return overlap
        
    def _singleTest(self, algorithm):
        if self._verbose > 0:
            print('\n===================')
            print(algorithm.name)
        outliers, best_param = algorithm.runTests()
        res = algorithm.singleRun(best_param).index
        if self._verbose > 0:
            print('\n===================\n')
        return res
    
    def _compareOutliers(self, index1, index2):
        overlap = set()
        for i in index1:
            if i in index2:
                overlap.add(i)
        for i in index2:
            if i in index1:
                overlap.add(i)
        if self._verbose > 1:
            print('Overlapping outliers:', len(overlap))
        return overlap
        

def parse(dataset):
    # print(dataset.describe())
    # print(dataset.iloc[5])
    
    # print(dataset.dtypes)
    
    dataset = dataset.dropna(axis=1, how='all')
    dataset = dataset.select_dtypes(exclude=['object'])
    
    fill = dict()
    for col in dataset:
        val = dataset[col].quantile(0.5)
        if dataset[col].mean() < dataset[col].quantile(0.8) and dataset[col].mean() > dataset[col].quantile(0.2):
            val = dataset[col].mean()
        fill[col] = val
    
    dataset = dataset.fillna(value=fill)
    
    # print(dataset.describe())
    # print(dataset.iloc[5])
    # print(dataset.mean())
    
    
    print(dataset.describe(include='all'))
    # print(dataset.iloc[5])
    # print(dataset.mean())
    
    return dataset
    
def printRes(dataset, outliers):
    for res in outliers:
        print(dataset[res==1])
    #   dataset[res==1].to_csv('outliers.csv')
        
def test(dataset, verbose=0):
    # algorithms = [KNNWrapper(range(20, 30), dataset, verbose),
    #               IFWrapper([1], dataset, verbose),
    #               LOFWrapper(range(20, 30), dataset, verbose)]
    # results = []
    
    # for a in algorithms:
    #     if verbose > 0:
    #         print('\n===================')
    #         print(a.name)
    #     outliers, best_param = a.runTests()
    #     results.append(a.singleRun(best_param).index)
    #     if verbose > 0:
    #         print('\n===================\n')
            
    # overlap = results[0]
    # for r in range(1, len(results)):
    #     overlap = compareIndexes(overlap, results[r])
    
    # return overlap
        
    
    # print('\n===================')
    # print('K-Nearest Neighbours')
    knn = KNNWrapper(range(20, 30), dataset, 2)
    # outliers, best_param = knn.runTests()
    # print(best_param)
    best_param = 29
    knn_res = knn.singleRun(best_param).index
    # printRes(inputData, outliers)
    # print('\n===================\n')
    
    
    # print('\n===================')
    # print('Isolation Forest')
    iforest = IFWrapper([1], dataset, 2)
    # outliers, best_param = iforest.runTests()
    # print(best_param)
    best_param = 1
    i_res = iforest.singleRun(best_param).index
    # printRes(inputData, outliers)
    # print('\n===================\n')
    
    
    # print('\n===================')
    # print('Connectivity Based Outlier Factor')
    # cof = COFWrapper(range(2, 50), dataset)
    # outliers, best_param = cof.runTests()
    # print(best_param)
    # printRes(inputData, outliers)
    # print('\n===================\n')
    
    
    # print('\n===================')
    # print('Local Outlier Factor')
    lof = LOFWrapper(range(20, 30), dataset)
    # outliers, best_param = lof.runTests()
    # print(best_param)
    best_param = 20
    lof_res = lof.singleRun(best_param).index
    # printRes(inputData, ou1tliers)
    # print('\n===================\n')
    
    # compareIndexes(knn_res, i_res)
    # compareIndexes(knn_res, lof_res)
    # compareIndexes(i_res, lof_res)
    return [len(knn_res), len(i_res), len(lof_res)], compareIndexes(knn_res, compareIndexes(i_res, lof_res))
    
    
def compareIndexes(i1, i2):
    sameIs = set()
    for i in i1:
        if i in i2:
            sameIs.add(i)
    for i in i2:
        if i in i1:
            sameIs.add(i)
    # print(np.array(sameIs))
    print(len(sameIs))
    return sameIs
    

def main():
    
    # print('hello world\n')
    
    start = datetime.now()
    inputData = pd.read_csv('reduced_log.csv')
    pd.options.display.max_columns = inputData.shape[1]
    # inputData = pd.read_csv('parsed.csv')
    print(datetime.now() - start)
    
    dataset = parse(inputData)
    
    low_values = dict()
    high_values = dict()
    for col in dataset:
        low_values[col] = dataset[col].quantile(0.2)
        high_values[col] = dataset[col].quantile(0.8)
    # print(low_values)
    # print(high_values)
    
    outlier_counts, overlap = test(dataset)
    
    print(outlier_counts)
    
    outliers_dataset = inputData.loc[list(overlap)]
    # print(outliers_dataset)
    # outliers_dataset.to_csv('outliers.csv')
    # outliers_dataset = pd.read_csv('outliers.csv')
    
    for i, r in outliers_dataset.iterrows():
        print(r)
        for l in low_values:
            if r[l] < low_values[l]:
                print('lower than 20% of', l)
        for h in high_values:
            if r[h] > high_values[h]:
                print('higher than 80% of', h)
    
    print('\nEnd of execution\n')
    print(datetime.now() - start)
    
main()
