# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 22:39:49 2020

@author: domis
"""

import pandas as pd

from log_parser import LogParser
from wrappers import KNNWrapper, IFWrapper, LOFWrapper

def main():
    inputData = pd.read_csv('D:\Dokumenty\College\Year 4\Sem2\FYP\shuttle.csv', index_col=0)
    print(inputData[inputData['y_1']==1])
    
    lp = LogParser('D:\Dokumenty\College\Year 4\Sem2\FYP\shuttle.csv', 0, 0)
    lp._dataset = lp._dataset.drop('y_1', axis=1)
    print(lp._dataset)
    lp._parse()
    overlap = lp._testAlgorithms()
    
    print()
    print()
    
    outliers_dataset = inputData.loc[list(overlap)]
    print('Overlap')
    print(outliers_dataset)
    print(outliers_dataset[outliers_dataset['y_1']==1])
    print((outliers_dataset[outliers_dataset['y_1']==1].size / outliers_dataset.size) * 100)
    print((outliers_dataset.size / inputData[inputData['y_1']==1].size) * 100)
    print((outliers_dataset[outliers_dataset['y_1']==1].size / inputData[inputData['y_1']==1].size) * 100)
    print()
    print()

    knn_res = lp._singleTest(KNNWrapper(range(20, 30), lp._dataset, lp._verbose))
    if_res = lp._singleTest(IFWrapper([1], lp._dataset, lp._verbose))
    lof_res = lp._singleTest(LOFWrapper(range(20, 30), lp._dataset, lp._verbose))
    
    knn_outliers = inputData.loc[list(knn_res)]
    if_outliers = inputData.loc[list(if_res)]
    lof_outliers = inputData.loc[list(lof_res)]
    
    print('KNN')
    print(knn_outliers)
    print(knn_outliers[knn_outliers['y_1']==1])
    print((knn_outliers[knn_outliers['y_1']==1].size / knn_outliers.size) * 100)
    print((knn_outliers.size / inputData[inputData['y_1']==1].size) * 100)
    print((knn_outliers[knn_outliers['y_1']==1].size / inputData[inputData['y_1']==1].size) * 100)
    print()
    print()
    
    print('IF')
    print(if_outliers)
    print(if_outliers[if_outliers['y_1']==1])
    print((if_outliers[if_outliers['y_1']==1].size / if_outliers.size) * 100)
    print((if_outliers.size / inputData[inputData['y_1']==1].size) * 100)
    print((if_outliers[if_outliers['y_1']==1].size / inputData[inputData['y_1']==1].size) * 100)
    print()
    print()
    
    print('LOF')
    print(lof_outliers)
    print(lof_outliers[lof_outliers['y_1']==1])
    print((lof_outliers[lof_outliers['y_1']==1].size / lof_outliers.size) * 100)
    print((lof_outliers.size / inputData[inputData['y_1']==1].size) * 100)
    print((lof_outliers[lof_outliers['y_1']==1].size / inputData[inputData['y_1']==1].size) * 100)
    print()
    print()
    
    return

if __name__ == '__main__':
    main()