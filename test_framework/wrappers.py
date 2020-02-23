# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 15:37:34 2020

@author: Dominik Wiecek
"""

import numpy as np
import pandas as pd
from pyod.models.knn import KNN
from pyod.models.cof import COF
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


class AlgorithmWrapper():
    def __init__(self, parameters, dataset):
        self._parameters = parameters
        self._dataset = dataset
    
    def runTests(self):
        results = []
        for p in self._parameters:
            print(p)
            results.append(self._predictOutliers(p))
        self._logResults(results)
        return results
    
    def _predictOutliers(self, param):
        pass
    
    def _logResults(self, results):
        for i, p in enumerate(self._parameters):
            print('Param:', p)
            print(results[i])
            print('====\n')
            
class KNNWrapper(AlgorithmWrapper):
    def _predictOutliers(self, param):
        alg = KNN(n_neighbors = param, n_jobs = -1)
        alg.fit(self._dataset)
        return alg.labels_
    
class COFWrapper(AlgorithmWrapper):
    def _predictOutliers(self, param):
        alg = COF(n_neighbors = param)
        alg.fit(self._dataset)
        print(alg.labels_)
        return alg.labels_
    
class IFWrapper(AlgorithmWrapper):
    def _predictOutliers(self, param):
        alg = IsolationForest()
        return np.array([1 if x == -1 else 0 for x in alg.fit_predict(self._dataset)])
        # return alg.fit_predict(self._dataset)
    
class LOFWrapper(AlgorithmWrapper):
    def _predictOutliers(self, param):
        alg = LocalOutlierFactor(n_neighbors = param)
        return np.array([1 if x == -1 else 0 for x in alg.fit_predict(self._dataset)])
    
def parse(dataset):
    # dataset = dataset.drop(['cut', 'color', 'clarity'], axis=1)
    dataset = dataset.drop(['price'], axis=1)
    # dataset.to_csv('diamonds_in.csv')
    
    cutMapping = {'Ideal': 5, 'Premium': 4, 'Good': 3, 'Very Good': 2, 'Fair': 1}
    dataset['cut'] = dataset['cut'].map(cutMapping)
    
    colorMapping = {'E': 0, 'I': 1, 'J': 2, 'H': 3, 'F': 4, 'G': 5, 'D': 6}
    dataset['color'] = dataset['color'].map(colorMapping)
    
    clarMapping = {'SI2': 1, 'SI1': 2, 'VS1': 3, 'VS2': 4, 'VVS2': 5, 'VVS1': 6, 'I1': 7, 'IF': 8}
    dataset['clarity'] = dataset['clarity'].map(clarMapping)
    
    return dataset
    
def printRes(dataset, outliers):
    for res in outliers:
        print(dataset[res==1])
    #   dataset[res==1].to_csv('outliers.csv')

def main():
    print('hello world\n')
    inputData = pd.read_csv('diamonds.csv', index_col=0)
    dataset = parse(inputData)
    print(dataset)
    
    print('\n===================')
    print('K-Nearest Neighbours')
    
    knn = KNNWrapper(range(2, 10), np.array(dataset))
    outliers = knn.runTests()
    
    printRes(inputData, outliers)
    
    print('\n===================\n')
    
    print('\n===================')
    print('Isolation Forest')
    
    iforest = IFWrapper([1], np.array(dataset))
    outliers = iforest.runTests()
    
    printRes(inputData, outliers)
    
    print('\n===================\n')
    
    # print('\n===================')
    # print('Connectivity Based Outlier Factor')
    
    # cof = COFWrapper(range(2, 10), np.array(dataset))
    # outliers = cof.runTests()
    
    # printRes(inputData, outliers)
    
    # print('\n===================\n')
    
    print('\n===================')
    print('Local Outlier Factor')
    
    lof = LOFWrapper(range(2, 10), np.array(dataset))
    outliers = lof.runTests()
    
    printRes(inputData, outliers)
    
    print('\n===================\n')
    
    print('\nEnd of execution\n')
    
main()
