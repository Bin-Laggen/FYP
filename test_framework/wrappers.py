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
            print('Param:', p, 'i:', i)
            print(results[i])
            print('====\n')
            
class KNNWrapper(AlgorithmWrapper):
    def _predictOutliers(self, param):
        alg = KNN(n_neighbors = param, n_jobs = -1)
        alg.fit(self._dataset)
        return alg.labels_
    
class IFWrapper(AlgorithmWrapper):
    def _predictOutliers(self, param):
        alg = IsolationForest()
        return alg.fit_predict(self._dataset)
    
    
class COFWrapper(AlgorithmWrapper):
    def _predictOutliers(self, param):
        alg = COF(n_neighbors = param)
        alg.fit(self._dataset)
        return alg.labels_
    
class LOFWrapper(AlgorithmWrapper):
    def _predictOutliers(self, param):
        alg = LocalOutlierFactor(n_neighbors = param)
        return alg.fit_predict(self._dataset)

def main():
    print('hello world\n')
    
    dataset = pd.read_csv('diamonds.csv')
    dataset = dataset.drop(['cut', 'color', 'clarity'], axis=1)
    dataset.to_csv('diamonds_in.csv')
    print(dataset)
    
    knn = KNNWrapper([3, 6, 9], np.array(dataset))
    outliers = knn.runTests()
    print(dataset[outliers[0]==1])
    dataset[outliers[0]==1].to_csv('outliers.csv')
    
    # iforest = IFWrapper([3, 6, 9], np.array(dataset))
    # iforest.runTests()
    
    print('\nEnd of execution\n')
    
main()
