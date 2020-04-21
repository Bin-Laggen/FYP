# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 13:55:35 2020

@author: domis
"""

import numpy as np

from wrappers import KNNWrapper, IFWrapper, LOFWrapper#, COFWrapper

class OutlierTestRunner():
    def __init__(self, dataset, verbose=0):
        self._verbose = verbose
        self._dataset = dataset
        self._algorithms = {'knn': KNNWrapper(dataset=self._dataset, verbose=self._verbose),
                'if': IFWrapper(dataset=self._dataset, varbose=self._verbose),
                'lof': LOFWrapper(dataset=self._dataset, verobse=self._verbose)}
        
    def testAllAlgorithms(self, algs=None):
        results = []

        if algs != None and isinstance(algs, list):
            algorithms = []
            for a in algs:
                if a.lower() in self._algorithms:
                    algorithms.append(a.lower())
        else:
            algorithms = self._algorithms
            
        for a in algorithms:
            results.append(self.singleTest(a))
                           
        overlap = results[0]
        for r in range(1, len(results)):
            overlap = self.__class__.compareOutliers(overlap, results[r])
        
        return results, overlap
        
    def singleTest(self, algorithm_name):
        algorithm = self._algorithms[algorithm_name]
        if self._verbose > 0:
            print('\n===================')
            print(algorithm.name)
        outliers, best_param = algorithm.runTests()
        res = algorithm.singleRun(best_param)
        if self._verbose > 0:
            print('\n===================\n')
        return res
    
    def getAvailableAlgorithms(self):
        return list(self._algorithms.keys())
    
    @staticmethod
    def compareOutliers(index1, index2):
        overlap = np.logical_and(index1, index2)
        overlap = overlap.astype(int)
        return overlap
    