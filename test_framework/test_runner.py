# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 13:55:35 2020

@author: domis
"""

import numpy as np

# from datetime import datetime

from wrappers import KNNWrapper, IFWrapper, LOFWrapper#, COFWrapper

class OutlierTestRunner():
    algorithms = ['knn', 'iforest', 'lof']
    
    def __init__(self, dataset, verbose=0):
        self._verbose = verbose
        self._dataset = dataset
        self._algorithms = {'knn': KNNWrapper(dataset=self._dataset, verbose=self._verbose),
                'iforest': IFWrapper(dataset=self._dataset, verbose=self._verbose),
                'lof': LOFWrapper(dataset=self._dataset, verbose=self._verbose)}
        
    def testAllAlgorithms(self, algs=None):
        results = {}

        if algs != None and isinstance(algs, list):
            algorithms = []
            for a in algs:
                if a.lower() in self._algorithms:
                    algorithms.append(a.lower())
        else:
            algorithms = self._algorithms.keys()
            
        for a in algorithms:
            res = self.testOneAlgorithm(a)
            results[a] = {'best_labels': res[0], 'best_param': res[1], 'scores': res[2]}
                           
        if len(results) > 1:
            overlap = None
            for _, r in results.items():
                overlap = self.__class__.compareOutliers(overlap, r['best_labels'])
            
            results['overlap'] = overlap
        return results
        
    def testOneAlgorithm(self, algorithm_name):
        algorithm = self._algorithms[algorithm_name]
        if self._verbose > 0:
            print('\n===================')
            print(algorithm.name)
        scores, best_param = algorithm.runTests()
        best_res = algorithm.singleRun(best_param)
        if self._verbose > 0:
            print('\n===================\n')
        return best_res, best_param, scores
    
    
    @staticmethod
    def compareOutliers(index1, index2):
        if index1 is None:
            return index2
        if index2 is None:
            return index1
        overlap = np.logical_and(index1, index2)
        overlap = overlap.astype(int)
        return overlap
    