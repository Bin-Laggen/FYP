# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 15:37:34 2020

@author: Dominik Wiecek
"""

import numpy as np

from pyod.models.knn import KNN
from pyod.models.cof import COF
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from sklearn import metrics

class AlgorithmWrapper():
    def __init__(self, dataset, parameters=None, verbose=0):
        self._denoms = [10000, 5000, 2500, 1000, 500, 250, 200, 150, 100, 75, 50, 40, 30, 20, 10]
        self._dataset = dataset
        if parameters == None:
            self._parameters = self._calculateInputParams()
        else:
            self._parameters = parameters
        self._verbose = verbose
        self.name = ''
    
    def runTests(self):
        results = []
        for p in self._parameters:
            if self._verbose > 0:
                print('Testing param:', p)
            results.append(self._predictOutliers(p))
        return results, self._parseResults(results)
    
    def singleRun(self, param, dataset=None):
        return self._predictOutliers(param, dataset)
    
    def getParams(self):
        return self._parameters
    
    def _predictOutliers(self, param, dataset=None):
        pass
    
    def _parseResults(self, results):
        scores = {'s': (-1000000, -1), 'ch': (-1000000, -1), 'db': (1000000, -1)}
        for i, p in enumerate(self._parameters):
            sil = metrics.silhouette_score(self._dataset, results[i],
                                          metric='euclidean',
                                          sample_size=5000)
            ch = metrics.calinski_harabasz_score(self._dataset, results[i])
            db = metrics.davies_bouldin_score(self._dataset, results[i])
            
            if sil > scores['s'][0]:
                scores['s'] = (sil, p)
            if ch > scores['ch'][0]:
                scores['ch'] = (ch, p)
            if db < scores['db'][0]:
                scores['db'] = (db, p)
                
            if self._verbose > 0:
                print('Param:', p)
                print(results[i])
                print('Silhoutte score:', sil, 'the higher the better')
                print('Calinski-Harabasz index:', ch, 'the higher the better')
                print('Davies–Bouldin index:', db, 'the lower the better')
                print('====\n')
            if self._verbose > 1:
                print(self._dataset[results[i]==1])
        if self._verbose > 0:
            print(scores)
        return self._calculateBestParam(scores)
    
    def _calculateBestParam(self, scores):
        params = dict()
        for k, v in scores.items():
            if v[1] not in params:
                params[v[1]] = 1
            else:
                params[v[1]] += 1
        return max(params, key=lambda key: params[key])
    
    def _calculateInputParams(self):
        pass
        # p = size / 10000
        # params.append(p)
        # params.append(size / 5000)
        # params.append(size / 2500)
        # params.append(size / 1000)
        # params.append(size / 500)
        # params.append(size / 250)
        # params.append(size / 200)
        # params.append(size / 150)
        # params.append(size / 100)
        # params.append(size / 75)
        # params.append(size / 50)
        # params.append(size / 40)
        # params.append(size / 30)
        # params.append(size / 20)
        # params.append(size / 10)
            
class KNNWrapper(AlgorithmWrapper):
    def __init__(self, dataset, parameters=None, verbose=0):
        super().__init__(dataset, parameters, verbose)
        self.name = 'K-Nearest Neighbours'
    
    def _predictOutliers(self, param, dataset=None):
        if dataset == None:
            dataset = self._dataset
        alg = KNN(n_neighbors = param, n_jobs = -1)
        alg.fit(dataset)
        return alg.labels_
    
    def _calculateInputParams(self):
        size = self._dataset.index.size
        params = []
        for d in self._denoms:
            p = round(size / d)
            if p * size < 500000000:
                params.append(p)
            else:
                params.append(round(500000000 / size))
                break
        return params
    
# class COFWrapper(AlgorithmWrapper):
#     def __init__(self, dataset, parameters=None, verbose=0):
#         super().__init__(dataset, parameters, verbose)
#         self.name = 'Connectivity Based Outlier Factor'
        
#     def _predictOutliers(self, param, dataset=None):
#         if dataset == None:
#             dataset = self._dataset
#         alg = COF(n_neighbors = param)
#         alg.fit(dataset)
#         return alg.labels_
    
class IFWrapper(AlgorithmWrapper):
    def __init__(self, dataset, parameters=None, verbose=0):
        super().__init__(dataset, parameters, verbose)
        self.name = 'Isolation Forest'
        
    def _predictOutliers(self, param, dataset=None):
        if dataset == None:
            dataset = self._dataset
        if param <= 0.5:
            alg = IsolationForest(contamination = param)
        else:
            alg = IsolationForest()
        return np.array([1 if x == -1 else 0 for x in alg.fit_predict(dataset)])
    
    def _calculateInputParams(self):
        size = self._dataset.index.size
        params = []
        for d in self._denoms:
            p = 100 / d
            params.append(p)
        return params
    
class LOFWrapper(AlgorithmWrapper):
    def __init__(self, dataset, parameters=None, verbose=0):
        super().__init__(dataset, parameters, verbose)
        self.name = 'Local Outlier Factor'        
        
    def _predictOutliers(self, param, dataset=None):
        if dataset == None:
            dataset = self._dataset
        alg = LocalOutlierFactor(n_neighbors = param)
        return np.array([1 if x == -1 else 0 for x in alg.fit_predict(dataset)])
    
    def _calculateInputParams(self):
        size = self._dataset.index.size
        params = []
        for d in self._denoms:
            p = round(size / d)
            if p * size < 500000000:
                params.append(p)
            else:
                break
        return params
    

