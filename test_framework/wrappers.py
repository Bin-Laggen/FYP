# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 15:37:34 2020

@author: Dominik Wiecek
"""

import numpy as np
import psutil

from datetime import datetime

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
        results = dict()
        for i, p in enumerate(self._parameters):
            if self._verbose > 0:
                print('Testing param:', p)
                
            scores = {'parameter': p}
            start = datetime.now()
            predicted_labels = self._predictOutliers(p)
            processing_time = datetime.now() - start
            scores = self._calculateScores(predicted_labels, scores)
            scores['time'] = processing_time
            results[i] = scores
            
            if self._verbose > 0:
                print(predicted_labels)
                
            if self._verbose > 1:
                print(self._dataset[predicted_labels==1])
                
            if self._verbose > 0:
                print('Silhoutte score:', scores['sil'], 'the higher the better')
                print('Calinski-Harabasz index:', scores['cal_har'], 'the higher the better')
                print('Davies–Bouldin index:', scores['dav_bou'], 'the lower the better')
                print('====\n')
                print()
                
            # results.append(self._predictOutliers(p))
        return results, self._calculateBestParam(results)
    
    def singleRun(self, param, dataset=None):
        return self._predictOutliers(param, dataset)
    
    def singleRunWithScore(self, param, dataset=None):
        scores = dict()
        res = self._predictOutliers(param, dataset)
        combined = self._calculateScores(res, scores)['combined']
        return res, combined
    
    def getParams(self):
        return self._parameters
    
    def _predictOutliers(self, param, dataset=None):
        pass
    
    def _calculateScores(self, result, scores):
        combined = 1
        changed = False
        try:
            sil = metrics.silhouette_score(self._dataset, result,
                                              metric='euclidean',
                                              sample_size=5000)
            combined *= sil
            changed = True
        except ValueError:
            sil = -1;
        scores['sil'] = sil
        
        try:
            ch = metrics.calinski_harabasz_score(self._dataset, result)
            combined *= ch
            changed = True
        except ValueError:
            ch = -1
        scores['cal_har'] = ch
            
        try:
            db = metrics.davies_bouldin_score(self._dataset, result)
            combined /= db
            changed = True
        except ValueError:
            db = 999999999
        scores['dav_bou'] = db
        if changed == True:
            scores['combined'] = combined
        else:
            scores['combined'] = None
        return scores
        
    def _calculateBestParam(self, results):
        best = (-1, -1)
        for _, score in results.items():
            if score['combined'] != None and score['combined'] > best[1]:
                best = (score['parameter'], score['combined'])
        return best[0]
    
    # def _parseResults(self, results):
    #     best_scores = {'sil': (-1000000, -1), 'cal_har': (-1000000, -1), 'dav_bou': (1000000, -1)}
    #     # for i, p in enumerate(self._parameters):
    #     #     sil = metrics.silhouette_score(self._dataset, results[i],
    #     #                                   metric='euclidean',
    #     #                                   sample_size=5000)
    #     #     ch = metrics.calinski_harabasz_score(self._dataset, results[i])
    #     #     db = metrics.davies_bouldin_score(self._dataset, results[i])
    #     for p, score in results.items():
    #         sil = score['sil']
    #         ch = score['cal_har']
    #         db = score['dav_bou']
            
    #         if sil > best_scores['sil'][0]:
    #             best_scores['sil'] = (sil, p)
    #         if ch > best_scores['cal_har'][0]:
    #             best_scores['cal_har'] = (ch, p)
    #         if db < best_scores['dav_bou'][0]:
    #             best_scores['dav_bou'] = (db, p)
                
    #     if self._verbose > -1:
    #         print(best_scores)
    #         print()
    #     return self._calculateBestParam(best_scores)
    
    # def _calculateBestParam(self, scores):
    #     params = dict()
    #     for k, v in scores.items():
    #         if v[1] not in params:
    #             params[v[1]] = 1
    #         else:
    #             params[v[1]] += 1
    #     return max(params, key=lambda key: params[key])
    
    def _calculateInputParams(self):
        pass
            
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
        sys_mem = psutil.virtual_memory().total
        GB = 1024 * 1024 * 1024
        available_mem = (sys_mem - (2 * GB)) / 3 # / 3
        for d in self._denoms:
            p = round(size / d)
            if p * size * 8 < available_mem:
                if p >= 1:
                    params.append(p)
                else:
                    continue
            else:
                params.append(round((available_mem - (GB / 2)) / (size * 8)))
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
        params = []
        for d in self._denoms:
            p = 100 / d
            params.append(round(p, 3))
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
        sys_mem = psutil.virtual_memory().total
        GB = 1024 * 1024 * 1024
        available_mem = (sys_mem - (2 * GB)) / 3 # / 3
        for d in self._denoms:
            p = round(size / d)
            if p * size * 8 < available_mem:
                if p >= 1:
                    params.append(p)
                else:
                    continue
            else:
                params.append(round((available_mem - (GB / 2)) / (size * 8)))
                break
        return params
    

