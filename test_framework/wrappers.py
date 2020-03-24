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

from sklearn import metrics

class AlgorithmWrapper():
    def __init__(self, parameters, dataset, verbose=0):
        self._parameters = parameters
        self._dataset = dataset
        self._verbose = verbose
        self.name = ''
    
    def runTests(self):
        results = []
        for p in self._parameters:
            print('Testing param:', p)
            results.append(self._predictOutliers(p))
        return results, self._parseResults(results)
    
    def singleRun(self, param, dataset=None):
        if dataset == None:
            dataset = self._dataset
        return dataset[self._predictOutliers(param)==1]
    
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
            
class KNNWrapper(AlgorithmWrapper):
    def __init__(self, parameters, dataset, verbose=0):
        super().__init__(parameters, dataset, verbose)
        self.name = 'K-Nearest Neighbours'
    
    def _predictOutliers(self, param, dataset=None):
        if dataset == None:
            dataset = self._dataset
        alg = KNN(n_neighbors = param, n_jobs = -1)
        alg.fit(dataset)
        return alg.labels_
    
class COFWrapper(AlgorithmWrapper):
    def __init__(self, parameters, dataset, verbose=0):
        super().__init__(parameters, dataset, verbose)
        self.name = 'Connectivity Based Outlier Factor'
        
    def _predictOutliers(self, param, dataset=None):
        if dataset == None:
            dataset = self._dataset
        alg = COF(n_neighbors = param)
        alg.fit(dataset)
        print(alg.labels_)
        return alg.labels_
    
class IFWrapper(AlgorithmWrapper):
    def __init__(self, parameters, dataset, verbose=0):
        super().__init__(parameters, dataset, verbose)
        self.name = 'Isolation Forest'
        
    def _predictOutliers(self, param, dataset=None):
        if dataset == None:
            dataset = self._dataset
        alg = IsolationForest()
        return np.array([1 if x == -1 else 0 for x in alg.fit_predict(dataset)])
    
class LOFWrapper(AlgorithmWrapper):
    def __init__(self, parameters, dataset, verbose=0):
        super().__init__(parameters, dataset, verbose)
        self.name = 'Local Outlier Factor'        
        
    def _predictOutliers(self, param, dataset=None):
        if dataset == None:
            dataset = self._dataset
        alg = LocalOutlierFactor(n_neighbors = param)
        return np.array([1 if x == -1 else 0 for x in alg.fit_predict(dataset)])
    
# def parse(dataset):
#     # print(dataset.describe())
#     # print(dataset.iloc[5])
    
#     # print(dataset.dtypes)
    
#     dataset = dataset.dropna(axis=1, how='all')
#     dataset = dataset.select_dtypes(exclude=['object'])
    
#     fill = dict()
#     for col in dataset:
#         val = dataset[col].quantile(0.5)
#         if dataset[col].mean() < dataset[col].quantile(0.8) and dataset[col].mean() > dataset[col].quantile(0.2):
#             val = dataset[col].mean()
#         fill[col] = val
    
#     dataset = dataset.fillna(value=fill)
    
#     # print(dataset.describe())
#     # print(dataset.iloc[5])
#     # print(dataset.mean())
    
    
#     print(dataset.describe(include='all'))
#     # print(dataset.iloc[5])
#     # print(dataset.mean())
    
#     return dataset
    
# def printRes(dataset, outliers):
#     for res in outliers:
#         print(dataset[res==1])
#     #   dataset[res==1].to_csv('outliers.csv')
        
# def test(dataset, verbose=0):
#     # algorithms = [KNNWrapper(range(20, 30), dataset, verbose),
#     #               IFWrapper([1], dataset, verbose),
#     #               LOFWrapper(range(20, 30), dataset, verbose)]
#     # results = []
    
#     # for a in algorithms:
#     #     if verbose > 0:
#     #         print('\n===================')
#     #         print(a.name)
#     #     outliers, best_param = a.runTests()
#     #     results.append(a.singleRun(best_param).index)
#     #     if verbose > 0:
#     #         print('\n===================\n')
            
#     # overlap = results[0]
#     # for r in range(1, len(results)):
#     #     overlap = compareIndexes(overlap, results[r])
    
#     # return overlap
        
    
#     # print('\n===================')
#     # print('K-Nearest Neighbours')
#     knn = KNNWrapper(range(20, 30), dataset, 2)
#     # outliers, best_param = knn.runTests()
#     # print(best_param)
#     best_param = 29
#     knn_res = knn.singleRun(best_param).index
#     # printRes(inputData, outliers)
#     # print('\n===================\n')
    
    
#     # print('\n===================')
#     # print('Isolation Forest')
#     iforest = IFWrapper([1], dataset, 2)
#     # outliers, best_param = iforest.runTests()
#     # print(best_param)
#     best_param = 1
#     i_res = iforest.singleRun(best_param).index
#     # printRes(inputData, outliers)
#     # print('\n===================\n')
    
    
#     # print('\n===================')
#     # print('Connectivity Based Outlier Factor')
#     # cof = COFWrapper(range(2, 50), dataset)
#     # outliers, best_param = cof.runTests()
#     # print(best_param)
#     # printRes(inputData, outliers)
#     # print('\n===================\n')
    
    
#     # print('\n===================')
#     # print('Local Outlier Factor')
#     lof = LOFWrapper(range(20, 30), dataset)
#     # outliers, best_param = lof.runTests()
#     # print(best_param)
#     best_param = 20
#     lof_res = lof.singleRun(best_param).index
#     # printRes(inputData, ou1tliers)
#     # print('\n===================\n')
    
#     # compareIndexes(knn_res, i_res)
#     # compareIndexes(knn_res, lof_res)
#     # compareIndexes(i_res, lof_res)
#     return [len(knn_res), len(i_res), len(lof_res)], compareIndexes(knn_res, compareIndexes(i_res, lof_res))
    
    
# def compareIndexes(i1, i2):
#     sameIs = set()
#     for i in i1:
#         if i in i2:
#             sameIs.add(i)
#     for i in i2:
#         if i in i1:
#             sameIs.add(i)
#     # print(np.array(sameIs))
#     print(len(sameIs))
#     return sameIs
    

# def main():
    
#     # print('hello world\n')
    
#     start = datetime.now()
#     inputData = pd.read_csv('reduced_log.csv')
#     pd.options.display.max_columns = inputData.shape[1]
#     # inputData = pd.read_csv('parsed.csv')
#     print(datetime.now() - start)
    
#     dataset = parse(inputData)
    
#     low_values = dict()
#     high_values = dict()
#     for col in dataset:
#         low_values[col] = dataset[col].quantile(0.2)
#         high_values[col] = dataset[col].quantile(0.8)
#     # print(low_values)
#     # print(high_values)
    
#     outlier_counts, overlap = test(dataset)
    
#     print(outlier_counts)
    
#     outliers_dataset = inputData.loc[list(overlap)]
#     # print(outliers_dataset)
#     # outliers_dataset.to_csv('outliers.csv')
#     # outliers_dataset = pd.read_csv('outliers.csv')
    
#     for i, r in outliers_dataset.iterrows():
#         print(r)
#         for l in low_values:
#             if r[l] < low_values[l]:
#                 print('lower than 20% of', l)
#         for h in high_values:
#             if r[h] > high_values[h]:
#                 print('higher than 80% of', h)
    
#     print('\nEnd of execution\n')
#     print(datetime.now() - start)
    
# main()
