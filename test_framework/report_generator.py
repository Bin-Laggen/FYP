# -*- coding: utf-8 -*- 
"""
Created on Fri Apr  3 13:43:03 2020

@author: domis
"""

import numpy as np
import pandas as pd

class ReportGenerator():
    def __init__(self, verbose=0):
        self._verbose = verbose
        
    def calculateMetrics(self, parsed_dataset, unparsed_dataset, outliers):
        self._parsed_dataset = parsed_dataset
        self._unparsed_dataset = unparsed_dataset
        self._outliers = outliers
        
        outliers_dataset = self._generateOutlierComments()
        pd.options.display.max_columns = outliers_dataset.shape[1]
        
        total = outliers_dataset.index.size
        print(outliers)
        outlier_count = np.count_nonzero(outliers)
        percentage = self._outlierPercentage(outlier_count, total)

        print('=============================\n')
        print('Dataset provided:\n')
        print(self._unparsed_dataset)
        print('\n\n=============================\n')
        print('Dataset after parsing:\n')
        print(self._parsed_dataset)
        print('\n\n=============================\n')
        print('Total rows in dataset:\n')
        print(total)
        print('\n\n=============================\n')
        print('Anomalies detected:\n')
        print(outliers_dataset[outliers==1])
        print('\n\n=============================\n')
        print('Number of anomalies found:\n')
        print(outlier_count)
        print('\n\n=============================\n')
        print('Percentage of anomalies in dataset:\n')
        print(percentage, '%')
        print('\n\n=============================\n')
        
    def _createRangeDicts(self):
        low_values = dict()
        high_values = dict()
        for col in self._parsed_dataset:
            low_values[col] = self._parsed_dataset[col].quantile(0.2)
            high_values[col] = self._parsed_dataset[col].quantile(0.8)
        return low_values, high_values
    
    def _generateOutlierComments(self):
        low_values, high_values = self._createRangeDicts()
        outliers_dataset = self._unparsed_dataset.copy()
        outliers_dataset['outlier'] = self._outliers
        
        comments = []
        for i, r in outliers_dataset.iterrows():
            comments.append(self._genComment(r, low_values, high_values))
        outliers_dataset['comment'] = comments
        return outliers_dataset
    
    def _genComment(self, row, low, high):
        com = ''
        # if r['outlier'] == 1:
        for l in low:
            if row[l] < low[l]:
                com += '"' + l + '"' + ' below quantile range;'
        for h in high:
            if row[h] > high[h]:
                com += '"' + h + '"' + ' above quantile range;'
        if com == '':
            com = None
        return com
    
    def _outlierPercentage(self, outliers, total):
        return (outliers / total) * 100
        
        