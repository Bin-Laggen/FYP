# -*- coding: utf-8 -*- 
"""
Created on Fri Apr  3 13:43:03 2020

@author: domis
"""

import numpy as np
import pandas as pd

from datetime import datetime

import matplotlib.pyplot as plt

import os
import shutil

class ReportGenerator():
    def __init__(self, directory='result/', verbose=0):
        if not directory.endswith(('/', '\\')):
            directory += '/'
        self._directory = directory
        if not os.path.isdir(self._directory):
            os.mkdir(self._directory)
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
    
    def benchmarkReport(self, stats):
        knn_df = pd.DataFrame.from_dict(stats['KNN'], orient='index')
        if_df = pd.DataFrame.from_dict(stats['IFOREST'], orient='index')
        lof_df = pd.DataFrame.from_dict(stats['LOF'], orient='index')
        
        output = '<!DOCTYPE html><html><head><link rel="stylesheet" type="text/css" href="benchmark.css"></head><body>'
        output += '<h1>Benchmark Result ' + datetime.now().strftime("%d/%m/%y %H:%M") + '</h1>'
        output += '<h3>K-Nearest Neighbours</h3><div class="wrap">'
        output += self._dataframeToHTML(knn_df, 'KNN')
        output += '<div class="image"><img src="KNN.png"/></div></div>'
        output += '<h3>Isolation Forest</h3><div class="wrap">'
        output += self._dataframeToHTML(if_df, 'IFOREST')
        output += '<div class="image"><img src="IFOREST.png"/></div></div>'
        output += '<h3>Local Outlier Factor</h3><div class="wrap">'
        output += self._dataframeToHTML(lof_df, 'LOF')
        output += '<div class="image"><img src="LOF.png"/></div></div>'
        output += '</body></html>'
        
        output = output.replace('parameter', 'Parameter')
        output = output.replace('num_found', 'Number of outliers detected')
        output = output.replace('od_acc', 'Outlier Detection Accuracy (%)')
        output = output.replace('cl_acc', 'Classification Accuracy (%)')
        output = output.replace('com_acc', 'Combined Accuracy (%)')
        output = output.replace('bal_acc', 'Balanced Accuracy (%)')
        output = output.replace('pre_acc_bin', 'Precision (Binary) Score (%)')
        output = output.replace('pre_acc_mac', 'Precision (Macro) Score (%)')
        output = output.replace('time', 'Processing Time (H:M:S)')
        
        shutil.copy2('benchmark.css', self._directory + 'benchmark.css')
        with open(self._directory + 'benchmark_result.html', 'w') as file:
            file.write(output)
            file.close()
            
        self._graphBenchmark(knn_df, 'KNN')
        self._graphBenchmark(if_df, 'IFOREST')
        self._graphBenchmark(lof_df, 'LOF')
        return
    
    def _dataframeToHTML(self, df, name):
        best_acc = df.loc[df['od_acc']==df['od_acc'].max()]
        best = best_acc['time'].idxmin()
        worst_acc = df.loc[df['od_acc']==df['od_acc'].min()]
        worst = worst_acc['time'].idxmax()
        output = '<table class="dataframe" id="' + name + '"><thead><tr>'
        for col in df:
            output += '<th>' + col + '</th>'
        output += '</tr></thead><tbody>'
        for row in df.itertuples():
            if row.Index == best:
                output += '<tr class="best">'
            elif row.Index == worst:
                output += '<tr class="worst">'
            else:
                output += '<tr>'
            for col in df.columns:
                att = getattr(row, col)
                if isinstance(att, float):
                    att = round(att, 3)
                output += '<td>' + str(att) + '</td>'
            output += '</tr>'
        output += '</tbody></table>'
        return output
    
    def _graphBenchmark(self, df, name):
        labels = [round(item, 3) for item in df['parameter']]
        od = [round(item, 2) for item in df['od_acc']]
        com = [round(item, 2) for item in df['com_acc']]
        
        x = np.arange(len(labels))
        width = 0.35  
        
        fig, ax = plt.subplots()
        fig.set_size_inches(15, 10)
        rects1 = ax.bar(x - width/2, od, width, label='Outlier')
        rects2 = ax.bar(x + width/2, com, width, label='Combined')
        
        ax.set_ylabel('Accuracy Scores')
        ax.set_title(name + ' accuracy scores based on parameter')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_facecolor('#1F2739')
        ax.legend()
        
        self._autolabelBar(ax, rects1)
        self._autolabelBar(ax, rects2)
        
        fig.tight_layout()
        
        plt.savefig(self._directory + name)
        return
        
    def _autolabelBar(self, ax, rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        color='white')
        return
    
    