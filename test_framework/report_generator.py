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
    algorithm_names = {'knn': 'K-Nearest Neighbours', 'iforest': 'Isolation Forest', 'lof': 'Local Outlier Factor'}
    def __init__(self, directory='result/', verbose=0):
        if not directory.endswith(('/', '\\')):
            directory += '/'
        self._directory = directory
        if not os.path.isdir(self._directory):
            os.mkdir(self._directory)
        self._verbose = verbose
        
    def reportOutlierDetection(self, parsed_dataset, unparsed_dataset, results):
        overlap = None
        overlap_data = None
        if 'overlap' in results:
            overlap = results['overlap']
            del results['overlap']
        
        metrics = dict()
        for alg_name, res in results.items():
            m, outlier_dataset = self._calculateMetrics(parsed_dataset, unparsed_dataset, res['best_labels'])
            m['best_param'] = res['best_param']
            metrics[alg_name] = m
            
            self._generateAlgorithmPage(alg_name, res, outlier_dataset)
            
        if type(overlap) != type(None):
            metrics['overlap'], overlap_data = self._calculateMetrics(parsed_dataset, unparsed_dataset, overlap)
            
        metrics_df = pd.DataFrame.from_dict(metrics)
        metrics_html = self._metricsDataframeToHTML(metrics_df)
        
        output = '<!DOCTYPE html><html><head><link rel="stylesheet" type="text/css" href="anomaly.css"></head><body>'
        output += '<h1>Anomaly Detection Result ' + datetime.now().strftime("%d/%m/%y %H:%M") + '</h1>'
        output += '<h3>Results</h3>'
        output += metrics_html
        
        if type(overlap_data) != type(None):
            output += '<h3>Anomalies found by all algorithms</h3>'
            output += overlap_data.to_html().replace('border="1"', '')
            
        output += '</body></html>'
        
        shutil.copy2('anomaly.css', self._directory + 'anomaly.css')
        with open(self._directory + 'anomaly_result.html', 'w') as file:
            file.write(output)
            file.close()
            
        return
    
    def _generateAlgorithmPage(self, alg_name, res, outlier_dataset):
        if alg_name in self.__class__.algorithm_names:
            name = self.__class__.algorithm_names[alg_name]
        else:
            name = alg_name
            
        scores_html = self._scoresDataframeToHTML(pd.DataFrame.from_dict(res['scores'], orient='index'), alg_name, res['best_param'])
        
        alg_page = '<!DOCTYPE html><html><head><link rel="stylesheet" type="text/css" href="anomaly.css"></head><body>'
        alg_page += '<h1>' + name + '</h1>'
        alg_page += '<h3>Results for detecting best parameter</h3>'
        alg_page += scores_html
        alg_page += '<h3>Anomalies found by ' + name + '</h3>'
        alg_page += outlier_dataset.to_html().replace('border="1"', '')
        alg_page += '</body></html>'
        
        alg_page = alg_page.replace('parameter', 'Parameter')
        alg_page = alg_page.replace('sil', 'Silhouette Score')
        alg_page = alg_page.replace('cal_har', 'Calinski-Harabasz Index')
        alg_page = alg_page.replace('dav_bou', 'Davies–Bouldin Index')
        alg_page = alg_page.replace('combined', 'Combined Score')
        alg_page = alg_page.replace('time', 'Processing Time')
        
        with open(self._directory + alg_name + '.html', 'w') as file:
            file.write(alg_page)
            file.close()
    
    
    def _calculateMetrics(self, parsed_dataset, unparsed_dataset, outliers):
        
        total = parsed_dataset.index.size
        outlier_count = np.count_nonzero(outliers)
        percentage = self._outlierPercentage(outlier_count, total)
        
        outliers_dataset = self._generateOutlierComments(parsed_dataset, unparsed_dataset, outliers)
        pd.options.display.max_columns = outliers_dataset.shape[1]

        data = {'num_found': outlier_count, 'total': total, 'perc': percentage}
        return data, outliers_dataset
        
    def _createRangeDicts(self, parsed_dataset):
        low_values = dict()
        high_values = dict()
        for col in parsed_dataset:
            low_values[col] = parsed_dataset[col].quantile(0.2)
            high_values[col] = parsed_dataset[col].quantile(0.8)
        return low_values, high_values
    
    def _generateOutlierComments(self, parsed_dataset, unparsed_dataset, outliers):
        low_values, high_values = self._createRangeDicts(parsed_dataset)
        outliers_dataset = unparsed_dataset.copy()
        outliers_dataset['outlier'] = outliers
        outliers_dataset = outliers_dataset[outliers_dataset['outlier']==1]
        
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
    
    def _metricsDataframeToHTML(self, df):
        output = '<table class="dataframe" id="metrics"><thead><tr><th></th>'
        for col in df:
            if col in self.__class__.algorithm_names:
                output += '<th><a href="' + col + '.html" target="_blank">' + self.__class__.algorithm_names[col] + '</a></th>'
            else:
                output += '<th>' + col + '</th>'
        output += '</tr></thead><tbody>'
        for row in df.itertuples():
            output += '<tr>'
            output += '<th>' + str(row.Index) + '</th>'
            for col in df.columns:
                att = getattr(row, col)
                if np.isnan(att):
                    output += '<td></td>'
                else:
                    output += '<td>' + str(att) + '</td>'
            output += '</tr>'
        output += '</tbody></table>'
        output = output.replace('border="1" ', '')
        output = output.replace('num_found', 'Number of anomalies detected')
        output = output.replace('total', 'Total number of entries in dataset')
        output = output.replace('perc', 'Percentage of anomalies in dataset')
        output = output.replace('best_param', 'Best parameter for algorithm')
        return output
    
    def _scoresDataframeToHTML(self, df, name, best):
        best_ix = df[df['parameter']==best].index
        output = '<table class="dataframe" id="' + name + '"><thead><tr>'
        for col in df:
            output += '<th>' + col + '</th>'
        output += '</tr></thead><tbody>'
        for row in df.itertuples():
            if row.Index == best_ix:
                output += '<tr class="best">'
            else:
                output += '<tr>'
            for col in df.columns:
                att = getattr(row, col)
                output += '<td>' + str(att) + '</td>'
            output += '</tr>'
        output += '</tbody></table>'
        return output
    
    def benchmarkReport(self, stats):
        knn_df = pd.DataFrame.from_dict(stats['knn'], orient='index')
        if_df = pd.DataFrame.from_dict(stats['iforest'], orient='index')
        lof_df = pd.DataFrame.from_dict(stats['lof'], orient='index')
        
        output = '<!DOCTYPE html><html><head><link rel="stylesheet" type="text/css" href="benchmark.css"></head><body>'
        output += '<h1>Benchmark Result ' + datetime.now().strftime("%d/%m/%y %H:%M") + '</h1>'
        output += '<h3>K-Nearest Neighbours</h3><div class="wrap">'
        output += self._benchmarkDataframeToHTML(knn_df, 'knn')
        output += '<div class="image"><img src="knn.png"/></div></div>'
        output += '<h3>Isolation Forest</h3><div class="wrap">'
        output += self._benchmarkDataframeToHTML(if_df, 'iforest')
        output += '<div class="image"><img src="iforest.png"/></div></div>'
        output += '<h3>Local Outlier Factor</h3><div class="wrap">'
        output += self._benchmarkDataframeToHTML(lof_df, 'lof')
        output += '<div class="image"><img src="lof.png"/></div></div>'
        output += '</body></html>'
        
        output = output.replace('parameter', 'Parameter')
        output = output.replace('score', 'Decision Score')
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
            
        self._graphBenchmark(knn_df, 'knn')
        self._graphBenchmark(if_df, 'iforest')
        self._graphBenchmark(lof_df, 'lof')
        return
    
    def _benchmarkDataframeToHTML(self, df, name):
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
    
    def _graphBenchmark(self, df, alg_name):
        labels = [round(item, 3) for item in df['parameter']]
        od = [round(item, 2) for item in df['od_acc']]
        com = [round(item, 2) for item in df['com_acc']]
        
        x = np.arange(len(labels))
        width = 0.35  
        
        fig, ax = plt.subplots()
        fig.set_size_inches(15, 10)
        rects1 = ax.bar(x - width/2, od, width, label='Outlier')
        rects2 = ax.bar(x + width/2, com, width, label='Combined')
        
        print(alg_name)
        if alg_name in self.__class__.algorithm_names:
            name = self.__class__.algorithm_names[alg_name]
        else:
            name = alg_name
        
        ax.set_ylabel('Accuracy Scores')
        ax.set_xlabel('Parameter')
        ax.set_title(name + ' accuracy scores based on parameter')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_facecolor('#1F2739')
        ax.legend()
        
        self._autolabelBar(ax, rects1)
        self._autolabelBar(ax, rects2)
        
        fig.tight_layout()
        
        plt.savefig(self._directory + alg_name)
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
    
    