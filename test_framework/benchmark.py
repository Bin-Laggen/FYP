# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 22:39:49 2020

@author: domis
"""

import pandas as pd

import psutil
import os
import sys
import threading
import time

import argparse

from datetime import datetime

from sklearn import metrics

from log_parser import CSVLogParser, JSONLogParser, ExcelLogParser
from test_runner import OutlierTestRunner
from wrappers import KNNWrapper, IFWrapper, LOFWrapper
from report_generator import ReportGenerator

global run

def benchmarkAllAlgorithms(parsed_dataset, input_data, outlier_label, stats, verbose):
    print('KNN')
    knn = KNNWrapper(parsed_dataset)
    knn_stats = dict()
    singleBenchmark(knn, input_data, outlier_label, knn_stats, verbose)
    # print(knn_stats)
    print('============================================================================')
    print()
    
    print('IFOREST')
    iforest = IFWrapper(parsed_dataset)
    if_stats = dict()
    singleBenchmark(iforest, input_data, outlier_label, if_stats, verbose)
    # print(if_stats)
    print('============================================================================')
    print()
    
    print('LOF')
    lof = LOFWrapper(parsed_dataset)
    lof_stats = dict()
    singleBenchmark(lof, input_data, outlier_label, lof_stats, verbose)
    # print(lof_stats)
    print('============================================================================')
    print()
    print('<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>')
    print()
    
    stats['KNN'] = knn_stats
    stats['IFOREST'] = if_stats
    stats['LOF'] = lof_stats
    return
        
def singleBenchmark(algorithm, input_data, outlier_label, stats, verbose):
    
    outliers_in_dataset = input_data[input_data[outlier_label]==1].index.size
    
    for i, param in enumerate(algorithm.getParams()):
        if verbose > 0:
            print('Param:', param)
        
        start = datetime.now()
        res = algorithm.singleRun(param)
        processing_time = datetime.now() - start
        
        C = metrics.confusion_matrix(input_data[outlier_label], res)
        
        od_acc = round((C[1,1] / outliers_in_dataset), 4)
        cl_acc = round(metrics.accuracy_score(input_data[outlier_label], res), 4)
        bal_acc = round(metrics.balanced_accuracy_score(input_data[outlier_label], res), 4)
        pre_acc_bin = round(metrics.precision_score(input_data[outlier_label], res, average='binary', zero_division=0), 4)
        pre_acc_mac = round(metrics.precision_score(input_data[outlier_label], res, average='macro', zero_division=0), 4)
        com_acc = round(od_acc * cl_acc, 4)
        
        if verbose > 0:
            print('True inliers:', C[0,0])
            print('False inliers:', C[1,0])
            print('True outliers:', C[1,1])
            print('False outliers:', C[0,1])
            print(C[1,1] + C[0,1], '/', outliers_in_dataset,
                  '=', (C[1,1] + C[0,1] / outliers_in_dataset) * 100, '%')
            print('Outlier Detection Accuraccy', od_acc * 100, '%')
            print('Classification Accuraccy', cl_acc * 100, '%')
            print('Combined Accuraccy', com_acc * 100, '%')
            print('Balanced Accuraccy', bal_acc * 100, '%')
            print('Precision (Binary) Accuraccy', pre_acc_bin * 100, '%')
            print('Precision (Macro) Accuraccy', pre_acc_mac * 100, '%')
            print(processing_time)
            print()
        
        stats[i] = {'parameter': param, 'num_found': C[1,1] + C[0,1], 'od_acc': od_acc * 100, 
                    'cl_acc': cl_acc * 100, 'com_acc': com_acc * 100, 'bal_acc': bal_acc * 100,
                    'pre_acc_bin': pre_acc_bin * 100, 'pre_acc_mac': pre_acc_mac * 100, 
                    'time': processing_time}
        
    return

def memoryCounterFunc(process):
    global run
    while run:
        time.sleep(1)
        print('\tMemory usage: %.6g' % (process.memory_info().rss / (1024 * 1024)), 'MB', end='\r') 
        
def readArgs():
    parser = argparse.ArgumentParser(description='    Outlier Detector Benchmark    ')
    parser.add_argument('file', help='Dataset file to use for benchmarking', type=str)
    parser.add_argument('outlier', help='Label of outlier column', type=str)
    parser.add_argument('-o', '--output', help='Output directory for results', default='result', type=str)
    parser.add_argument('-v', '--verbose', help='Set verbosity level', default=0, type=int,
                        choices=[0, 1, 2])
    options_group = parser.add_argument_group(description='Dataset info')
    options_group.add_argument('-j', '--json', help='JSON format of dataset', default='records',
                               choices=['split','records','index', 'columns','values', 'table'])
    options_group.add_argument('-s', '--sheet', help='Name of Excel sheet containing dataset, all if not specified',
                               default=None)
    options_group.add_argument('-i', '--index', help='Dataset index column (-1 if N/A)', default=0,
                               type=int)
    args = parser.parse_args()
    return args

def main():
    
    args = readArgs()
    
    filename = args.file
    
    try:
        file = open(filename, 'r')
        file.close()
    except FileNotFoundError:
        print('===   ERROR   ===', file=sys.stderr)
        print('Input file does not exist, check filename or path:', filename, file=sys.stderr)
        exit(1)
        
    outlier_label = args.outlier
    output_dir = args.output
    
    index = args.index
    if index < 0:
        index = None
    
    verbose = args.verbose
    
    file_type = filename.split('.')[-1]
    
    if verbose > 0:
        print()
        print('File:', filename)
        print('Type:', file_type)
        print('Outlier label:', outlier_label)
        print('Verbose level:', verbose)
        print('Index column:', index)
        print('Sheet:', args.sheet)
        print('JSON:', args.json)
        print()
    
    
    if file_type.lower() == 'csv':
        lp = CSVLogParser(filename, index, outlier_label, verbose)
    elif file_type.lower() == 'json':
        lp = JSONLogParser(filename, args.json, outlier_label, verbose)
    elif file_type.lower().startswith('xls'):
        lp = ExcelLogParser(filename, args.sheet, index, outlier_label, verbose)
    else:
        print('===   ERROR   ===', file=sys.stderr)
        print('Unknown file type:', file_type, file=sys.stderr)
        exit(1)
    
    try:
        parsed_dataset, original = lp.parse()
    except KeyError:
        print('===   ERROR   ===', file=sys.stderr)
        print('Invalid outlier label:', outlier_label, file=sys.stderr)
        exit(1)
    
    if verbose > 0:
        print(parsed_dataset)
    
    global run
    run = True
    show_mem_usage = parsed_dataset.index.size > 25000
    
    if show_mem_usage and verbose > 0:
        process = psutil.Process(os.getpid())
        t = threading.Thread(target=memoryCounterFunc, args=(process,), daemon=True)
        t.start()
        
    try:
        stats = dict()
        benchmarkAllAlgorithms(parsed_dataset, original, outlier_label, stats, verbose)
    except pd.core.indexing.IndexingError:
        print('===   ERROR   ===', file=sys.stderr)
        print('Incorrect index column provided', file=sys.stderr)
        print('If no index is specified in the dataset pass -i -1 as a parameter to this benchmark', file=sys.stderr)
        exit(1)
    finally:
        if show_mem_usage:
            run = False
            t.join()
        
    print()
    rg = ReportGenerator(output_dir, verbose)
    rg.benchmarkReport(stats)
    print()
    print('============================================================================')
    print('|                             END OF BENCHMARK                             |')
    print('============================================================================')
    print()
    
    return

if __name__ == '__main__':
    main()