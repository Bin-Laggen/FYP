# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 22:39:49 2020

@author: domis
"""

import pandas as pd
import numpy as np

import psutil
import os
import sys
import threading
import time

import argparse

from datetime import datetime

from log_parser import CSVLogParser, JSONLogParser, ExcelLogParser
from test_runner import OutlierTestRunner
from wrappers import KNNWrapper, IFWrapper, LOFWrapper

global run

def benchmarkAllAlgorithms(parsed_dataset, input_data, outlier_label, process):
    print('KNN')
    knn = KNNWrapper(parsed_dataset)
    singleBenchmark(knn, input_data, outlier_label, process)
    print('============================================================================')
    print()
    
    print('IF')
    iforest = IFWrapper(parsed_dataset)
    singleBenchmark(iforest, input_data, outlier_label, process)
    print('============================================================================')
    print()
    
    print('LOF')
    lof = LOFWrapper(parsed_dataset)
    singleBenchmark(lof, input_data, outlier_label, process)
    print('============================================================================')
    print()
    print('<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>')
    print()
    return
        
def singleBenchmark(algorithm, input_data, outlier_label, process):
    for param in algorithm.getParams():
        print('Param:', param)
        start = datetime.now()
        res = algorithm.singleRun(param)
        end = datetime.now() - start
        outliers = input_data.loc[pd.Series(res==1)]
        overlap = OutlierTestRunner.compareOutliers(res, input_data[outlier_label])
        print(outliers.index.size, '/', input_data[input_data[outlier_label]==1].index.size, '=', (outliers.index.size / input_data[input_data[outlier_label]==1].index.size) * 100, '%')
        print((np.count_nonzero(overlap) / input_data[input_data[outlier_label]==1].index.size) * 100, '%')
        print(end)
        print()
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
    parser.add_argument('-v', '--verbose', help='Set verbosity level', default=0, type=int, choices=[0, 1, 2])
    options_group = parser.add_argument_group(description='Dataset info')
    options_group.add_argument('-j', '--json', help='JSON format of dataset', default='records', choices=['split','records','index', 'columns','values', 'table'])
    options_group.add_argument('-s', '--sheet', help='Name of Excel sheet containing dataset, all if not specified', default=None)
    options_group.add_argument('-i', '--index', help='Dataset index column (-1 if N/A)', default=0, type=int)
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
    
    index = args.index
    if index < 0:
        index = None
    
    verbose = args.verbose
    
    file_type = filename.split('.')[-1]
    
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
    
    print(parsed_dataset)
    
    process = psutil.Process(os.getpid())
    global run
    run = True
    t = threading.Thread(target=memoryCounterFunc, args=(process,), daemon=True)
    t.start()
        
    try:
        benchmarkAllAlgorithms(parsed_dataset, original, outlier_label, process)
    except Exception as e:
        run = False
        t.join()
        raise e
    
    print()
    print('============================================================================')
    print('|                             END OF BENCHMARK                             |')
    print('============================================================================')
    print()
    
    run = False
    t.join()
    return

if __name__ == '__main__':
    main()