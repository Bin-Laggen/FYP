# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 15:13:28 2020

@author: domis
"""

import argparse

import log_parser
import test_runner
import report_generator

def readArgs():
    parser = argparse.ArgumentParser(description='    Outlier Detector Benchmark    ')
    parser.add_argument('file', help='Dataset file to use for benchmarking', type=str)
    parser.add_argument('outlier', help='Label of outlier column', type=str)
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
    
    filename = 'reduced_log.csv'
    lp = log_parser.CSVLogParser(filename, index_column=0, verbose=2)
    parsed_dataset, unparsed_dataset = lp.parse()
    
    otr = test_runner.OutlierTestRunner(parsed_dataset, verbose=2)
    # algs = otr.getAvailableAlgorithms()
    
    # for alg_name in algs:
    #     otr.singleTest(alg_name)
    
    # otr.testAllAlgorithms()
    
    results, overlap = otr.testAllAlgorithms(['knn', 'lof'])
    
    rg = report_generator.ReportGenerator(verbose=2)
    
    # rg.calculateMetrics(parsed_dataset, unparsed_dataset, overlap)
    
    for r in results:
        rg.calculateMetrics(parsed_dataset, unparsed_dataset, r)
    
    return


if __name__ == '__main__':   
    main()
    