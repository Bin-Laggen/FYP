# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 15:13:28 2020

@author: domis
"""

import argparse
import sys

import log_parser
import test_runner
import report_generator

def readArgs():
    parser = argparse.ArgumentParser(description='    Anomaly Detector    ')
    
    parser.add_argument('file', help='Dataset file to run anomaly detection on', type=str)
    parser.add_argument('-v', '--verbose', help='Set verbosity level', default=0, type=int,
                        choices=[0, 1, 2])
    parser.add_argument('-a', '--algorithms', help='Set algorithms to use', nargs='*', 
                        default=test_runner.OutlierTestRunner.algorithms, type=str,
                        choices=test_runner.OutlierTestRunner.algorithms)
    parser.add_argument('-o', '--output', help='Output directory for results', default='result', type=str)
    parser.add_argument('-l', '--label', help='Outlier label', default=None, type=str)
    
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
        
    algorithms = args.algorithms
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
        print('Algorithms:', algorithms)
        print('Output dir:', output_dir)
        print('Verbose level:', verbose)
        print('Index column:', index)
        print('Sheet:', args.sheet)
        print('JSON:', args.json)
        print()
    
    outlier = args.label
    if file_type.lower() == 'csv':
        lp = log_parser.CSVLogParser(filename, index, outlier, verbose)
    elif file_type.lower() == 'json':
        lp = log_parser.JSONLogParser(filename, args.json, outlier, verbose)
    elif file_type.lower().startswith('xls'):
        lp = log_parser.ExcelLogParser(filename, args.sheet, index, outlier, verbose)
    else:
        print('===   ERROR   ===', file=sys.stderr)
        print('Unknown file type:', file_type, file=sys.stderr)
        exit(1)
    
    parsed_dataset, original = lp.parse()
    
    if verbose > 0:
        print(parsed_dataset)
    
    otr = test_runner.OutlierTestRunner(parsed_dataset, verbose)
    rg = report_generator.ReportGenerator(output_dir, verbose)
    
    results = otr.testAllAlgorithms(algorithms)
    rg.reportOutlierDetection(parsed_dataset, original, results)
    
    return


if __name__ == '__main__':   
    main()
    