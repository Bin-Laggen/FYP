# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 15:13:28 2020

@author: domis
"""

import log_parser
import test_runner
import report_generator

def main():
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
    