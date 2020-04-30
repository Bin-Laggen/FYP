# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 17:33:13 2020

@author: domis
"""

import pandas as pd

class LogParser():
    def __init__(self, outlier=None, verbose=0):
        self._verbose = verbose
        self._outlier = outlier
        
    def _readfile(self):
        pass
        
    def parse(self):
        unparsed = self._dataset
        if self._outlier != None:
            self._dataset = self._dataset.drop(self._outlier, axis=1)
        if self._verbose > 0:
            print('Pre-parsed dataset\n')
            print(self._dataset.describe(include='all'))
            print()
        
        self._dataset = self._dataset.dropna(axis=1, how='all')
        self._dataset = self._dataset.select_dtypes(exclude=['object'])
        
        fill = dict()
        for col in self._dataset:
            val = self._dataset[col].quantile(0.5)
            if self._dataset[col].mean() < self._dataset[col].quantile(0.8) and self._dataset[col].mean() > self._dataset[col].quantile(0.2):
                val = self._dataset[col].mean()
            fill[col] = val
        
        self._dataset = self._dataset.fillna(value=fill)
        
        if self._verbose > 1:
            print('Mean values for each column')
            for k, v in fill.items():
                print(k + ':', v)
            print()
        
        if self._verbose > 0:
            print('Parsed dataset\n')
            print(self._dataset.describe(include='all'))
            print()
        
        return self._dataset, unparsed
    
class CSVLogParser(LogParser):
    def __init__(self, filename, index_column=None, outlier=None, verbose=0):
        super().__init__(outlier, verbose)
        self._filename = filename
        self._index_column = index_column
        self._readFile()
    
    def _readFile(self):
        self._dataset = pd.read_csv(self._filename, index_col=self._index_column)
        # print(self._dataset)
        # if self._index_column == None:
        #     print('adding index')
        #     self._dataset.set_index(pd.Index(range(self._dataset.index.size)), inplace=True)
        #     print(self._dataset)
        pd.options.display.max_columns = self._dataset.shape[1]
        
class JSONLogParser(LogParser):
    def __init__(self, filename, json_format='records', outlier=None, verbose=0):
        super().__init__(outlier, verbose)
        self._filename = filename
        self._json_format = json_format
        self._readFile()
        
    def _readFile(self):
        self._dataset = pd.read_json(self._filename, orient=self._json_format)
        pd.options.display.max_columns = self._dataset.shape[1]
        
class SQLLogParser(LogParser):
    def __init__(self, sql, con, index_column=None, outlier=None, verbose=0):
        super().__init__(outlier, verbose)
        self._sql = sql
        self._con = con
        self._index_column = index_column
        self._readFile()
        
    def _readFile(self):
        self._dataset = pd.read_sql(self._sql, self._con, index_col=self._index_column)
        if self._index_column == None:
            self._dataset.set_index(pd.Index(range(self._dataset.index.size)), inplace=True)
        pd.options.display.max_columns = self._dataset.shape[1]
    
class ExcelLogParser(LogParser):
    def __init__(self, filename, sheet_name=None, index_column=None, outlier=None, verbose=0):
        super().__init__(outlier, verbose)
        self._filename = filename
        self._index_column = index_column
        self._sheet = sheet_name
        self._readFile()
    
    def _readFile(self):
        self._dataset = pd.read_excel(self._filename, self._sheet, index_col=self._index_column)
        if self._index_column == None:
            self._dataset.set_index(pd.Index(range(self._dataset.index.size)), inplace=True)
        pd.options.display.max_columns = self._dataset.shape[1]
    
class PickleLogParser(LogParser):
    def __init__(self, filename, outlier=None, verbose=0):
        super().__init__(outlier, verbose)
        self._filename = filename
        self._readFile()
    
    def _readFile(self):
        self._dataset = pd.read_pickle(self._filename)
        pd.options.display.max_columns = self._dataset.shape[1]

class DataFrameLogParser(LogParser):
    def __init__(self, dataset, outlier=None, verbose=0):
        super().__init__(outlier, verbose)
        self._dataset = dataset

def main():
    
    # print('hello world\n')
    
    return

if __name__ == '__main__':   
    main()
