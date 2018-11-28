# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 15:51:37 2018

@author: Giorgia Tandoi
"""
from arch.bootstrap import StationaryBootstrap, CircularBlockBootstrap
import pandas as pd
import numpy as np

#takes as input a dataframe representing the dataset to bootstrap, and returns 
#n bootstrapped datasets as an array of dataframes, with the same column lables
#as the input data frame
def generateDatasets(data, n, block_size):
    newDatasets = []
    for i in range(n):
        data = np.array(data)
        bs = CircularBlockBootstrap(block_size, data)
        for d in bs.bootstrap(1):
            bs_data = d[0][0]
        bs_data = np.array(bs_data)
        newDatasets.append(bs_data)
    return newDatasets
