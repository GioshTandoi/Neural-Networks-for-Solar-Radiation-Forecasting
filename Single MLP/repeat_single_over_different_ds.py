# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 16:57:02 2018

@author: Giorgia Tandoi
"""
import pandas as pd
import numpy as np
from math import sqrt
import time as t
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import mlp_utils as mlp
import matplotlib.pyplot as plt
import os

"""
    This Script performs the forecasting pipeline for 
    a SINGLE MLP over 10 different bootstrapped datasets separately. 
    The results are stored in a .csv file, and the variance of the predictions 
    are calculated. 
"""

#Parameters settings
split_limit = 186
diff_interval = 1
time_steps = 7
lag_size = 5
n_epochs = 10
batch_size = 31
n_neurons = 3

#--------------------------------------------------PREPARE THE TESTING SET
#-----------------------------------------------load dataset
data = pd.DataFrame.from_csv(r"C:\Users\Utente\Desktop\TESI\CODE\CURRENT\DATA\DailyMeanDataset.csv", 
                                 sep='\t', encoding='utf-8')
    
#project solar radiation values
r_inc = data['r_inc'].values
raw_data = np.array(data.values)

n_features = raw_data.shape[1]
n_features = n_features*lag_size

#--------------------------------------------1-make time serie stationary
diff_values = mlp.difference(raw_data, 1)
diff_r_inc = mlp.difference(r_inc, 1)

#-----------------------2-reframe the time series as supervised learning problem
supervised = mlp.timeseries_to_supervised(diff_values, diff_r_inc, lag_size, time_steps)

#----------------------------------------3-Rescale the dataset
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled = scaler.fit_transform(supervised)
#---------------------------------------------4-split data in training and testing 
test = supervised[split_limit:,:]
#--------------------------------------------------UNIQUE TESTING SET 
test_X, test_y = test[:, 0:n_features], test[:, n_features:]
test_y = mlp.inverse_transform(r_inc, test_X, test_y, n_features, scaler)

#directory that contains the BOOTSTRAPPED datasets
directory_in_str = r"C:\Users\Utente\Desktop\TESI\CODE\CURRENT\DATA\bootstrapped_daily_mean_dataset\only_rinc"
directory = os.fsencode(directory_in_str)

mapes = []
rmses = []
predictions = []

#Loop over the bootstrapped datasets to apply the forecasting pipeline and 
#forecast the fixed training samples. 
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".csv"): 
            start = t.time()
            #-----------------------------------------------load dataset
            data = pd.DataFrame.from_csv(r"C:\Users\Utente\Desktop\TESI\CODE\CURRENT\DATA\bootstrapped_daily_mean_dataset\only_rinc\\"+filename, 
                                         sep='\t', encoding='utf-8')
            
            #project solar radiation values
            r_inc = data['r_inc'].values
            raw_data = np.array(data.values)
            n_features = raw_data.shape[1]
            n_features = n_features*lag_size
            
            #--------------------------------------------1-make time serie stationary
            diff_values = mlp.difference(raw_data, 1)           
            diff_r_inc = mlp.difference(r_inc, 1)
            
            #-----------------------2-reframe the time series as supervised learning problem
            supervised = mlp.timeseries_to_supervised(diff_values, diff_r_inc, lag_size, time_steps)
            
            #----------------------------------------3-Rescale the dataset
            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaled = scaler.fit_transform(supervised)
            #---------------------------------------------4-split data in training and testing 
            train = supervised[:split_limit,:]
            
            #---------------------------------------------5-fit the model and make a prediction 
            model = mlp.fit_mlp(train, batch_size, n_epochs, n_neurons, time_steps,
                            lag_size, n_features)
            
            X_train = pd.DataFrame(data=train[:, 0:n_features]).values
            
            yhat = np.array(model.predict(test_X))
                   
            #----------------------------------------------6-7 Invert all transformations
            yhat = mlp.inverse_transform(r_inc, test_X, yhat, n_features, scaler)         
            
            end = t.time()
            #----------------------------------------------8 Evaluate Performance
            rmse = sqrt(metrics.mean_squared_error(test_y[:,-1], yhat[:,-1] ))
            mape = np.mean(np.abs((test_y[:,-1] - yhat[:,-1]) / test_y[:,-1])) * 100
            time = end-start
            
            rmses.append(rmse)
            mapes.append(mape)
            predictions.append(yhat[:,-1])
            
            
errors = pd.DataFrame()
errors['rmse'] = rmses
errors['mape'] = mapes
    
errors.to_csv(r"C:\Users\Utente\Desktop\TESI\CODE\CURRENT\MLP\SINGLE\variability_over_different_datasets\three_days\variability.csv",
              sep='\t', encoding='utf-8')

var = np.var(predictions, axis=0)
var = np.mean(var)
print("VARIANCE")
print(var)