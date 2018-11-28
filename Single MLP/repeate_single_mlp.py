# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 01:09:09 2018

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

"""
    This Script repeats the forecasting pipeline for 
    a SINGLE MLP different times 
    and prints all the resulting errors in a .csv, and only for the 
    best one it also prints to the console a scatter plot, and 
    another plot picturing the real trsting values against the 
    predicted ones. 
"""
rmses = []
mapes = []
times = []

#Parameters Settings
split_limit = 186
diff_interval = 1
time_steps = 3
lag_size = 5
n_epochs = 20
batch_size = 31
n_neurons = 3

#looping variables
current_rmse = 100
ite = 0

for j in range(10):
    start = t.time()
    #-----------------------------------------------load dataset
    data = pd.DataFrame.from_csv(r"C:\Users\Utente\Desktop\TESI\CODE\CURRENT\DATA\DailyMeanDataset.csv", 
                                 sep='\t', encoding='utf-8')
    
    #project solar radiation values
    r_inc = data['r_inc'].values
    
    #features, drop the 'r_inc' values if it is not included in the feature set
    data = data.drop(columns=['r_inc'])
    raw_data = np.array(data.values)
    
    #--------------------------------------------1-make time serie stationary
    diff_values = mlp.difference(raw_data, 1)
    n_features = raw_data.shape[1]
    diff_r_inc = mlp.difference(r_inc, 1)
    
    #-----------------------2-reframe the time series as a supervised learning problem
    supervised = mlp.timeseries_to_supervised(diff_values, diff_r_inc, lag_size, time_steps)

    #----------------------------------------3-Rescale the dataset
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled = scaler.fit_transform(supervised)
    
    #---------------------------------------------4-split data in training and testing 
    train, test = supervised[:split_limit,:], supervised[split_limit:,:]
    
    #---------------------------------------------5-fit the model and make a prediction 
    model = mlp.fit_mlp(train, batch_size, n_epochs, n_neurons, time_steps,
                    lag_size, n_features)
    
    n_features = n_features*lag_size
    X_train = pd.DataFrame(data=train[:, 0:n_features]).values
    
    test_X, test_y = test[:, 0:n_features], test[:, n_features:]
    
    yhat = np.array(model.predict(test_X))    

    #----------------------------------------------6-7 Invert all transformations
    yhat = mlp.inverse_transform(r_inc, test_X, yhat, n_features, scaler)
    test_y = mlp.inverse_transform(r_inc, test_X, test_y, n_features, scaler)
    
    end = t.time()
    
    #----------------------------------------------8 Evaluate Performance
    rmse = sqrt(metrics.mean_squared_error(test_y[:,-1], yhat[:,-1] ))
    mape = np.mean(np.abs((test_y[:,-1] - yhat[:,-1]) / test_y[:,-1])) * 100
    time = end-start
    
    if rmse<current_rmse: 
        current_rmse = rmse
        iteration = ite
        best_prediction = yhat 
    ite = ite+1
    rmses.append(rmse)
    mapes.append(mape)
    times.append(time)
    
    
errors = pd.DataFrame()
errors['rmse'] = rmses
errors['mape'] = mapes
errors['time'] = times
    
errors.to_csv(r"C:\Users\Utente\Desktop\TESI\CODE\CURRENT\MLP\SINGLE\variability\one_day\4\variability.csv",
              sep='\t', encoding='utf-8')


#-----------------------------------Display best results 
print("Best result at iteration n. %d"%iteration)
for i in range(time_steps):
    actual = test_y[:, i]
    predicted = best_prediction[:, i]
    rmse = sqrt(metrics.mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    print("-----------------------------")
    print('t+%d RMSE: %f' % ((i+1), rmse))
    print('t+%d mape: %f' % ((i+1), mape))
    
print("-----------------------------")
print('Test Time: %f' %times[iteration])

# line plot of observed vs predicted
plt.figure(1)
plt.plot(test_y[:,-1], label = "Real Solar Radiation")
plt.plot(best_prediction[:,-1], label = "Predicted Solar Radiation")
plt.legend()
plt.show()

plt.figure(1)
plt.scatter(test_y[:,-1],best_prediction[:,-1],c='r', alpha=0.5, label='Solar Radiation')
plt.xlabel("Real target values")
plt.ylabel("Predicted target values")
axes = plt.gca()
m, b = np.polyfit(test_y[:,-1], best_prediction[:,-1], 1)
x_plot = np.linspace(axes.get_xlim()[0], axes.get_xlim()[1], 100)
plt.plot(x_plot, m*x_plot + b,'-')
plt.legend(loc='upper left')
plt.show()