# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 01:20:12 2018

@author: Giorgia Tandoi
"""
import mlp_utils as mlp
from math import sqrt
import matplotlib.pyplot as plt
import time as t
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import numpy as np

"""
    This Script performs the forecasting pipeline for 
    a SINGLE MLP only one time 
    and prints to the console a scatter plot, and 
    another plot picturing the real testing values against the 
    predicted ones. 
"""

#Parameters Settings
split_limit = 186
diff_interval = 1
time_steps = 3
lag_size = 6
n_epochs = 10
batch_size = 31
n_neurons = 3


start = t.time()
#-----------------------------------------------load dataset
data = pd.DataFrame.from_csv(r"C:\Users\Utente\Desktop\TESI\CODE\CURRENT\DATA\DailyMeanDataset.csv", 
                             sep='\t', encoding='utf-8')

#project solar radiation values
r_inc = data['r_inc'].values

#features, drop the 'r_inc' values if it is not included in the feature set
#data = data.drop(columns=['r_inc'])
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
train, test = supervised[:split_limit,:], supervised[split_limit:,:]

#---------------------------------------------5-fit the model and make a prediction 
model = mlp.fit_mlp(train, batch_size, n_epochs, n_neurons, time_steps,
                lag_size, n_features)

X_train = pd.DataFrame(data=train[:, 0:n_features]).values

test_X, test_y = test[:, 0:n_features], test[:, n_features:]

yhat = np.array(model.predict(test_X))

#----------------------------------------------6-7 Invert all transformations
yhat = mlp.inverse_transform(r_inc, test_X, yhat, n_features, scaler)
test_y = mlp.inverse_transform(r_inc, test_X, test_y, n_features, scaler)

end = t.time()

#----------------------------------------------8 Evaluate Performance for every time step
for i in range(time_steps):
    actual = test_y[:, i]
    predicted = yhat[:, i]
    rmse = sqrt(metrics.mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    print("-----------------------------")
    print('t+%d RMSE: %f' % ((i+1), rmse))
    print('t+%d mape: %f' % ((i+1), mape))

time = end-start
print("-----------------------------")
print('Test Time: %f' %time)
# line plot of observed vs predicted
plt.figure(1)
split_limit = split_limit+time_steps+1
plt.plot(r_inc[split_limit:-time_steps], label = "Real solar Radiation")
plt.plot(yhat[:,-1], label = "Predicted solar Radiation")
plt.legend()
plt.show()

plt.figure(1)
plt.scatter(test_y[:,-1],yhat[:,-1],c='r', alpha=0.5, label='Solar Radiation')
plt.xlabel("Real target values")
plt.ylabel("Predicted target values")
axes = plt.gca()
m, b = np.polyfit(test_y[:,-1], yhat[:,-1], 1)
x_plot = np.linspace(axes.get_xlim()[0], axes.get_xlim()[1], 100)
plt.plot(x_plot, m*x_plot + b,'-')
plt.legend(loc='upper left')
plt.show()