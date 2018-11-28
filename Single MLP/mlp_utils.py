# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 01:20:12 2018

@author: Giorgia Tandoi
"""
from math import sqrt
import matplotlib.pyplot as plt
import time as t
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import numpy as np

# convert series to supervised learning
def timeseries_to_supervised(data, r_inc, n_in=1, n_out=1, dropnan=True):

	n_vars = 1 if type(data) is list else data.shape[1]
	df_inp = pd.DataFrame(data=data)
	df_out = pd.DataFrame(data=r_inc)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df_inp.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df_out.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (n_vars+1))]
		else:
			names += [('var%d(t+%d)' % (n_vars+1, i))]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True) 
	return np.array(agg.values)

  
# create a differenced series
def difference(dataset, interval=1):
	diff = list()  
	dataset = np.array(dataset)
	for i in range(interval, len(dataset)):    
		value = dataset[i] - dataset[i - interval]
		diff.append(value.tolist())
	return np.array(diff)

# invert differenced value and rescale transformation    
def inverse_transform(history, test_X, yhat, n_features, scaler):
    inverted = list()
    for i in range(len(yhat)):
        forecast = np.array(yhat[i])
        if yhat.ndim==1: 
            forecast = forecast.reshape(1, 1)
        else:
            forecast = forecast.reshape(1, yhat.shape[1])
        X = np.array(test_X[i])
        X = X.reshape(1, test_X.shape[1])
        X_and_forecast = np.concatenate((X,forecast), axis=1)
        inv_scale = np.array(scaler.inverse_transform(X_and_forecast))
        inv_scale = np.array(inv_scale[0, n_features:])
        
        index = len(history) - len(yhat) +i -1
        last_ob = history[index]
        inverted_diff = list()
        inverted_diff.append(inv_scale[0]+last_ob)
        for j in range(1, len(inv_scale)):
            inverted_diff.append(inv_scale[j] + inverted_diff[j-1])
        inverted.append(inverted_diff)
    inverted = np.array(inverted)
    return inverted


# fit an LSTM network to training data
def fit_mlp(train, batch_size, nb_epoch, neurons, time_steps, lag_size, n_features):
	n_features = n_features*lag_size
	X, y = train[:, 0:n_features], train[:, n_features:]
	model = MLPRegressor(hidden_layer_sizes=neurons, activation='tanh', 
                      solver='lbfgs', batch_size=batch_size, random_state=1)
	model.fit(X, y)
	return model