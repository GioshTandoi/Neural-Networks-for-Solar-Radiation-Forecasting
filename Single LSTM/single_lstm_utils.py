# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 17:18:03 2018

@author: Giorgia Tandoi
"""

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 500)

# convert series to supervised learning
def timeseries_to_supervised(features, r_inc, n_in=1, n_out=1, dropnan=True):

	n_vars = 1 if type(features) is list else features.shape[1]
	df_inp = pd.DataFrame(data=features)
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
	#print(agg)
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

# invert transformations
def inverse_transform(history, test_X, yhat, n_features, scaler):
    inverted = list()
    for i in range(len(yhat)):
        forecast = np.array(yhat[i])
        forecast = forecast.reshape(1, len(forecast))
        X = np.array(test_X[i])
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
def fit_lstm(train, batch_size, nb_epoch, neurons, time_steps, lag_size, n_features):
	n_features = n_features*lag_size
	X, y = train[:, 0:n_features], train[:, n_features:]
	X = X.reshape(X.shape[0], 1, X.shape[1])
	model = Sequential()
	model.add(LSTM(neurons, input_shape=(X.shape[1], X.shape[2])))
	model.add(Dense(y.shape[1]))
	model.compile(loss='mean_squared_error', optimizer='adam')
	model.fit(X, y, epochs=nb_epoch, batch_size=batch_size, verbose=0, shuffle=False)
	return model