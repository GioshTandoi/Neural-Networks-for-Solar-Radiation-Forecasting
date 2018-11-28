# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 22:53:07 2018

@author: Giorgia Tandoi
"""

import pandas as pd
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import time as t

"""
    This Script performs the forecasting pipeline for 
    a SINGLE LSTM only one time 
    and prints to the console a scatter plot, and 
    another plot picturing the real testing values against the 
    predicted ones. 
"""

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
        #X = X.reshape(1, X.shape[0])
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


#Parameters Settings
split_limit = 186
diff_interval = 1
time_steps = 3
lag_size = 8
n_epochs = 50
batch_size = 31
n_neurons = 2

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

#--------------------------------------------1-make time series stationary
diff_values = difference(raw_data, 1)
diff_r_inc = difference(r_inc, 1)

#-----------------------2-reframe the time series as supervised learning problem
supervised = timeseries_to_supervised(diff_values, diff_r_inc, lag_size, time_steps)

#----------------------------------------3-Rescale Dataset

scaler = MinMaxScaler(feature_range=(-1, 1))
scaled = scaler.fit_transform(supervised)
#---------------------------------------------4-split data in training and testing 
train, test = supervised[:split_limit,:], supervised[split_limit:,:]

#---------------------------------------------5-fit the model and make a prediction 
lstm_model = fit_lstm(train, batch_size, n_epochs, n_neurons, 
                      time_steps, lag_size, n_features)

n_features = n_features*lag_size
X_train = pd.DataFrame(data=train[:, 0:n_features]).values

X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
lstm_model.predict(X_train)

test_X, test_y = test[:, 0:n_features], test[:, n_features:]
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

yhat = lstm_model.predict(test_X)

#----------------------------------------------6-7 Invert all transformations
yhat = inverse_transform(r_inc, test_X, yhat, n_features, scaler)
test_y = inverse_transform(r_inc, test_X, test_y, n_features, scaler)

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
plt.plot(test_y[:,-1], label = "Real Solar Radiation")
plt.plot(yhat[:,-1], label = "Predicted Solar Radiation")
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