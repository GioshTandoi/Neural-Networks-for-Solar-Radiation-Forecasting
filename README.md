# Neural Networks for Solar Radiation Forecasting
##### Bachelor's degree thesis project 
This project contains different scripts in Python which have been developed in order to predict the expected value of solar radiation within three different time horizons: predictions for one day, three days and one week ahead. 
## Requirements 
Python 3.6 

Keras – Tensor Flow  2.2

Arch 4.6.0

Scikitlearn 0.20

## WHAT IS SOLAR RADIATION
Solar radiation, often called the solar resource, is a general term for the electromagnetic radiation emitted by the sun.  
This electromagnetic radiation coming from the sun is absorbed by the earth in the form of energy.

## WHY FORECASTING SOLAR RADIATION
### 1.
Nowadays human beings are moving forward to a sustainable world development, due to climate change and shrinking fossil resources, and renewable energy resources play an important role in this scenario. 
However, renewable energies like solar energy represent a tough challenge to face: though solar radiation is an inexhaustible source of energy, it is also extremely fluctuating, in a way that goes beyond human control. It suffers from non-stationary weather conditions, such as cloudy or partly cloudy days, which means unpredictability and uncontrollability. Energy producers and power grids need to ensure that the power system is able to handle expected and unexpected changes in production and consumption, while maintaining a continuous balance between the energy produced and the energy consumed due to the high cost of power reserve and battery storage. Therefore, forecasting the output power of such energy system, i.e. the solar radiation, allows producers and grid balancing authorities to manage the energy distribution, preserving service continuity and hence to gracefully integrate solar energy in the energy supply chain.          
There are different types of solar power plants; operatively, in the case of PV (Photovoltaic) power plants we can firstly obtain the predicted value of solar radiation based on a solar radiation model, then use the PV output formula to calculate the power output of the system.

## DATASET AND PROBLEM STRUCTURE
Countless studies have prooved the existence of a relantionship between solar radiation and other climate-related
variables such as Relative Humidity, Air Temperature, Rain, Cloudiness, Number of Sun Hours. 
This study has been possible thanks to a dataset made available by [SysMan](http://www.sys-man.it/ "SysMan") an italian company specialized in IT consulting and IoT solutions. The dataset comprises different climate variables, and spans a period that goes from the first of July 2017 to the 28th of February 2018. Four variables have been selected in order to predict solar radiation values: 
1. Maximum Air Temperature 
2. Number of Sun Hours 
3. Relative Humidity 
4. Solar Radiation 
These values originally had a hourly resolution: they have been averaged to compute their daily values, therefore the dataset utilized for the 
forecasting process ultimately comprised 243 records. 
![alt text](https://github.com/GioshTandoi/Neural-Networks-for-Solar-Radiation-Forecasting/blob/master/code_diagram.png)
  
The diagram above shows the process implemented in order to forecast the solar radiation: the forecasting pipeline. 
Essentially it showses the structure of the code reported in the scripts. 

### ALGORITHMS 
The forecasting pipeline has been implemented using two different types of ANN: 
#### Multilayer Perceptron
#### Long-Short-Term-Memory
These implementations have been tested for every time horizon with four different feature combinations. 
The best result for each time horizon has been reached using only Solar Radiation samples and a single Multilayer Perceptron. 
Therefore in order to improve the accuracy of the forecast an ensemble model has been developed. 
#### Bagged Multilayer Perceptrons
The ensemble model has been developed using the same forecasting pipeline. It improved the performance up to 0.6%. 
