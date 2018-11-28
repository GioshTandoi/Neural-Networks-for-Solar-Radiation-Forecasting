# Neural Networks for Solar Radiation Forecasting
##### Bachelor's degree thesis project 
This project contains different scripts in Python which have been developed in order to predict the expected values of solar radiation within three different time horizons: one day, three days and one week ahead. 
## Requirements 
Python 3.6 

Keras – Tensor Flow  2.2

Arch 4.6.0

Scikitlearn 0.20

## WHAT IS SOLAR RADIATION
Solar radiation, often called the solar resource, is a general term for the electromagnetic radiation emitted by the sun.  
This electromagnetic radiation coming from the sun is absorbed by the earth in the form of energy, in particular, the solar irradiance is the sun’s radiant power  per unit area usually represented in units of kWm^(-2) or in MJ m^(-2 ) 〖day〗^(-1), whereas the solar irradiation in the sun’s radiant energy incident on a surface of unit area expressed in units 〖kWh m〗^(-2) or in MJm^(-2), and is typically expressed on an average daily basis for a given month.

## WHY FORECASTING SOLAR RADIATION
### 1.
Nowadays, human beings are moving forward to a sustainable world development, due to climate change and shrinking fossil resources, and renewable energy resources play a big and important role in this process. 
But renewable energies, like solar energy, represent also a tough challenge to face: though solar radiation is an inexhaustible source of energy, it is also extremely fluctuating, in a way that goes beyond human control. It suffers from non-stationary weather conditions, such as cloudy or partly cloudy days, which means unpredictability and uncontrollability. Energy producers and power grids need to ensure that the power system is able to handle expected and unexpected changes in production and consumption, while maintaining a continuous balance between the energy produced and the energy consumed, because power reserve and battery storage are an expensive burden. Therefore forecasting the output power of such energy system, and hence, solar radiation is important, because it allows producers and grid balancing authorities to manage the energy distribution to consumers, preserving service continuity and hence to gracefully integrate it in the energy supply chain.          
There are different types of solar power plants; in the case of PV (Photovoltaic) power plants, for example, we can firstly obtain the predicted value of solar radiation based on a solar radiation model, then use the PV output formula to calculate the power output of the system.

### 2. 
Solar radiation is one of the variables needed to compute the value of evapotranspiration through the Penman-Monteith equation, recommended by the Food and Agriculture Organization of the United Nations (FAO). 
Evapotranspiration is an important part of the water cycle, and it is determinant for an effective irrigation, since it represents the loss of water due to the evaporation and plant transpiration, therefore providing a relatively objective and reliable estimate of the water requirements of actively growing plants in a farm situation.

## DATASET AND PROBLEM STRUCTURE
Countless studies have prooved the existence of a relantionship between solar radiation and other climate-related
variables such as Relative Humidity, Air Temperature, Rain, Cloudiness, Number of Sun Hours. 
This study has been possible thanks to a dataset made available by [SysMan](http://www.sys-man.it/ "SysMan") an italian company specialized in IT consulting and IoT solutions. The dataset comprises different climate variables, and spans a period that goes from the first of July 2017 to the 28th of February 2018. Four variables have been selected in order to predict solar radiation values: 
1. Maximum Air Temperature 
2. Number of Sun Hours 
3. Relative Humidity 
4. Solar Radiation 
These values originally had a hourly resolution: they have been averaged on a daily basis, therefore the dataset utilized for the 
forecasting process ultimately comprised 243 records. 
![alt text](https://github.com/GioshTandoi/Neural-Networks-for-Solar-Radiation-Forecasting/blob/master/code_diagram.png)
  
The diagram above shows the process implemented in order to forecast the solar radiation: I call this the forecasting pipeline. 
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
