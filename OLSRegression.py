# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 12:18:33 2018

@author: Kritin
"""

from sklearn import linear_model
reg = linear_model.LinearRegression()

datapath = '/Users/Kritin/Desktop/School_Stuff/Research/EEGMachineLearning/SSVEP_CNN/' 
SSVEP_l_data = io.loadmat(datapath+'SSVEP_l.mat')
SVEP_l = SSVEP_l_data['SSVEP_l']

xdata =  
ydata = 

reg.fit(xdata, ydata)
print(reg.coef_)
OLSScore = reg.score(xdata, ydata)
print(OLSScore)
