# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 11:40:36 2018

@author: Kritin
"""

from sklearn.linear_model import Ridge
from matplotlib.pyplot as plt

RidgeScores = []

datapath = '/Users/Kritin/Desktop/School_Stuff/Research/EEGMachineLearning/SSVEP_CNN/' 
SSVEP_l_data = io.loadmat(datapath+'SSVEP_l.mat')
SVEP_l = SSVEP_l_data['SSVEP_l']

xdata =  
ydata = 

#make an array of alpha coefficients 

alphas = []
#use a different alpha coefficient 
for a in alphas:
    ridge = Ridge(alpha = a)
    ridge.fit(xdata, ydata)
    RidgeScores.append(ridge.score(xdata,ydata))

#print the ridge score coefficients
print(RidgeScores)

#plot the alphas in relation to the ridge scores
ax = plt.gca()
ax.plot(alphas, RidgeScores)
plt.xlabel('Alpha Coefficients')
plt.ylabel('RidgeScore R^2 Coefficient')
plt.show()

    
    
    