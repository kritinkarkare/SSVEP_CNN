# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 14:17:29 2018

@author: Kritin
"""
from UI_experiment import UIExperiment
import numpy as np    


def main():
    
    
    scores_epoch = []
    score = []
    #scores_batch_size = []
    validation_split = 0.25
    #just changing epochs
    epochs = 5
    batch_size = 10
    for i in range(0, 10):
        
        print("The number of epochs is " + str(epochs))
        score = UIExperiment(batch_size, epochs, validation_split)
        scores_epoch = np.concatenate((scores_epoch, score))
        epochs = epochs + 5
    
    '''
    #changing batch size in increments of 10    
    epochs = 5
    batch_size = 10
    for x in range(0, 10):
        
        print("The batch size is " + str(batch_size))
        UIExperiment(batch_size, epochs, validation_split)    
        batch_size = batch_size *2 
        
    '''
    return scores_epoch
        
if __name__ == "__main__":
    main()
    