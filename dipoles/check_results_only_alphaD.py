#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 17:10:54 2025

@author: anteravlic
"""

import numpy as np
import helper
import matplotlib.pyplot as plt



n = 6


params = np.loadtxt('params_'+str(n)+'_only_alphaD.txt')

test_set = []
with open("train_set.txt", "r") as f:
    for line in f:
        tup = tuple(map(str, line.strip().split(",")))  # Convert back to tuple of integers
        test_set.append(tup)

print(test_set)




alphaD, alphaD_test, times = helper.plot_alphaD_simple(test_set,params,n)


plt.figure(3)
rel =  np.abs(np.array(alphaD_test)-np.array(alphaD))/np.array(alphaD_test)
plt.plot([i for i in range(len(test_set))],rel, marker = '.', label = 'QRPA calc', ls = '--') 
plt.axhline(np.mean(rel), marker = '.', label = 'QRPA calc', ls = '--', color = 'black') 
#plt.axhline(np.std(rel)+np.mean(rel), marker = '.', label = 'QRPA calc', ls = '--', color = 'black') 
plt.yscale('log')
plt.title('n = '+str(n))


plt.xlabel('Test set index number', size = 16)
plt.ylabel('Relative error', size = 16)