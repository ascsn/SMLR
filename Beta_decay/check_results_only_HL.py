#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 21:47:30 2025

@author: anteravlic
"""

'''
Takes the parameters file once the training has been done and plots the final results

'''
import numpy as np
import helper
from numpy.polynomial.polynomial import Polynomial
import matplotlib.pyplot as plt
import tensorflow as tf

A = 74
Z = 28
g_A = 1.2


'''
Construction of phase space integrals
'''

poly = helper.fit_phase_space(0, Z, A, 15)
coeffs = Polynomial(poly).coef

n = 6

params = np.loadtxt('params_'+str(n)+'_only_HL.txt')

test_set = []
with open("test_set.txt", "r") as f:
    for line in f:
        tup = tuple(map(str, line.strip().split(",")))  # Convert back to tuple of integers
        test_set.append(tup)

print(test_set)



HLs_test = helper.data_table_only_HL(test_set, coeffs, g_A)



opt_D, opt_S1, opt_S2 = helper.modified_DS_only_HL(params,n)

prediction = []
for idx, alpha in enumerate(test_set):

        M_true = opt_D + float(alpha[0]) * opt_S1 + float(alpha[1]) * opt_S2
        
        

        eigenvalues, eigenvectors = tf.linalg.eigh(M_true)

        
        ''' Add half-lives to optimization as well'''

        log_hls = eigenvalues[0]
        

        prediction.append(np.e**log_hls)
        
        
        
plt.plot([i for i in range(len(test_set))], prediction, marker = '.', label = 'emulator')  
plt.plot([i for i in range(len(test_set))], HLs_test, marker = '.', label = 'QRPA calc', ls = '--')  
plt.yscale('log')
plt.legend()
plt.ylabel('$T_{1/2}$ [s]', size = 16)
plt.xlabel('Test set index', size = 16)
plt.title('n = '+str(n))

plt.figure(2)
rel =  np.abs(np.array(HLs_test)-np.array(prediction))/np.array(HLs_test)
plt.plot([i for i in range(len(test_set))],rel, marker = '.', label = 'QRPA calc', ls = '--') 
plt.axhline(np.mean(rel), marker = '.', label = 'QRPA calc', ls = '--', color = 'black') 

plt.yscale('log')

plt.title('only HL')
plt.xlabel('Test set index number', size = 16)
plt.ylabel('Relative error', size = 16)