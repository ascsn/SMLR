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
import matplotlib.ticker as ticker

A = 74
Z = 28
g_A = 1.2


'''
Construction of phase space integrals
'''

poly = helper.fit_phase_space(0, Z, A, 15)
coeffs = Polynomial(poly).coef

n = 30

params = np.loadtxt('params_'+str(n)+'.txt')

test_set = []
with open("test_set.txt", "r") as f:
    for line in f:
        tup = tuple(map(str, line.strip().split(",")))  # Convert back to tuple of integers
        test_set.append(tup)

print(test_set)

idx = 40

helper.plot_Lorentzian_for_idx(idx, test_set,n,params, coeffs, g_A)

HLs, HLs_test, times = helper.plot_half_lives(test_set,params,n, coeffs, g_A)

plt.figure(2)
# plt.plot([i for i in range(len(test_set))], HLs, marker = '.', label = 'emulator')  
# plt.plot([i for i in range(len(test_set))], HLs_test, marker = '.', label = 'QRPA calc', ls = '--')  
# plt.yscale('log')
# plt.legend()
# plt.ylabel('$T_{1/2}$ [s]', size = 16)
# plt.xlabel('Test set index', size = 16)
# plt.title('${}^{74}$Ni, n = '+str(n), size = 16)
# plt.title('n = '+str(n))
plt.plot([i for i in range(len(test_set))], HLs, marker = '.', label = 'emulator', color = 'red', alpha = 0.8)  
plt.plot([i for i in range(len(test_set))], HLs_test, marker = '.', label = 'FAM QRPA calc', ls = '--', color = 'blue', alpha = 0.8)  
#plt.legend(frameon = False, ncol = 2)
plt.ylabel(r'$T_{1/2}$ (s)', size = 18)
plt.xlabel('Test set index', size = 18)
#plt.title('${}^{180}$Yb, $n$ = '+str(n), size = 18)
plt.gca().tick_params(axis="y",direction="in", which = 'both', labelsize = 12)
plt.gca().tick_params(axis="x",direction="in", which = 'both', labelsize = 12)
#plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(5))
plt.yscale('log')
plt.annotate('$n = $'+str(n), (0.1,0.1), xycoords='axes fraction', size = 18)
plt.savefig('half_lives_emulator.pdf', bbox_inches='tight')

plt.figure(3)
rel =  np.abs(np.array(HLs_test)-np.array(HLs))/np.array(HLs_test)
plt.plot([i for i in range(len(test_set))],rel, marker = '.', label = 'QRPA calc', ls = '--') 
plt.axhline(np.mean(rel), marker = '.', label = 'QRPA calc', ls = '--', color = 'black') 
#plt.axhline(np.std(rel)+np.mean(rel), marker = '.', label = 'QRPA calc', ls = '--', color = 'black') 
plt.yscale('log')
plt.title('n = '+str(n))


plt.xlabel('Test set index number', size = 16)
plt.ylabel('Relative error', size = 16)