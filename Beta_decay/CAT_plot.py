#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 14:13:38 2025

@author: anteravlic
"""

import numpy as np
import matplotlib.pyplot as plt
import helper
from numpy.polynomial.polynomial import Polynomial

colors = ['#1f77b4', '#ff7f0e', '#2ca02c']


A = 74
Z = 28
g_A = 1.2

poly = helper.fit_phase_space(0, Z, A, 15)
coeffs = Polynomial(poly).coef

num_par = [10,20,30]
fold=0.5


for n in range(len(num_par)):
    params = np.loadtxt('params_'+str(num_par[n])+'.txt')
    
    test_set = []
    with open("test_set.txt", "r") as f:
        for line in f:
            tup = tuple(map(str, line.strip().split(",")))  # Convert back to tuple of integers
            test_set.append(tup)
    
    #print(test_set)
    
    
    HLs, HLs_test, times = helper.plot_half_lives(test_set,params,num_par[n], coeffs, g_A)
    
    
    plt.figure(3)
    rel =  np.abs(np.array(HLs_test)-np.array(HLs))/np.array(HLs_test)
    plt.scatter(np.array(times)*1000, rel, label = 'n(Alg 1) = '+str(num_par[n]), alpha = 0.8, color = colors[n])
    print(n, np.mean(rel))
    
    dat = np.concatenate((np.array(times).reshape(len(times),1), rel.reshape(len(times),1)), axis=1)
    np.savetxt('CAT_Emulator1_'+str(n)+'.txt', dat)
 
plt.axhline(np.mean(rel), color = 'black')
num_par = [6,8,10]


for n in range(len(num_par)):
    params = np.loadtxt('params_'+str(num_par[n])+'_only_HL.txt')
    
    test_set = []
    with open("test_set.txt", "r") as f:
        for line in f:
            tup = tuple(map(str, line.strip().split(",")))  # Convert back to tuple of integers
            test_set.append(tup)
    
    #print(test_set)
    
    
    HLs, HLs_test, times = helper.plot_half_lives_only_HL(test_set,params,num_par[n], coeffs, g_A)
    
    
    plt.figure(3)
    rel =  np.abs(np.array(HLs_test)-np.array(HLs))/np.array(HLs_test)
    plt.scatter(np.array(times)*1000, rel, label = 'n(Alg 2) = '+str(num_par[n]), alpha = 0.8, marker ='x', color = colors[n])
    print(n, np.mean(rel))
    
    dat = np.concatenate((np.array(times).reshape(len(times),1), rel.reshape(len(times),1)), axis=1)
    np.savetxt('CAT_Emulator2_'+str(n)+'.txt', dat)
    
plt.yscale('log')
#plt.xscale('log')
plt.xlim(3.5e-4*1000,0.0009*1000)
plt.axhline(np.mean(rel), ls = '--', color = 'black')

plt.annotate('QRPA $\sim 10^3$ s', xy=(9.9e-4, 3e-6), xytext=(4e-4, 3e-6),
             arrowprops=dict(arrowstyle='->'), fontsize=12)
#plt.axvline(1e3)

plt.xlabel('$time$ (ms)', size = 16)
plt.ylabel('Relative error', size = 16)
plt.tight_layout()

plt.gca().tick_params(axis="y",direction="in", which = 'both', labelsize = 12)
plt.gca().tick_params(axis="x",direction="in", which = 'both', labelsize = 12)

plt.legend(ncol=2, frameon = False, fontsize = 11)
#plt.ylim(1e-5,1e1)


#plt.savefig('CAT_beta.pdf', bbox_inches='tight')


# '''
# Optional on the same CAT plot also add the simple emulator
# '''
# num_par = [6,8,10,12]


# for n in range(len(num_par)):
#     params = np.loadtxt('params_'+str(num_par[n])+'_only_alphaD.txt')
    
#     test_set = []
#     with open("test_set.txt", "r") as f:
#         for line in f:
#             tup = tuple(map(str, line.strip().split(",")))  # Convert back to tuple of integers
#             test_set.append(tup)
    
#     print(test_set)
    
    
#     alphaD, alphaD_test, times = helper.plot_alphaD_simple(test_set,params,num_par[n])
    
    
#     plt.figure(3)
#     rel =  np.abs(np.array(alphaD_test)-np.array(alphaD))/np.array(alphaD_test)
#     plt.scatter(times, rel, label = 'n = '+str(num_par[n]), alpha = 0.8, marker = 'x')
    
# plt.legend(frameon = False)

# plt.xlim(8e-5, 1e-3)
