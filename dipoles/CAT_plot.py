#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 14:13:38 2025

@author: anteravlic
"""

import numpy as np
import matplotlib.pyplot as plt
import helper

num_par = [10,20,30]
fold=1.8

colors = ['#1f77b4', '#ff7f0e', '#2ca02c']


for n in range(len(num_par)):
    params = np.loadtxt('params_'+str(num_par[n])+'.txt')
    
    test_set = []
    with open("test_set.txt", "r") as f:
        for line in f:
            tup = tuple(map(str, line.strip().split(",")))  # Convert back to tuple of integers
            test_set.append(tup)
    
    #print(test_set)
    
    
    alphaD, alphaD_test, times = helper.plot_alphaD(test_set,params,num_par[n])
    
    
    plt.figure(3)
    rel =  np.abs(np.array(alphaD_test)-np.array(alphaD))/np.array(alphaD_test)
    plt.scatter(np.array(times)*1e3, rel, label = 'n(Alg 1) = '+str(num_par[n]), alpha = 0.8, color = colors[n])
    print(n, (np.mean(rel))*100)
    
    

plt.yscale('log')
plt.axhline((np.mean(rel)), color = 'black', ls = '-')
#plt.xscale('log')
#plt.xlim(3e-4,1e-3)

plt.annotate('QRPA $\sim 10^3$ s', xy=(1.1, 5e-6), xytext=(0.8, 5e-6),
             arrowprops=dict(arrowstyle='->'), fontsize=12)
#plt.axvline(1e3)

plt.xlabel('$time$ (ms)', size = 16)
plt.ylabel('Relative error', size = 16)
plt.tight_layout()

plt.gca().tick_params(axis="y",direction="in", which = 'both', labelsize = 12)
plt.gca().tick_params(axis="x",direction="in", which = 'both', labelsize = 12)


'''
Optional on the same CAT plot also add the simple emulator
'''
num_par = [6,8,10]


for n in range(len(num_par)):
    params = np.loadtxt('params_'+str(num_par[n])+'_only_alphaD.txt')
    
    test_set = []
    with open("test_set.txt", "r") as f:
        for line in f:
            tup = tuple(map(str, line.strip().split(",")))  # Convert back to tuple of integers
            test_set.append(tup)
    
    #print(test_set)
    
    
    alphaD, alphaD_test, times = helper.plot_alphaD_simple(test_set,params,num_par[n])
    
    
    plt.figure(3)
    rel =  np.abs(np.array(alphaD_test)-np.array(alphaD))/np.array(alphaD_test)
    plt.scatter(np.array(times)*1e3, rel, label = 'n(Alg 2) = '+str(num_par[n]), alpha = 0.8, marker = 'x',
                color = colors[n])
    
    print(n, np.log10(np.mean(rel)))

plt.axhline((np.mean(rel)), color = 'black', ls = '--')
    
plt.legend(frameon = False, ncol = 2, fontsize = 11)

plt.xlim(0.3,0.9)

plt.ylim(1e-6,1e-1)
#plt.xscale('log')
#plt.savefig('CAT_dipole.pdf', bbox_inches='tight')






