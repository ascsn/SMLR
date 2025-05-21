#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 20 23:13:46 2025

@author: anteravlic
"""
import numpy as np
import matplotlib.pyplot as plt
import helper
from numpy.polynomial.polynomial import Polynomial

colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

num_par = [10,20,30]

for n in range(len(num_par)):
    
    data = np.loadtxt('CAT_Emulator1_'+str(n)+'.txt')
    
    
    plt.figure(3)
    rel =  data[:,1]
    plt.scatter(data[:,0]*1000, rel, label = 'n(Alg 1) = '+str(num_par[n]), alpha = 0.8, color = colors[n])
    print(n, np.mean(rel))
    
 
plt.axhline(np.mean(rel), color = 'black')
num_par = [6,8,10]


for n in range(len(num_par)):
    
    data = np.loadtxt('CAT_Emulator2_'+str(n)+'.txt')
    
   
    
    plt.figure(3)
    rel =  data[:,1]
    plt.scatter(data[:,0]*1000, rel, label = 'n(Alg 2) = '+str(num_par[n]), alpha = 0.8, marker ='x', color = colors[n])
    print(n, np.mean(rel))
    

    
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