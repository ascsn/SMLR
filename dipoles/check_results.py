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
import helper_mod
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker



n = 10
# fold=1.8

params = np.loadtxt('params_'+str(n)+'.txt')

train_set = []
with open("train_set.txt", "r") as f:
    for line in f:
        tup = tuple(map(str, line.strip().split(",")))  # Convert back to tuple of integers
        train_set.append(tup)

print(train_set)



idx = 60

helper_mod.plot_Lorentzian_for_idx(idx, train_set, n, params)

emu_alphaD, alphaD_train, times = helper_mod.plot_alphaD(idx, train_set, params, n) #does not need idx as positional argument?


#sys.exit(-1)



plt.figure(2)
plt.plot([i for i in range(len(train_set))], emu_alphaD, marker = '.', label = 'emulator', color = 'red', alpha = 0.8)  
plt.plot([i for i in range(len(train_set))], alphaD_train, marker = '.', label = 'FAM QRPA calc', ls = '--', color = 'blue', alpha = 0.8)  
#plt.legend(frameon = False, ncol = 2)
plt.xlabel('Test set index', size = 18)
plt.ylabel(r'$\alpha_D$ (fm$^3$)', size = 18)
plt.gca().tick_params(axis="y",direction="in", which = 'both', labelsize = 12)
plt.gca().tick_params(axis="x",direction="in", which = 'both', labelsize = 12)
plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(5))
plt.annotate('$n = $'+str(n), (0.1, 0.9), xycoords='axes fraction', size = 18)
#plt.savefig('dipole_polarizability_emulator.pdf', bbox_inches='tight')




plt.figure(3)
rel =  np.abs(np.array(alphaD_train) - np.array(emu_alphaD)) / np.array(alphaD_train)
plt.scatter(emu_alphaD, alphaD_train, marker = 'o', label = 'QRPA calc', color = 'blue', alpha = 0.8) 
x = np.linspace(15.8, 19, 100)
plt.plot(x, x, color = 'black')
plt.xlabel(r'FAM QRPA $\alpha_D$', size = 16)
plt.ylabel(r'Emulated $\alpha_D$', size = 16)
plt.gca().tick_params(axis="y",direction="in", which = 'both', labelsize = 12)
plt.gca().tick_params(axis="x",direction="in", which = 'both', labelsize = 12)
plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
plt.title('n = '+str(n))
plt.xlim(15.8, 19)
plt.ylim(15.8, 19)

#plt.axhline(np.mean(rel), marker = '.', label = 'QRPA calc', ls = '--', color = 'black') 
#plt.axhline(np.std(rel)+np.mean(rel), marker = '.', label = 'QRPA calc', ls = '--', color = 'black') 
#plt.yscale('log')

# plt.xlabel(r'FAM QRPA $\alpha_D$', size = 16)
# plt.ylabel(r'Emulated $\alpha_D$', size = 16)
#plt.savefig('dipole_polarizability_reconstruction.pdf', bbox_inches='tight')



