#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 20 22:27:30 2025

@author: anteravlic
"""

import numpy as np
import helper
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

'''
Test performance on the train set
'''



n = 40

params = np.loadtxt('params_'+str(n)+'.txt')

test_set = []
with open("test_set.txt", "r") as f:
    for line in f:
        tup = tuple(map(str, line.strip().split(",")))  # Convert back to tuple of integers
        test_set.append(tup)

print(test_set)

'''
I chose by hand test set indices 17, 40 and 70 so that they are sufficiently different
'''
idxs = [4,1,2]
x1, orig1, opt1 = helper.data_Lorentzian_for_idx(idxs[0], test_set,n,params)
x2, orig2, opt2 = helper.data_Lorentzian_for_idx(idxs[1], test_set,n,params)
x3, orig3, opt3 = helper.data_Lorentzian_for_idx(idxs[2], test_set,n,params)
plt.plot(x1, orig1, color = 'red', label ='$b_{TV} ='+test_set[idxs[0]][1][:4]+'$ fm${}^{-2}$, $d_{TV}='+test_set[idxs[0]][0][:4]+'$')
plt.plot(x2, orig2, color = 'blue', label ='$b_{TV} ='+test_set[idxs[1]][1][:4]+'$ fm${}^{-2}$, $d_{TV}='+test_set[idxs[1]][0][:4]+'$')
plt.plot(x3, orig3, color = 'green', label ='$b_{TV} ='+test_set[idxs[2]][1][:4]+'$ fm${}^{-2}$, $d_{TV}='+test_set[idxs[2]][0][:4]+'$')
plt.plot(x1, opt1, color = 'red', ls = ':')
plt.plot(x2, opt2, color = 'blue', ls = ':')
plt.plot(x3, opt3, color = 'green', ls = ':')

plt.xlabel('$\omega$ (MeV)', size = 18)
plt.ylabel('$S$ (e$^2$ fm$^2$/MeV)', size = 18)

plt.annotate('${}^{180}$Yb', (0.7,0.5), xycoords='axes fraction', size = 18)

plt.gca().tick_params(axis="y",direction="in", which = 'both', labelsize = 12)
plt.gca().tick_params(axis="x",direction="in", which = 'both', labelsize = 12)

plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(0.5))

plt.legend()

plt.xlim(5,30)
plt.ylim(0)