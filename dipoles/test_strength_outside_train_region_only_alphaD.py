#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 28 14:47:48 2025

@author: anteravlic
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 24 12:19:24 2025

@author: anteravlic
"""

import numpy as np
import helper
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import tensorflow as tf
import os
import re


n = 2

params = np.loadtxt('params_'+str(n)+'_only_alphaD.txt')

test_set = []
with open("train_set.txt", "r") as f:
    for line in f:
        tup = tuple(map(str, line.strip().split(",")))  # Convert back to tuple of integers
        test_set.append(tup)
cv_set = []
with open("cv_set.txt", "r") as f:
    for line in f:
        tup = tuple(map(str, line.strip().split(",")))  # Convert back to tuple of integers
        cv_set.append(tup)

strength = []
alphaD = []
fmt_data = []

formatted_alpha_values = []
formatted_beta_values = []

strength_dir = '../dipoles_data_all/total_strength/'
alphaD_dir = '../dipoles_data_all/total_alphaD/'

# Pattern for strength files: strength_beta_alpha.out
pattern = re.compile(r'strength_([0-9.]+)_([0-9.]+)\.out')

for fname in os.listdir(strength_dir):
    match = pattern.match(fname)
    if match:
        beta = match.group(1)
        alpha = match.group(2)
        if ((alpha, beta)) not in test_set and ((alpha, beta)) not in cv_set:
            #if ((float(beta) <= 4.0 and float(beta) >= 1.5) and (float(alpha) <= 1.8 and float(alpha) >= 0.8)):
            if (float(alpha) > 0.5) :
                    strength_file = os.path.join(strength_dir, fname)
                    alphaD_file = os.path.join(alphaD_dir, f'alphaD_{beta}_{alpha}.out')
            
                    if os.path.exists(alphaD_file):
                        # Read data
                        file_strength = np.loadtxt(strength_file)
                        file_alphaD = np.loadtxt(alphaD_file)
            
                        # Apply filter on strength
                        file_strength = file_strength[file_strength[:, 0] > 1]
            
                        # Store
                        strength.append(file_strength)
                        alphaD.append(file_alphaD)
                        fmt_data.append((alpha, beta))
                        
                        formatted_alpha_values.append(alpha)
                        formatted_beta_values.append(beta)



# Example lists
alpha = formatted_alpha_values
beta = formatted_beta_values

# Combine the lists into pairs
#combined = [(x, y) for x in alpha for y in beta]
combined = []
for i in range(len(alpha)):
    combined.append((alpha[i], beta[i]))


    
    
alphaD, alphaD_test, times = helper.plot_alphaD_simple(combined,params,n)
    

alphaD_opt = np.array(alphaD)
alphaD_orig = np.array(alphaD_test)




# plot the points
'''
Discrepancy on plot with points
'''
plt.figure(1)
x = [float(i) for i,j in combined]
y = [float(j) for i,j in combined]
c = np.abs(alphaD_opt - alphaD_orig)/alphaD_orig
plt.scatter(x, y, c = c, marker='s', cmap = 'Spectral')
plt.colorbar(label = 'Relative error')


plt.figure(2)
plt.scatter(alphaD_opt, alphaD_orig, marker = 'o')
x = np.linspace(np.min(alphaD_opt), 24, 100)
plt.plot(x,x, color = 'black')

plt.xlim(13,24)
plt.ylim(13,24)

plt.title('Emulator 2')



