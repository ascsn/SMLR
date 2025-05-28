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
from matplotlib.colors import LogNorm


n = 40

params = np.loadtxt('params_'+str(n)+'.txt')

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
            #if ((float(beta_val) <= 4.0 and float(beta_val) >= 1.5) and (float(alpha_val) <= 1.8 and float(alpha_val) >= 0.4)):
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




alphaD_opt = []

for idx in range(len(combined)):

    
    
    
    opt_D, opt_S1, opt_S2, opt_v0, fold = helper.modified_DS(params, n)
    opt_eigenvalues, opt_eigenvectors = helper.generalized_eigen(opt_D.numpy(), opt_S1.numpy(), opt_S2.numpy(), combined[idx])
    
    projections = tf.linalg.matvec(tf.transpose(opt_eigenvectors), opt_v0)
    
    # Square each projection
    B = tf.square(projections)
    
    mask = tf.cast((opt_eigenvalues > 1) &  (opt_eigenvalues < 40), dtype=tf.float64)
    
    # Apply the mask to zero out B where eigenvalue is negative
    opt_dot_products = B #* mask
    
    
    
    x = strength[idx][:,0]
    orig = strength[idx][:,1]
        
    opt_Lor = helper.give_me_Lorentzian(x, opt_eigenvalues, opt_dot_products, fold)
    
    alphaD_opt.append(helper.calculate_alphaD(opt_eigenvalues, B))
    
    #plt.plot(x, opt_Lor, ls = '--')
    print(combined[idx])
    #plt.plot(x, orig, ls = '-')

alphaD_opt = np.array(alphaD_opt)

plt.figure(2)
#plt.plot([i for i in range(len(combined))], alphaD_opt, marker = '.')
#plt.plot([i for i in range(len(combined))], [i[2] for i in alphaD], marker = '.')
alphaD = np.vstack(alphaD)
plt.scatter(alphaD[:,2], alphaD_opt )
x = np.linspace(np.min(alphaD_opt), 24, 100)
plt.plot(x,x, color = 'black')
plt.title('Emulator 1')

plt.xlim(13,24)
plt.ylim(13,24)


plt.figure(1)
idx = 51
opt_D, opt_S1, opt_S2, opt_v0, fold = helper.modified_DS(params, n)
opt_eigenvalues, opt_eigenvectors = helper.generalized_eigen(opt_D.numpy(), opt_S1.numpy(), opt_S2.numpy(), combined[idx])

projections = tf.linalg.matvec(tf.transpose(opt_eigenvectors), opt_v0)

# Square each projection
B = tf.square(projections)

mask = tf.cast((opt_eigenvalues > 3) &  (opt_eigenvalues < 40), dtype=tf.float64)

# Apply the mask to zero out B where eigenvalue is negative
opt_dot_products = B #* mask



x = strength[idx][:,0]
orig = strength[idx][:,1]
    
opt_Lor = helper.give_me_Lorentzian(x, opt_eigenvalues, opt_dot_products, fold)


plt.plot(x, opt_Lor, ls = '--')
print(combined[idx])
plt.plot(x, orig, ls = '-')

# plot the points
'''
Discrepancy on plot with points
'''
plt.figure(3)
x = [float(i) for i,j in combined]
y = [float(j) for i,j in combined]
c = np.abs(alphaD_opt - alphaD[:,2])/alphaD[:,2]
plt.scatter(x, y, c = c, marker='s', cmap = 'Spectral', norm = LogNorm(vmin=1e-3,vmax=1e-1))
plt.colorbar(label = 'Relative error')



