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
import matplotlib.patches as patches
from numpy.polynomial.polynomial import Polynomial

A = 80
Z = 28
g_A = 1.2
nucnam='Ni_80'




'''
Construction of phase space integrals
'''

poly = helper.fit_phase_space(0, Z, A, 15)
coeffs = Polynomial(poly).coef


n = 13
retain = 0.9

params = np.loadtxt(f'params_best_n{n}_retain{retain}.txt')


train_set = []
with open("train_set.txt", "r") as f:
    for line in f:
        tup = tuple(map(str, line.strip().split(",")))  # Convert back to tuple of integers
        train_set.append(tup)
'''
The values of parameters should be read directly from the file name
'''
strength_dir = '../beta_decay_data_'+nucnam+'/'

# Pattern for strength files: strength_beta_alpha.out
pattern = re.compile(r'lorm_'+nucnam+'_([0-9.]+)_([0-9.]+)\.out')

formatted_alpha_values = []
formatted_beta_values = []

all_points = []

for fname in os.listdir(strength_dir):
    match = pattern.match(fname)
    if match:
        beta_val = match.group(1)
        alpha_val = match.group(2)
        all_points.append((alpha_val, beta_val))
        
        formatted_alpha_values.append(alpha_val)
        formatted_beta_values.append(beta_val)

# Example lists
alpha = formatted_alpha_values
beta = formatted_beta_values

# Combine the lists into pairs
combined = []
for i in range(len(alpha)):
    combined.append((alpha[i], beta[i]))

'''
This is added to compute a central data point
'''
# Compute centroid
combined_ar = np.array(combined, dtype = float)
centroid = combined_ar.mean(axis=0)

# Compute distances from each point to centroid
distances = np.linalg.norm(combined_ar - centroid, axis=1)

# Find index of closest point
central_index = np.argmin(distances)
central_point = tuple(combined[central_index])
print('Central data point in train set:', central_point)

Lors, HLs = helper.data_table(combined, coeffs, g_A, nucnam)

opt_strength = []
HLs_opt = []


for idx in range(len(combined)):
    
    alpha = float(combined[idx][0])
    beta  = float(combined[idx][1])

    
    
    opt_D, opt_S1, opt_S2, opt_v0, fold, x1, x2, x3 = helper.modified_DS(params, n)


    
    M_true = opt_D + (float(alpha)-float(central_point[0])) * opt_S1 \
        + (float(beta) - float(central_point[1])) * opt_S2
    

    eigenvalues, eigenvectors = tf.linalg.eigh(M_true)
    
    n_i = eigenvalues.shape[0]
    k_keep = int(round(retain * n_i))         # how many eigenvalues to keep
    k_keep = max(1, min(k_keep, n_i))         # safety: clamp between 1 and n
    
    left  = (n_i - k_keep) // 2               # starting index of the centered block
    right = left + k_keep                     # ending index (exclusive)
    
    eigenvalues  = eigenvalues[left:right]
    eigenvectors = eigenvectors[:, left:right]
    
    projections = tf.linalg.matvec(tf.transpose(eigenvectors), opt_v0)
    
    # Square each projection
    B = tf.square(projections)
    
    mask = tf.cast((eigenvalues > -10) & (eigenvalues < 15), dtype=tf.float64)

    # Apply the mask to zero out B where eigenvalue is negative
    B = B * mask
    

    #B = [tf.square(tf.tensordot(eigenvectors[:, i], v0_mod, axes=1)) for i in range(eigenvectors.shape[1])]
    Lor_true = tf.constant(Lors[idx][:,1], dtype=tf.float64)

    #Generate the x values
    x = tf.constant(Lors[idx][:,0], dtype=tf.float64)
    
    width = tf.sqrt(tf.square(fold) + tf.square(x1 + x2*float(alpha) + x3*float(beta)))
    

    # Use tf.map_fn to apply the give_me_Lorentzian function over the x values
    Lor = helper.give_me_Lorentzian(x, eigenvalues, B, width)
    
    

    
    ''' Add half-lives to optimization as well'''

    hls = helper.half_life_loss(eigenvalues, B, coeffs, g_A)
    
   
    opt_strength.append(Lor)
    
    HLs_opt.append(hls)
    





plt.figure(1)
idx =  4
alpha = float(combined[idx][0])
beta  = float(combined[idx][1])



plt.plot(x, Lors[idx][:,1], ls = '--')
print(combined[idx])
plt.plot(x, opt_strength[idx], ls = '-')
#plt.ylim(0,20)



plt.figure(2, figsize = (6,4))
#plt.plot([i for i in range(len(combined))], alphaD_opt, marker = '.')
#plt.plot([i for i in range(len(combined))], [i[2] for i in alphaD], marker = '.')
Hls_opt = np.vstack(HLs_opt)
plt.scatter(HLs, HLs_opt )
# for i in range(len(alphaD_opt)):
#     plt.text(alphaD[:,2][i], alphaD_opt[i], str(i), fontsize=9, ha='right', va='bottom')
x = np.linspace(np.min(HLs_opt), np.max(HLs_opt), 100)
plt.plot(x,x, color = 'black')
plt.title('Emulator 1')
plt.xscale('log')
plt.yscale('log')

# plt.xlim(13,24)
# plt.ylim(13,24)


# plot the points
'''
Discrepancy on plot with points
'''
plt.figure(3)
x = [float(i) for i,j in combined]
y = [float(j) for i,j in combined]
c = np.abs(HLs_opt - np.array(HLs))/np.array(HLs)
plt.scatter(x, y, c = c, marker='s', cmap = 'Spectral', norm = LogNorm())
plt.colorbar(label = 'Relative error')

# Add index numbers
for idx, (xi, yi) in enumerate(zip(x, y)):
    plt.text(xi, yi, str(idx), ha='center', va='center', fontsize=8, color='black')

# Convert train_set to float arrays for min/max
train_alpha = [float(a) for (a, b) in train_set]
train_beta  = [float(b) for (a, b) in train_set]

alpha_min = min(train_alpha)
alpha_max = max(train_alpha)
beta_min  = min(train_beta)
beta_max  = max(train_beta)

# Add rectangle to ax_grid
train_rect = patches.Rectangle(
    (alpha_min, beta_min),
    alpha_max - alpha_min,
    beta_max - beta_min,
    linewidth=1.5,
    edgecolor='black',
    facecolor='none'
)
plt.gca().add_patch(train_rect)


# '''
# Plot L2 norm of strength rather than alphaD
# '''
# plt.figure(3)
# x = [float(i) for i,j in combined]
# y = [float(j) for i,j in combined]
# #c = np.abs(alphaD_opt - alphaD[:,2])/alphaD[:,2]
# # L2 (Euclidean) norm of the difference for each pair
# l2_norms = [np.linalg.norm(a - b)/np.linalg.norm(b)  for a, b in zip(opt_strength, orig_strength)]
# plt.scatter(x, y, c = l2_norms, marker='s', cmap = 'Spectral')
# plt.colorbar(label = 'L2-norm Relative error')

# # Add index numbers
# for idx, (xi, yi) in enumerate(zip(x, y)):
#     plt.text(xi, yi, str(idx), ha='center', va='center', fontsize=8, color='black')

# # Convert train_set to float arrays for min/max
# train_alpha = [float(a) for (a, b) in test_set]
# train_beta  = [float(b) for (a, b) in test_set]

# alpha_min = min(train_alpha)
# alpha_max = max(train_alpha)
# beta_min  = min(train_beta)
# beta_max  = max(train_beta)

# # Add rectangle to ax_grid
# train_rect = patches.Rectangle(
#     (alpha_min, beta_min),
#     alpha_max - alpha_min,
#     beta_max - beta_min,
#     linewidth=1.5,
#     edgecolor='black',
#     facecolor='none'
# )
# plt.gca().add_patch(train_rect)

# # '''
# # Print the cost function from training set
# # '''
# # strength, alphaD = helper.data_table(test_set)



# # alphaD = np.vstack(alphaD)

# # '''
# # Put alphaD in a list
# # '''
# # alphaD_list = [float(a[2]) for a in alphaD]
    

# # alphaD_opt = np.array(alphaD_opt)
# # alphaD_list = [float(a[2]) for a in alphaD]
# # weight = 100
# # strength, alphaD = helper.data_table(test_set)
# # cost, Lor, Lor_true,x, alphaD_train = helper.cost_function_batched_mixed(params, n, test_set, strength, alphaD_list, weight, central_point)
# # print('Cost on train set: ', cost)



