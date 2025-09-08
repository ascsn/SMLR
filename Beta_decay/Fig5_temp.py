#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 14:39:56 2025

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
from matplotlib.colors import LogNorm
import matplotlib.patches as patches
from numpy.polynomial.polynomial import Polynomial
import scipy.interpolate as interpolate

A = 80
Z = 28
g_A = 1.2
nucnam='Ni_80'

# Set global font size
plt.rcParams.update({'font.size': 16})
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'


fig, ax = plt.subplots(2,1 , dpi = 150, figsize = (5,10))

'''
Construction of phase space integrals
'''

poly = helper.fit_phase_space(0, Z, A, 15)
coeffs = Polynomial(poly).coef


n = 14

params = np.loadtxt('params_'+str(n)+'.txt')


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
    








# plot the points
'''
Discrepancy on plot with points
'''

x = [float(i) for i,j in combined]
y = [float(j) for i,j in combined]
c = np.abs(HLs_opt - np.array(HLs))/np.array(HLs)
im = ax[0].scatter(x, y, c = c, marker='s', cmap = 'Spectral', norm = LogNorm())
cbar_ax = fig.add_axes([0.92, 0.55, 0.02, 0.3])  # adjust these numbers as needed
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label(r"Relative error on $T_{1/2}$", fontsize=14)

# Add index numbers
# for idx, (xi, yi) in enumerate(zip(x, y)):
#     plt.text(xi, yi, str(idx), ha='center', va='center', fontsize=8, color='black')

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
ax[0].add_patch(train_rect)
ax[0].set_ylabel('$g_0$', size = 22)
ax[0].set_xlabel('$V_0^{is}$', size = 22)


'''
This here plots the Lorentzians
'''
idx =  [4, 98, 61]

for i in range(len(idx)):
    
    alpha = float(combined[idx[i]][0])
    beta  = float(combined[idx[i]][1])
    

    ax[0].scatter(alpha,beta,
            s=200, facecolors='none', edgecolors='red', linewidths=2)
    
    
    
    opt_D, opt_S1, opt_S2, opt_v0, fold, x1, x2, x3 = helper.modified_DS(params, n)
    
    
    
    M_true = opt_D + (float(alpha)-float(central_point[0])) * opt_S1 \
        + (float(beta) - float(central_point[1])) * opt_S2
    
    
    eigenvalues, eigenvectors = tf.linalg.eigh(M_true)
    
    projections = tf.linalg.matvec(tf.transpose(eigenvectors), opt_v0)
    
    # Square each projection
    B = tf.square(projections)
    
    mask = tf.cast((eigenvalues > -10) & (eigenvalues < 15), dtype=tf.float64)
    
    # Apply the mask to zero out B where eigenvalue is negative
    B = B * mask
    
    
    #B = [tf.square(tf.tensordot(eigenvectors[:, i], v0_mod, axes=1)) for i in range(eigenvectors.shape[1])]
    Lor_true = tf.constant(Lors[idx[i]][:,1], dtype=tf.float64)
    
    #Generate the x values
    x = tf.constant(Lors[idx[i]][:,0], dtype=tf.float64)
    
    width = tf.sqrt(tf.square(fold) + tf.square(x1 + x2*float(alpha) + x3*float(beta)))
    
    
    # Use tf.map_fn to apply the give_me_Lorentzian function over the x values
    Lor = helper.give_me_Lorentzian(x, eigenvalues, B, width)
    
    #ax[1].plot(x, Lor, color = 'black', ls = '--')
    f1 = interpolate.interp1d(x, Lor, kind = 'cubic')
    #ax[1].plot(x, Lor_true)
    f2 = interpolate.interp1d(x, Lor_true, kind = 'cubic')
    
    xnew = np.linspace(np.min(x), np.max(x), 1000)
    ax[1].plot(xnew, f1(xnew), color = 'black', ls = '--', lw = 3, alpha = 0.7, zorder=99)
    ax[1].plot(xnew, f2(xnew), lw = 3, label = '$g_0 = $'+str(round(beta,2))+', $V_0^{is} = $'+str(round(alpha,2)))
    
ax[1].set_ylim(0)
ax[1].set_xlim(np.min(x), np.max(x))
ax[1].legend(frameon = False, fontsize = 13)


ax[1].set_xlabel(r'$\omega$ (MeV)', size = 22)
ax[1].set_ylabel('$S$(1/MeV)', size = 22)

ax[0].annotate('(a)', (-0.25, 1.02), xycoords='axes fraction', size = 24)
ax[1].annotate('(b)', (-0.25, 1.02), xycoords='axes fraction', size = 24)
ax[1].annotate('${}^{80}$Ni', (0.2, 0.5), xycoords='axes fraction', size = 26)

plt.subplots_adjust(hspace = 0.25)

plt.savefig('../figs/fig5_beta_temp.pdf', bbox_inches='tight')




