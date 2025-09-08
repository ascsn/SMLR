#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 15:53:38 2025

@author: anteravlic
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import re
import helper
from scipy.interpolate import griddata
import tensorflow as tf
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.interpolate import griddata


# split the data into test set and so on
# alpha_values = np.linspace(0.400,0.850,10)
# formatted_alpha_values = [f"{num:.4f}" for num in alpha_values]

# beta_values = np.linspace(1.4,2.0,13)
# formatted_beta_values = [f"{num:.4f}" for num in beta_values]
'''
The values of parameters should be read directly from the file name
I have to read this because data outside the training region was not included
in the main.py module, so I have to read it again.
'''
strength_dir = '../dipoles_data_all/total_strength/'
alphaD_dir = '../dipoles_data_all/total_alphaD/'

# Pattern for strength files: strength_beta_alpha.out
pattern = re.compile(r'strength_([0-9.]+)_([0-9.]+)\.out')

formatted_alpha_values = []
formatted_beta_values = []

all_points = []

for fname in os.listdir(strength_dir):
    match = pattern.match(fname)
    if match:
        beta_val = match.group(1)
        alpha_val = match.group(2)
        all_points.append((alpha_val, beta_val))
        
        #if ((float(beta_val) <= 4.0 and float(beta_val) >= 1.5) and (float(alpha_val) <= 1.8 and float(alpha_val) >= 0.8)):
        #if ((float(alpha_val) <= 1.5)):
            #print(alpha_val, beta_val)
        formatted_alpha_values.append(alpha_val)
        formatted_beta_values.append(beta_val)


# Example lists
alpha = formatted_alpha_values
beta = formatted_beta_values

# Combine the lists into pairs
#combined = [(x, y) for x in alpha for y in beta]
combined = []
for i in range(len(alpha)):
    combined.append((alpha[i], beta[i]))
# Shuffle the combined list

'''
Calculate central point on train set
'''
test_set = []
with open("train_set.txt", "r") as f:
    for line in f:
        tup = tuple(map(str, line.strip().split(",")))  # Convert back to tuple of integers
        test_set.append(tup)
# cv_set = []
# with open("cv_set.txt", "r") as f:
#     for line in f:
#         tup = tuple(map(str, line.strip().split(",")))  # Convert back to tuple of integers
#         cv_set.append(tup)
        
# Compute centroid
combined_ar = np.array(test_set, dtype = float)
centroid = combined_ar.mean(axis=0)

# Compute distances from each point to centroid
distances = np.linalg.norm(combined_ar - centroid, axis=1)

# Find index of closest point
central_index = np.argmin(distances)
central_point = tuple(test_set[central_index])
print('Central data point in train set:', central_point)


#

strength, alphaD = helper.data_table(combined)


alphaD = np.vstack(alphaD)

# x = alphaD[:, 0]
# y = alphaD[:, 1]
# z = alphaD[:, 2]

# # Create grid to interpolate onto
# xi = np.linspace(x.min(), x.max(), 500)
# yi = np.linspace(y.min(), y.max(), 500)
# X, Y = np.meshgrid(xi, yi)

# # Interpolate z values on the grid
# Z = griddata((x, y), z, (X, Y), method='cubic')  # use 'linear' or 'nearest' if cubic fails




# # Plot
# plt.figure(figsize=(7, 6))
# pcm = plt.pcolormesh(X, Y, Z, shading='auto', cmap='Spectral')
# plt.colorbar(pcm, label=r'$\alpha_D$ (fm$^{-3}$)')
# plt.xlabel('$b_{TV}$', size = 18)
# plt.ylabel('$d_{TV}$', size = 18)

# # Contour lines
# contours = plt.contour(X, Y, Z, levels=10, colors='k', linewidths=0.8)
# plt.clabel(contours, inline=True, fontsize=8)


# plt.tight_layout()
# plt.show()






'''
Emulator figures
'''
n = 7
params = np.loadtxt('params_'+str(n)+'_only_alphaD.txt')
params = params.astype(np.float32)
alphaD_opt, alphaD_orig, times = helper.plot_alphaD_simple(combined,params,n, central_point)
    

alphaD_opt = np.array(alphaD_opt)
alphaD_orig = np.array(alphaD_orig)

x_em = []
y_em = []
z_em = []

for i in range(len(combined)):
    x_em.append(float(combined[i][0]))
    y_em.append(float(combined[i][1]))
    z_em.append(alphaD_opt[i])
    
x_em = np.array(x_em).reshape(len(x_em),1)
y_em = np.array(y_em).reshape(len(y_em),1)
z_em = np.array(z_em).reshape(len(z_em),1)


emulator = np.concatenate((x_em, y_em, z_em), axis = 1)


    
'''
Emulator 1
'''

n = 16
retain = 0.6
params = np.loadtxt('params_'+str(n)+'_'+str(retain)+'.txt')
params = params.astype(np.float32)
alphaD_em1 = []
for idx in range(len(combined)):

    

    alpha_tensor = tf.constant(float(combined[idx][0]), dtype=tf.float32)  # (batch,)
    beta_tensor  = tf.constant(float(combined[idx][1]), dtype=tf.float32)

    
    
    
    opt_D, opt_S1, opt_S2,opt_S3,opt_S4, opt_v0,opt_v1, opt_v2, fold, x1, x2, x3, x4 = helper.modified_DS_affine_v(params, n)
    #opt_eigenvalues, opt_eigenvectors = helper.generalized_eigen(opt_D.numpy(), opt_S1.numpy(), opt_S2.numpy(), combined[idx], central_point)
    exp1 = tf.exp( -(alpha_tensor- float(central_point[0])) * x1 )
    
    M_true = opt_D + (alpha_tensor- float(central_point[0])) * opt_S1 \
                + (beta_tensor- float(central_point[1])) * opt_S2 \
                + (beta_tensor- float(central_point[1])) * exp1 * opt_S3  \
                
    opt_eigenvalues, opt_eigenvectors = tf.linalg.eigh(M_true)
    
    n_i = opt_eigenvalues.shape[0]
    k_keep = int(round(retain * n_i))         # how many eigenvalues to keep
    k_keep = max(1, min(k_keep, n_i))         # safety: clamp between 1 and n
    
    left  = (n_i - k_keep) // 2               # starting index of the centered block
    right = left + k_keep                     # ending index (exclusive)
    
    opt_eigenvalues  = opt_eigenvalues[left:right]
    opt_eigenvectors = opt_eigenvectors[:, left:right]
    
    v_eff = opt_v0 \
          + (alpha_tensor- float(central_point[0])) * opt_v1 \
          + (beta_tensor- float(central_point[1])) * opt_v2 
              
    projections = tf.linalg.matvec(tf.transpose(opt_eigenvectors), v_eff)
    
    # Square each projection
    B = tf.square(projections)
    
    #mask = tf.cast((opt_eigenvalues > 5) &  (opt_eigenvalues < 40), dtype=tf.float32)
    
    # Apply the mask to zero out B where eigenvalue is negative
    opt_dot_products = B #* mask
    
    
    eta_new = tf.sqrt(fold**2 + (x2 + x3*(alpha_tensor- float(central_point[0])) + x4*(beta_tensor- float(central_point[1])))**2)
        
    
    alphaD_em1.append(helper.calculate_alphaD(opt_eigenvalues, B))

alphaD_em1 = np.array(alphaD_em1)

x_em1 = []
y_em1 = []
z_em1 = []

for i in range(len(combined)):
    x_em1.append(float(combined[i][0]))
    y_em1.append(float(combined[i][1]))
    z_em1.append(alphaD_em1[i])
    
x_em1 = np.array(x_em1).reshape(len(x_em1),1)
y_em1 = np.array(y_em1).reshape(len(y_em1),1)
z_em1 = np.array(z_em1).reshape(len(z_em1),1)


emulator1 = np.concatenate((x_em1, y_em1, z_em1), axis = 1)

# emulator1: columns [x, y, z] = [b_TV, d_TV, alpha_D]
x = alphaD[:, 1].astype(float)   # b_TV  -> x-axis
y = alphaD[:, 0].astype(float)   # d_TV  -> y-axis
z = alphaD[:, 2].astype(float)   # α_D   -> value

# 1) Build a regular grid (adjust resolution as needed)
nx, ny = 100, 100
xi = np.linspace(x.min(), x.max(), nx)
yi = np.linspace(y.min(), y.max(), ny)
X, Y = np.meshgrid(xi, yi)

# 2) Interpolate scattered data onto the grid
Z_lin = griddata((x, y), z, (X, Y), method='linear')

# 3) Optional: fill NaNs (outside convex hull / sparse areas) with 'nearest'
Z_near = griddata((x, y), z, (X, Y), method='nearest')
Z = np.where(np.isnan(Z_lin), Z_near, Z_lin)

# 4) Plot
plt.figure(figsize=(6.5, 5), dpi=200)
im = plt.imshow(
    Z, origin='lower', aspect='auto', cmap='Spectral',
    extent=[xi.min(), xi.max(), yi.min(), yi.max()], vmin = 13, vmax= 23
)
plt.xlabel(r'$b_{TV}$ (fm$^{2}$)', fontsize=12)
plt.ylabel(r'$d_{TV}$', fontsize=12)
plt.title(r'FOM interpolated', fontsize=12)
cbar = plt.colorbar(im)
cbar.set_label(r'$\alpha_D$ (fm$^{3}$)')

# Optional: add contours for structure
try:
    cs = plt.contour(X, Y, Z, levels=12, colors='k', linewidths=0.5)
    plt.clabel(cs, inline=True, fontsize=8)
except Exception:
    pass


from matplotlib.patches import Rectangle

# --- Load training points (alpha,beta) if not already loaded ---
# If you already loaded them earlier as `test_set`, reuse that.
train_pts = []
with open("train_set.txt", "r") as f:
    for line in f:
        a, b = line.strip().split(",")
        train_pts.append((float(a), float(b)))
train_pts = np.array(train_pts)  # shape (N, 2), columns: [alpha, beta]

# IMPORTANT: match your plot axes ordering!
# In your emulator arrays you use:
#   x = emulator1[:, 0]
#   y = emulator1[:, 1]
# If emulator1[:,0] is alpha and emulator1[:,1] is beta, keep as-is:
x_train = train_pts[:, 0]
y_train = train_pts[:, 1]

# If instead your plot uses x=beta, y=alpha, just swap:
# x_train = train_pts[:, 1]
# y_train = train_pts[:, 0]

xmin, xmax = np.min(x_train), np.max(x_train)
ymin, ymax = np.min(y_train), np.max(y_train)

ax = plt.gca()
rect = Rectangle(
    (xmin, ymin),
    xmax - xmin,
    ymax - ymin,
    fill=False,
    edgecolor='k',
    linewidth=2.0,
    linestyle='--',
    alpha=0.9
)
ax.add_patch(rect)



plt.tight_layout()
plt.show()