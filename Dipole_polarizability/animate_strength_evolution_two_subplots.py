#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 11:38:04 2025

@author: anteravlic
"""


import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tensorflow as tf
import numpy as np
import os
import re
import helper
from matplotlib.colors import LogNorm
import matplotlib.patches as patches

# Set larger font size for all plots
plt.rcParams.update({'font.size': 16})


'''
Read in all the data
'''

n = 13
retain = 0.6

params = np.loadtxt(f'params_best_n{n}_retain{retain}.txt')
params = params.astype(np.float32)

train_set = []
with open("train_set.txt", "r") as f:
    for line in f:
        tup = tuple(map(str, line.strip().split(",")))  # Convert back to tuple of integers
        train_set.append(tup)

        
        
'''
Calculate central point on train set
'''
# Compute centroid
combined_ar = np.array(train_set, dtype = float)
centroid = combined_ar.mean(axis=0)

# Compute distances from each point to centroid
distances = np.linalg.norm(combined_ar - centroid, axis=1)

# Find index of closest point
central_index = np.argmin(distances)
central_point = tuple(train_set[central_index])
print('Central data point in train set:', central_point)

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
        #if ((alpha, beta)) in train_set:
        #if ((alpha, beta)) not in train_set and ((alpha, beta)) not in cv_set:
            #if ((float(beta_val) <= 4.0 and float(beta_val) >= 1.5) and (float(alpha_val) <= 1.8 and float(alpha_val) >= 0.4)):
        #if (float(alpha) > 0.5) :
        strength_file = os.path.join(strength_dir, fname)
        alphaD_file = os.path.join(alphaD_dir, f'alphaD_{beta}_{alpha}.out')

        if os.path.exists(alphaD_file):
            # Read data
            file_strength = np.loadtxt(strength_file)
            file_alphaD = np.loadtxt(alphaD_file)


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
    
    
x_all = [float(a) for (a, b) in combined]
y_all = [float(b) for (a, b) in combined]


'''
Import alpha_D values
'''
'''
This is added to compute a central data point
'''
Amin = min(float(a) for a, b in combined)
Amax = max(float(a) for a, b in combined)
Bmin = min(float(b) for a, b in combined)
Bmax = max(float(b) for a, b in combined)

cx = (Amin + Amax) / 2.0
cy = (Bmin + Bmax) / 2.0

central_point = min(combined, key=lambda t: (float(t[0]) - cx)**2 + (float(t[1]) - cy)**2)


print('Central data point in train set:', central_point)

alphaD_opt = []
opt_strength = []
opt_eigs = []
opt_Bs = []
orig_strength = []


for idx in range(len(combined)):
    
    alpha_tensor = tf.constant(float(combined[idx][0]), dtype=tf.float32)  # (batch,)
    beta_tensor  = tf.constant(float(combined[idx][1]), dtype=tf.float32)

    
    
    
    opt_D, opt_S1, opt_S2,opt_S3,opt_S4, opt_v0,opt_v1, opt_v2, fold, x1, x2, x3, x4 = helper.modified_DS_affine_v(params, n)
    #opt_eigenvalues, opt_eigenvectors = helper.generalized_eigen(opt_D.numpy(), opt_S1.numpy(), opt_S2.numpy(), combined[idx], central_point)
    exp1 = tf.exp( -(alpha_tensor- float(central_point[0])) * x1)

    
    M_true = opt_D + (alpha_tensor- float(central_point[0])) * opt_S1 \
                + (beta_tensor- float(central_point[1])) * opt_S2 \
                + (beta_tensor- float(central_point[1])) * exp1 * opt_S3  \
                #+ beta_tensor * exp2 * opt_S4
                
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
    
    mask = tf.cast((opt_eigenvalues > 1), dtype=tf.float32)
    
    # Apply the mask to zero out B where eigenvalue is negative
    opt_dot_products = B * mask
    
    
    
    
    
    
    x = strength[idx][:,0]
    x = x.astype(np.float32)
    orig = strength[idx][:,1]
    orig_strength.append(orig)
    
    eta_new = tf.sqrt(fold**2 + (x2 + x3*(alpha_tensor- float(central_point[0])) + x4*(beta_tensor- float(central_point[1])))**2)
        
    opt_Lor = helper.give_me_Lorentzian(x, opt_eigenvalues, opt_dot_products, eta_new)
    
    opt_Bs.append(opt_dot_products)
    opt_eigs.append(opt_eigenvalues)
    
    opt_strength.append(opt_Lor)
    
    alphaD_opt.append(helper.calculate_alphaD(opt_eigenvalues, B))
    
    #plt.plot(x, opt_Lor, ls = '--')
    #print(combined[idx])
    #plt.plot(x, orig, ls = '-')

alphaD_opt = np.array(alphaD_opt)

alphaD = np.array(alphaD)
alphaD_error = np.abs(alphaD_opt - alphaD[:, 2]) / alphaD[:, 2]

#################################################

########################################
# Sweep mode: alpha or beta

mode = 'beta'   # or 'beta'
fixed_value = '1.1000'

# Build sorted sweep list
if mode == 'alpha':
    alpha_beta_filtered = [(a, b) for (a, b) in combined if b == fixed_value]
    alpha_beta_sorted = sorted(alpha_beta_filtered, key=lambda x: float(x[0]))
elif mode == 'beta':
    alpha_beta_filtered = [(a, b) for (a, b) in combined if a == fixed_value]
    alpha_beta_sorted = sorted(alpha_beta_filtered, key=lambda x: float(x[1]))
else:
    raise ValueError(f"Unknown mode {mode}")
    
if mode == 'alpha':
    param_min = min(float(a) for (a, b) in alpha_beta_sorted)
    param_max = max(float(a) for (a, b) in alpha_beta_sorted)
else:
    param_min = min(float(b) for (a, b) in alpha_beta_sorted)
    param_max = max(float(b) for (a, b) in alpha_beta_sorted)

print(f"Sweeping mode: {mode}, fixed value: {fixed_value}, number of frames: {len(alpha_beta_sorted)}")
########################################


fig, (ax_strength, ax_grid, ax_eigs) = plt.subplots(3, 1, figsize=(8, 14), dpi=150, 
                                                    gridspec_kw={'height_ratios': [1, 1, 1]})

fig.suptitle(f"PMM dimension n = {n}", fontsize=16, y=0.96)
fig.subplots_adjust(top=0.92, hspace=0.4)

# For eigenvalues panel:
param_vals_for_eigs = []
eigenvalues_history_full = []  # full eigvals
eigenvalues_history_trunc = []  # truncated eigvals

# Prepare first line (not really needed since we clear each time)
line_eigs, = ax_eigs.plot([], [], 'bo-')

ax_eigs.set_xlabel(r"$\alpha$")
ax_eigs.set_ylabel("Eigenvalues")
ax_eigs.set_xlim(min(x_all), max(x_all))
ax_eigs.set_ylim(0, 40)  # adjust as needed!
ax_eigs.grid(True)

line1, = ax_strength.plot([], [], 'k-', label='Original')
line2, = ax_strength.plot([], [], 'r--', label='PMM-Lorentzian')
strength_xlim = (0, 40)
strength_ylim = (0, 20)

# Static scatter for test set
sc = ax_grid.scatter(x_all, y_all, c=alphaD_error, cmap='Spectral', norm=LogNorm(), marker='s')
marker_x, = ax_grid.plot([], [], 'kx', markersize=10, markeredgewidth=2)

# Grid setup
ax_grid.set_xlabel(r"$\alpha$")
ax_grid.set_ylabel(r"$\beta$")
ax_grid.set_xlim(min(x_all)-0.1, max(x_all)+0.1)
ax_grid.set_ylim(min(y_all)-0.1, max(y_all)+0.1)
fig.colorbar(sc, ax=ax_grid, label=r"Relative error on $\alpha_D$")

# Strength setup
ax_strength.set_xlim(*strength_xlim)
ax_strength.set_ylim(*strength_ylim)
ax_strength.set_xlabel("E [MeV]")
ax_strength.set_ylabel("Strength")
ax_strength.legend()

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
ax_grid.add_patch(train_rect)



def init():
    print('initializing')
    line1.set_data([], [])
    line2.set_data([], [])
    return line1, line2




def animate(i):
    alpha, beta = alpha_beta_sorted[i]
    idx = combined.index((alpha, beta))
    print(f"Frame {i}: alpha = {alpha}, beta = {beta}")
    
    alpha_tensor = tf.constant(float(alpha), dtype=tf.float32)  # (batch,)
    beta_tensor  = tf.constant(float(beta), dtype=tf.float32)

    opt_D, opt_S1, opt_S2,opt_S3,opt_S4, opt_v0,opt_v1, opt_v2, fold, x1, x2, x3, x4 = helper.modified_DS_affine_v(params, n)
    #opt_eigenvalues, opt_eigenvectors = helper.generalized_eigen(opt_D.numpy(), opt_S1.numpy(), opt_S2.numpy(), combined[idx], central_point)
    exp1 = tf.exp( -(alpha_tensor- float(central_point[0])) * x1)

    
    M_true = opt_D + (alpha_tensor- float(central_point[0])) * opt_S1 \
                + (beta_tensor- float(central_point[1])) * opt_S2 \
                + (beta_tensor- float(central_point[1])) * exp1 * opt_S3  \
                #+ beta_tensor * exp2 * opt_S4
                
    opt_eigenvalues, opt_eigenvectors = tf.linalg.eigh(M_true)
    eigvals_full = opt_eigenvalues
    
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
    
    mask = tf.cast((opt_eigenvalues > 1), dtype=tf.float32)
    
    # Apply the mask to zero out B where eigenvalue is negative
    opt_dot_products = B * mask
    
    
    
    
    
    
    x = strength[idx][:,0]
    x = x.astype(np.float32)
    orig = strength[idx][:,1]
    
    eta_new = tf.sqrt(fold**2 + (x2 + x3*(alpha_tensor- float(central_point[0])) + x4*(beta_tensor- float(central_point[1])))**2)
        
    opt_Lor = helper.give_me_Lorentzian(x, opt_eigenvalues, opt_dot_products, eta_new)

    # --- Strength plot ---
    ax_strength.cla()
    ax_strength.set_xlim(*strength_xlim)
    ax_strength.set_ylim(*strength_ylim)
    ax_strength.set_xlabel("E [MeV]")
    ax_strength.set_ylabel("Strength")
    ax_strength.set_title(fr"$\alpha = {alpha},\ \beta = {beta}$")
    ax_strength.plot(x, orig, 'k-', label='Original')
    ax_strength.plot(x, opt_Lor, 'r--', label='PMM-Lorentzian')
    ax_strength.stem(opt_eigenvalues, B, basefmt=" ", markerfmt='go', linefmt='g-')
    ax_strength.legend()

    # --- Update marker in grid ---
    marker_x.set_data([float(alpha)], [float(beta)])
    
   # --- Save current parameter and eigenvalues ---
    if mode == 'alpha':
        param_val = float(alpha)
    else:
        param_val = float(beta)
    
    param_vals_for_eigs.append(param_val)
    eigenvalues_history_full.append(eigvals_full)
    eigenvalues_history_trunc.append(opt_eigenvalues)
    
    # Convert to arrays for plotting
    param_array = np.array(param_vals_for_eigs)
    eigen_full_matrix = np.array(eigenvalues_history_full)
    eigen_trunc_matrix = np.array(eigenvalues_history_trunc)
    
    param_label = r"$\alpha$" if mode == 'alpha' else r"$\beta$"
    
    # Clear axis
    ax_eigs.cla()
    ax_eigs.set_xlabel(param_label)
    ax_eigs.set_ylabel("Eigenvalues")
    ax_eigs.set_xlim(param_min, param_max)
    ax_eigs.set_ylim(-50, 100)
    ax_eigs.grid(True)
    ax_eigs.set_title("Eigenvalue evolution")
    
    # Plot full in gray
    for k in range(eigen_full_matrix.shape[1]):
        ax_eigs.plot(param_array, eigen_full_matrix[:, k], '-', color='gray', linewidth=2, alpha=1)
    
    # Plot truncated in blue
    for k in range(eigen_trunc_matrix.shape[1]):
        ax_eigs.plot(param_array, eigen_trunc_matrix[:, k], 'o-', markersize=4)



    return []

#Make animation
ani = animation.FuncAnimation(
    fig, animate, init_func=init,
    frames=len(alpha_beta_sorted), interval=700, repeat=False  # no blit!
)

# Save MP4
ani.save(f"animate_strength_evolution_sweep_{mode}_fixed_{fixed_value}_PMM{n}.mp4", writer='ffmpeg', fps=1.3)