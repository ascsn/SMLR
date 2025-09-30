#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 10:54:06 2025

@author: anteravlic
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 11:38:04 2025

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
import matplotlib.animation as animation

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
    
    
x_all = [float(a) for (a, b) in combined]
y_all = [float(b) for (a, b) in combined]

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
    
HLs_opt = np.array(HLs_opt)
HLs = np.array([i.numpy() for i in HLs])
HLs_error = np.abs(HLs_opt - HLs) / HLs


#################################################

########################################
# Sweep mode: alpha or beta

mode = 'beta'   # or 'beta'
fixed_value = '2.000'

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
strength_xlim = (-10, 0.782)
strength_ylim = (0, 20)

# Static scatter for test set
sc = ax_grid.scatter(x_all, y_all, c=HLs_error, cmap='Spectral', norm=LogNorm(), marker='s')
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
    
    alpha_tensor = tf.constant(float(alpha), dtype=tf.float64)  # (batch,)
    beta_tensor  = tf.constant(float(beta), dtype=tf.float64)

    opt_D, opt_S1, opt_S2, opt_v0, fold, x1, x2, x3 = helper.modified_DS(params, n)


    
    M_true = opt_D + (alpha_tensor-float(central_point[0])) * opt_S1 \
        + (beta_tensor - float(central_point[1])) * opt_S2
    

    eigenvalues, eigenvectors = tf.linalg.eigh(M_true)
    eigvals_full = eigenvalues
    
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
    
       
    
    orig = Lor_true
    
    eta_new =  tf.sqrt(tf.square(fold) + tf.square(x1 + x2*alpha_tensor + x3*beta_tensor))
        
    opt_Lor = helper.give_me_Lorentzian(x, eigenvalues, B, eta_new)

    # --- Strength plot ---
    ax_strength.cla()
    ax_strength.set_xlim(*strength_xlim)
    ax_strength.set_ylim(*strength_ylim)
    ax_strength.set_xlabel("E [MeV]")
    ax_strength.set_ylabel("Strength")
    ax_strength.set_title(fr"$\alpha = {alpha},\ \beta = {beta}$")
    ax_strength.plot(x, orig, 'k-', label='Original')
    ax_strength.plot(x, opt_Lor, 'r--', label='PMM-Lorentzian')
    ax_strength.stem(eigenvalues, B, basefmt=" ", markerfmt='go', linefmt='g-')
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
    eigenvalues_history_trunc.append(eigenvalues)
    
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
    ax_eigs.set_ylim(-20, 20)
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