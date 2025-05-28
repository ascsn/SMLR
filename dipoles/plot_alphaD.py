#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 26 12:05:38 2025

@author: anteravlic
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import helper
from scipy.interpolate import griddata
import tensorflow as tf


# split the data into test set and so on
# alpha_values = np.linspace(0.400,0.850,10)
# formatted_alpha_values = [f"{num:.4f}" for num in alpha_values]

# beta_values = np.linspace(1.4,2.0,13)
# formatted_beta_values = [f"{num:.4f}" for num in beta_values]
'''
The values of parameters should be read directly from the file name
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
        
        if ((float(beta_val) <= 4.0 and float(beta_val) >= 1.5) and (float(alpha_val) <= 1.8 and float(alpha_val) >= 0.8)):
        #if ((float(alpha_val) >= 0.5)):
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


#

strength, alphaD = helper.data_table(combined)


alphaD = np.vstack(alphaD)

x = alphaD[:, 0]
y = alphaD[:, 1]
z = alphaD[:, 2]

# Create grid to interpolate onto
xi = np.linspace(x.min(), x.max(), 500)
yi = np.linspace(y.min(), y.max(), 500)
X, Y = np.meshgrid(xi, yi)

# Interpolate z values on the grid
Z = griddata((x, y), z, (X, Y), method='cubic')  # use 'linear' or 'nearest' if cubic fails




# Plot
plt.figure(figsize=(7, 6))
pcm = plt.pcolormesh(X, Y, Z, shading='auto', cmap='Spectral')
plt.colorbar(pcm, label=r'$\alpha_D$ (fm$^{-3}$)')
plt.xlabel('$b_{TV}$', size = 18)
plt.ylabel('$d_{TV}$', size = 18)

# Contour lines
contours = plt.contour(X, Y, Z, levels=10, colors='k', linewidths=0.8)
plt.clabel(contours, inline=True, fontsize=8)


plt.tight_layout()
plt.show()


plt.figure(2,dpi=200)

cmap = plt.get_cmap('viridis')

# Sample 5 evenly spaced colors from the colormap
n_colors = len(np.unique(x))
colors = [cmap(i / (n_colors - 1)) for i in range(n_colors)]
print(np.unique(x))

for i in range(len(np.unique(x))):
    
    data_tmp = alphaD[alphaD[:,0] == np.unique(x)[i]]
    data_tmp = data_tmp[data_tmp[:,1].argsort()]
    plt.plot(data_tmp[:,1], data_tmp[:,2], label = str(np.unique(x)[i]), color = colors[i])
    plt.scatter(data_tmp[:,1], data_tmp[:,2],color='k')
plt.legend(loc = 'upper right', bbox_to_anchor=(1.2, 1))

'''
Emulator figures
'''
n = 3
params = np.loadtxt('params_'+str(n)+'_only_alphaD.txt')
alphaD, alphaD_test, times = helper.plot_alphaD_simple(combined,params,n)
    

alphaD_opt = np.array(alphaD)
alphaD_orig = np.array(alphaD_test)

x_em = []
y_em = []
z_em = []

for i in range(len(combined)):
    x_em.append(float(combined[i][0]))
    y_em.append(float(combined[i][1]))
    z_em.append(alphaD[i])
    
x_em = np.array(x_em).reshape(len(x_em),1)
y_em = np.array(y_em).reshape(len(y_em),1)
z_em = np.array(z_em).reshape(len(z_em),1)


emulator = np.concatenate((x_em, y_em, z_em), axis = 1)

unique = np.unique(emulator[:,1])
for i in range(len(unique)):
    
    data_tmp = emulator[emulator[:,1] == unique[i]]
    data_tmp = data_tmp[data_tmp[:,0].argsort()]
    plt.plot(data_tmp[:,0], data_tmp[:,2], label = str(x[i]), color = 'red', ls = '--')
    #plt.scatter(data_tmp[:,0], data_tmp[:,2],color='k')
    
'''
Emulator 1
'''

n = 50
params = np.loadtxt('params_'+str(n)+'.txt')
alphaD_em1 = []
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
    
    alphaD_em1.append(helper.calculate_alphaD(opt_eigenvalues, B))
    
    #plt.plot(x, opt_Lor, ls = '--')
    print(combined[idx])
    #plt.plot(x, orig, ls = '-')

alphaD_em1 = np.array(alphaD_em1)

x_em1 = []
y_em1 = []
z_em1 = []

for i in range(len(combined)):
    x_em1.append(float(combined[i][0]))
    y_em1.append(float(combined[i][1]))
    z_em1.append(alphaD_em1[i])
    
x_em1 = np.array(x_em1).reshape(len(x_em),1)
y_em1 = np.array(y_em1).reshape(len(y_em),1)
z_em1 = np.array(z_em1).reshape(len(z_em),1)


emulator1 = np.concatenate((x_em1, y_em1, z_em1), axis = 1)

unique = np.unique(emulator1[:,1])
for i in range(len(unique)):
    
    data_tmp = emulator1[emulator1[:,1] == unique[i]]
    data_tmp = data_tmp[data_tmp[:,0].argsort()]
    plt.plot(data_tmp[:,0], data_tmp[:,2], label = str(x[i]), color = 'k', ls = '--')
    #plt.scatter(data_tmp[:,0], data_tmp[:,2],color='k')
    
    


