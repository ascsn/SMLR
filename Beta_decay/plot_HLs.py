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
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


import helper
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial
import random as rn
import os
import re


'''
Main data regarding the decay
'''
A = 80
Z = 28
g_A = 1.2
nucnam='Ni_80'


'''
Construction of phase space integrals
'''

poly = helper.fit_phase_space(0, Z, A, 15)
coeffs = Polynomial(poly).coef

# Example coefficients: f(x) = 2 - x + 0*x^2 + 3*x^3
coefficients = tf.constant(coeffs[::-1], dtype=tf.float64)

# Range of x values for plotting
x_values = np.linspace(0.611, 15,100)
x_tensor = tf.constant(x_values, dtype=tf.float64)

# Evaluate the polynomial at all x values
y_values = helper.evaluate_polynomial_tf(coefficients, x_tensor).numpy()



# Plotting
x = np.linspace(0.611, 15,100)
plt.figure(figsize=(8, 5))
plt.plot(x_values, y_values)
plt.xlabel('x', fontsize=12)
plt.ylabel('f(x)', fontsize=12)
plt.title('Polynomial Plot', fontsize=14)
plt.axhline(0, color='gray', lw=0.8)
plt.axvline(0, color='gray', lw=0.8)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.yscale('log')

for i in range(len(x)):
    if (i % 10 == 0):
        plt.scatter(x[i], helper.phase_factor(0, Z, A, x[i]/helper.emass), color = 'red')


plt.show()


'''
Construct the data set
'''
rn.seed(20)

# split the data into test set and so on
#alpha_values = np.linspace(0,1.5,16)
#formatted_alpha_values = [f"{num:.3f}" for num in alpha_values]
#beta_values = np.linspace(0,1.,11)
#formatted_beta_values = [f"{num:.3f}" for num in beta_values]
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
        
        #if ((float(beta_val) <= 0.8 and float(beta_val) >= 0.2) and (float(alpha_val) <= 1.3 and float(alpha_val) >= 0.2)):
            #print(alpha_val, beta_val)
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

'''
Putting original data in the right format
'''

Lors, HLs = helper.data_table(combined, coeffs, g_A, nucnam)

x_orig = []
y_orig = []
z_orig = []
for i in range(len(combined)):
    x_orig.append(float(combined[i][0]))
    y_orig.append(float(combined[i][1]))
    z_orig.append(HLs[i])
    
x_orig = np.array(x_orig).reshape(len(x_orig),1)
y_orig = np.array(y_orig).reshape(len(y_orig),1)
z_orig = np.array(z_orig).reshape(len(z_orig),1)


original = np.concatenate((x_orig, y_orig, z_orig), axis = 1)




'''
Emulator 2
'''
n = 5

params = np.loadtxt('params_'+str(n)+'_only_HL.txt')
D_mod, S1_mod, S2_mod = helper.modified_DS_only_HL(params, n)


hls_em2 = []


for idx, alpha_p in enumerate(combined):

    M_true = D_mod + (float(alpha_p[0]) - float(central_point[0])) * S1_mod \
                   + (float(alpha_p[1]) - float(central_point[1])) * S2_mod
    
    

    eigenvalues, eigenvectors = tf.linalg.eigh(M_true)


    
    ''' Add half-lives to optimization as well'''
    hls = 0.0
    # for i in range(n):
    #     hls += eigenvalues[i]**2
    hls = eigenvalues[int(n/2)]
        
    hls_em2.append(10**hls)

x_em2 = []
y_em2 = []
z_em2 = []

for i in range(len(combined)):
    x_em2.append(float(combined[i][0]))
    y_em2.append(float(combined[i][1]))
    z_em2.append(hls_em2[i])
    
x_em2 = np.array(x_em2).reshape(len(x_em2),1)
y_em2 = np.array(y_em2).reshape(len(y_em2),1)
z_em2 = np.array(z_em2).reshape(len(z_em2),1)


emulator2 = np.concatenate((x_em2, y_em2, z_em2), axis = 1)


    
'''
Emulator 1
'''

n = 12
params = np.loadtxt('params_'+str(n)+'.txt')
hls_em1 = []
for idx in range(len(combined)):

    

    alpha_tensor = tf.constant(float(combined[idx][0]), dtype=tf.float64)  # (batch,)
    beta_tensor  = tf.constant(float(combined[idx][1]), dtype=tf.float64)

    
    D_mod, S1_mod, S2_mod, v0_mod, eta, x1, x2, x3 = helper.modified_DS(params, n)
    
    M_true = D_mod + (alpha_tensor-float(central_point[0])) * S1_mod \
        + (beta_tensor - float(central_point[1])) * S2_mod
    

    eigenvalues, eigenvectors = tf.linalg.eigh(M_true)
    
    projections = tf.linalg.matvec(tf.transpose(eigenvectors), v0_mod)
    
    # Square each projection
    B = tf.square(projections)
    
    mask = tf.cast((eigenvalues > -10) & (eigenvalues < 15), dtype=tf.float64)

    # Apply the mask to zero out B where eigenvalue is negative
    B = B * mask
    

    #B = [tf.square(tf.tensordot(eigenvectors[:, i], v0_mod, axes=1)) for i in range(eigenvectors.shape[1])]
    Lor_true = tf.constant(Lors[idx][:,1], dtype=tf.float64)

    #Generate the x values
    x = tf.constant(Lors[idx][:,0], dtype=tf.float64)
    
    width = tf.sqrt(tf.square(eta) + tf.square(x1 + x2*float(alpha[0]) + x3*float(alpha[1])))
    

    # Use tf.map_fn to apply the give_me_Lorentzian function over the x values
    Lor = helper.give_me_Lorentzian(x, eigenvalues, B, width)
    
    

    hls = helper.half_life_loss(eigenvalues, B, coeffs, g_A)
    hls_em1.append(hls)
    

hls_em1 = np.array(hls_em1)

x_em1 = []
y_em1 = []
z_em1 = []

for i in range(len(combined)):
    x_em1.append(float(combined[i][0]))
    y_em1.append(float(combined[i][1]))
    z_em1.append(hls_em1[i])
    
x_em1 = np.array(x_em1).reshape(len(x_em1),1)
y_em1 = np.array(y_em1).reshape(len(y_em1),1)
z_em1 = np.array(z_em1).reshape(len(z_em1),1)


emulator1 = np.concatenate((x_em1, y_em1, z_em1), axis = 1)


'''
Creating first figure for 3 emulators
'''
plt.figure(2,dpi=200)

cmap = plt.get_cmap('Spectral')

x = [float(i) for i in alpha]
y = [float(i) for i in beta]

# Sample 5 evenly spaced colors from the colormap
n_colors = len(np.unique(x))
colors = [cmap(i / (n_colors - 1)) for i in range(n_colors)]

'''
This is the true HL data
'''
for i in range(len(np.unique(x))):
    
    data_tmp = original[original[:,0] == np.unique(x)[i]]
    data_tmp = data_tmp[data_tmp[:,1].argsort()]
    plt.plot(data_tmp[:,1], data_tmp[:,2], color = colors[i], lw = 2.5, alpha = 0.8)
    #plt.scatter(data_tmp[:,1], data_tmp[:,2],color='k')
#plt.legend(loc = 'upper right', bbox_to_anchor=(1.2, 1))
norm = Normalize(vmin=np.min(x), vmax=np.max(x))
sm = ScalarMappable(cmap='Spectral', norm=norm)
sm.set_array([])  # for compatibility

# Add colorbar
cbar = plt.colorbar(sm, label=r'$g_0$')
plt.xlabel('$V_0$', size = 18)
plt.ylabel(r'$T_{1/2}$ (s)', size = 18)
#plt.xlim(0.0,0.4)
#plt.ylim(0,15)
plt.yscale('log')

'''
Emulator 2 data
'''
unique = np.unique(emulator2[:,0])
for i in range(len(unique)):
    
    data_tmp = emulator2[emulator2[:,0] == unique[i]]
    data_tmp = data_tmp[data_tmp[:,1].argsort()]
    plt.plot(data_tmp[:,1], data_tmp[:,2], color = 'magenta', ls = ':')
    #plt.scatter(data_tmp[:,0], data_tmp[:,2],color='k')



'''
Emulator 1 data
'''
unique = np.unique(emulator1[:,0])
for i in range(len(unique)):
    
    data_tmp = emulator1[emulator1[:,0] == unique[i]]
    data_tmp = data_tmp[data_tmp[:,1].argsort()]
    plt.plot(data_tmp[:,1], data_tmp[:,2], color = 'k', ls = '--')
    #plt.scatter(data_tmp[:,0], data_tmp[:,2],color='k')
    
plt.plot([],[], color = 'k', ls = '--', label = 'Emulator 1')
plt.plot([],[], color = 'magenta', ls = ':', label = 'Emulator 2')
plt.legend(frameon = False)
#plt.ylim(12.5,23)
#plt.yscale('log')
    
'''
Creating second figure for 3 emulators
'''

plt.figure(3,dpi=200)

'''
This is the true HL data
'''
for i in range(len(np.unique(x))):
    
    data_tmp = original[original[:,1] == np.unique(x)[i]]
    data_tmp = data_tmp[data_tmp[:,0].argsort()]
    plt.plot(data_tmp[:,0], data_tmp[:,2], color = colors[i], lw = 2.5, alpha = 0.8)
    #plt.scatter(data_tmp[:,1], data_tmp[:,2],color='k')
#plt.legend(loc = 'upper right', bbox_to_anchor=(1.2, 1))
norm = Normalize(vmin=np.min(x), vmax=np.max(x))
sm = ScalarMappable(cmap='Spectral', norm=norm)
sm.set_array([])  # for compatibility

# Add colorbar
cbar = plt.colorbar(sm, label=r'$V_0$')
plt.xlabel('$g_0$', size = 18)
plt.ylabel(r'$T_{1/2}$ (s)', size = 18)

'''
Emulator 2 data
'''
unique = np.unique(emulator2[:,1])
for i in range(len(unique)):
    
    data_tmp = emulator2[emulator2[:,1] == unique[i]]
    data_tmp = data_tmp[data_tmp[:,0].argsort()]
    plt.plot(data_tmp[:,0], data_tmp[:,2], color = 'magenta', ls = ':')
    #plt.scatter(data_tmp[:,0], data_tmp[:,2],color='k')



'''
Emulator 1 data
'''
unique = np.unique(emulator1[:,1])
for i in range(len(unique)):
    
    data_tmp = emulator1[emulator1[:,1] == unique[i]]
    data_tmp = data_tmp[data_tmp[:,0].argsort()]
    plt.plot(data_tmp[:,0], data_tmp[:,2], color = 'k', ls = '--')
    #plt.scatter(data_tmp[:,0], data_tmp[:,2],color='k')
    
plt.plot([],[], color = 'k', ls = '--', label = 'Emulator 1')
plt.plot([],[], color = 'magenta', ls = ':', label = 'Emulator 2')
plt.legend(frameon = False)
#plt.ylim(12.5,23)
plt.yscale('log')




