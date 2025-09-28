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

n = 16
retain = 0.5

params = np.loadtxt(f'params_best_n{n}_retain{retain}.txt')
params = params.astype(np.float32)

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
        
        
'''
Calculate central point on train set
'''
# Compute centroid
        
'''
Calculate central point on train set
'''
# Compute centroid
alpha_float = np.array([float(a) for (a,b) in test_set])
beta_float  = np.array([float(b) for (a,b) in test_set])



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
        #if ((alpha, beta)) in test_set:
        #if ((alpha, beta)) not in test_set and ((alpha, beta)) not in cv_set:
        #if ((float(beta) <= 4.0 and float(beta) >= 1.5) and (float(alpha) <= 1.8 and float(alpha) >= 0.4)):
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


#plt.show()


plt.figure(1)
idx =  46

x = strength[idx][:,0]
plt.plot(x, opt_strength[idx])
plt.stem(opt_eigs[idx], opt_Bs[idx])
print(alphaD[idx][2],alphaD_opt[idx])
plt.ylim(0,8)

orig = strength[idx][:,1]
alpha_tensor = tf.constant(float(combined[idx][0]), dtype=tf.float32)  # (batch,)
beta_tensor  = tf.constant(float(combined[idx][1]), dtype=tf.float32)
eta_new = tf.sqrt(fold**2 + (x2 + x3*(alpha_tensor- float(central_point[0])) + x4*(beta_tensor- float(central_point[1])))**2)

print('eta: ', eta_new)

print(combined[idx])
plt.plot(x, orig, ls = '-')
plt.title('n = '+str(n)+ ' retain = '+str( retain))


'''
Calculate RMSE of all the Lorentzians
'''
rmse = 0
for idx in range(len(combined)):
    
    
    rmse += np.sum((strength[idx][:,1] - opt_strength[idx])**2)
    print(rmse)
    
print('RMSE: ', np.sqrt(rmse))
# plt.figure(9)
# x = [3,4,5,6,7,8,9,10]
# y1 = [83.21,67.77,59.73,51.76,47.31,46.95,48.79,42.42]
# y2 = [76.48,67,52.79,46.06,47.69,41.93,35.02,34.82]
# y3 = [67.86,70.86,52.62,61.46,51.24,44.67,37.96,36.51]
# y4 = [74.34,84.27,51.22,55.65,56.64,66.23,64.38,70.52]
# # y1 = [43.64,39.85,30.01,26.43,23.51,22.24,19.82,18.72]
# # y2 = [42.32,38.27,28.19,24.45,23.39,20.91,17.81,15.61]
# # y3 = [41.14,36.94,28.16,23.55,22.42,20.12,16.93,15.84]
# # y4 = [41.64,36.95,29.62,22.80,21.93,19.49,16.50,15.54]
# plt.plot(x,y1, marker = 'o', mec = 'white', label = '100% retain')
# plt.plot(x,y2, marker = 'o', mec = 'white', label = '60% retain')
# plt.plot(x,y3, marker = 'o', mec = 'white', label = '40% retain')
# plt.plot(x,y4, marker = 'o', mec = 'white', label = '20% retain')
# plt.ylabel('RMSE', size = 16)
# plt.xlabel('$n$', size = 16)
# plt.legend()
# sys.exit(-1)




plt.figure(2, figsize = (6,4))
#plt.plot([i for i in range(len(combined))], alphaD_opt, marker = '.')
#plt.plot([i for i in range(len(combined))], [i[2] for i in alphaD], marker = '.')
alphaD = np.vstack(alphaD)
plt.scatter(alphaD[:,2], alphaD_opt )
for i in range(len(alphaD_opt)):
    plt.text(alphaD[:,2][i], alphaD_opt[i], str(i), fontsize=9, ha='right', va='bottom')
x = np.linspace(np.min(alphaD_opt), 24, 100)
plt.plot(x,x, color = 'black')
plt.title('Emulator 1')

# plt.xlim(13,24)
# plt.ylim(13,24)


# plot the points
# '''
# Discrepancy on plot with points
# '''
# plt.figure(3)
# x = [float(i) for i,j in combined]
# y = [float(j) for i,j in combined]
# c = np.abs(alphaD_opt - alphaD[:,2])/alphaD[:,2]
# plt.scatter(x, y, c = c, marker='s', cmap = 'Spectral', norm = LogNorm())
# plt.colorbar(label = 'Relative error')

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


'''
Plot L2 norm of strength rather than alphaD
'''
plt.figure(3)
x = [float(i) for i,j in combined]
y = [float(j) for i,j in combined]
c = np.abs(alphaD_opt - alphaD[:,2])/alphaD[:,2]
# L2 (Euclidean) norm of the difference for each pair
#l2_norms = [np.linalg.norm(a - b)/np.linalg.norm(b)  for a, b in zip(opt_strength, orig_strength)]
plt.scatter(x, y, c = c, marker='s', cmap = 'Spectral', norm = LogNorm())
plt.colorbar(label = r'Relative error $\alpha_D$')

# Add index numbers
for idx, (xi, yi) in enumerate(zip(x, y)):
    plt.text(xi, yi, str(idx), ha='center', va='center', fontsize=8, color='black')

# Convert train_set to float arrays for min/max
train_alpha = [float(a) for (a, b) in test_set]
train_beta  = [float(b) for (a, b) in test_set]

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
# Print the cost function from training set
# '''
# strength, alphaD = helper.data_table(test_set)



# alphaD = np.vstack(alphaD)

# '''
# Put alphaD in a list
# '''
# alphaD_list = [float(a[2]) for a in alphaD]
    

# alphaD_opt = np.array(alphaD_opt)
# alphaD_list = [float(a[2]) for a in alphaD]
# weight = 100
# strength, alphaD = helper.data_table(test_set)
# cost, Lor, Lor_true,x, alphaD_train = helper.cost_function_batched_mixed(params, n, test_set, strength, alphaD_list, weight, central_point)
# print('Cost on train set: ', cost)



