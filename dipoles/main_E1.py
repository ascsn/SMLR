#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 21:33:09 2025

@author: anteravlic
"""
import helper_mod
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial
from numpy.random import default_rng
import re, os
import sys

from matplotlib.colors import LogNorm
#from data_loader import list_beta_alpha_pairs, data_table



# import helper_mod
# print("main_E1 is loading helper from:", helper_mod.__file__)


# sys.exit(-1)
# if __name__ == "__main_E1__":
#     print("Running main.py as a script")

rng = default_rng(42)


# -------------------------------------------------------------------------
# Configuration: filtering ranges for beta and alpha
# -------------------------------------------------------------------------
min_beta, max_beta   = 1.5, 4.0   # beta range (btv)
min_alpha, max_alpha = 0.4, 1.8   # alpha range (dtv)

# -------------------------------------------------------------------------
# 1. Discover all (beta, alpha) pairs within specified ranges
# -------------------------------------------------------------------------
full_pairs = helper_mod.beta_alpha_pairs()  # every available (beta, alpha)
filtered_pairs = helper_mod.beta_alpha_pairs(
    min_beta=min_beta, max_beta=max_beta,
    min_alpha=min_alpha, max_alpha=max_alpha
)


#print(filtered_pairs)

#sys.exit(-1)


# -------------------------------------------------------------------------
# 2. Create train / cross-validation splits (60/40)
# perhaps using sklearn.model_selection.train_test_split?
# -------------------------------------------------------------------------
rng.shuffle(filtered_pairs)
N_total  = len(filtered_pairs)
n_train  = int(0.6 * N_total)
train_set = filtered_pairs[:n_train]
cv_set    = filtered_pairs[n_train:]

print(f"train_set length: {len(train_set)}, cv_set length: {len(cv_set)}")

#sys.exit(-1)
    
    
# save parameters in a file
with open("train_set.txt", "w") as f:
    for tup in train_set:
        f.write(",".join(map(str, tup)) + "\n")  # Convert tuple to comma-separated string
with open("cv_set.txt", "w") as f:
    for tup in cv_set:
        f.write(",".join(map(str, tup)) + "\n")  # Convert tuple to comma-separated string    
    
    
#sys.exit(-1)
    
    
# -------------------------------------------------------------------------
# 3. Scatter plot showing data partitions
# -------------------------------------------------------------------------
x_all = [float(alpha) for beta, alpha in full_pairs]
y_all = [float(beta)  for beta, alpha in full_pairs]

x_tr  = [float(alpha) for beta, alpha in train_set]
y_tr  = [float(beta)  for beta, alpha in train_set]

x_cv  = [float(alpha) for beta, alpha in cv_set]
y_cv  = [float(beta)  for beta, alpha in cv_set]

plt.figure(figsize=(6,4))
plt.scatter(x_all, y_all, c='lightgray', marker='.', label=f'all ({len(full_pairs)})')
plt.scatter(x_tr,  y_tr,  c='C0',       marker='o', label=f'train ({len(train_set)})')
plt.scatter(x_cv,  y_cv,  c='C2',       marker='s', label=f'cv    ({len(cv_set)})')
plt.xlabel(r'$\alpha$ (dtv)', fontsize=14)
plt.ylabel(r'$\beta$ (btv)', fontsize=14)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False)
plt.tight_layout()
plt.show()

# -------------------------------------------------------------------------
# 4. Load and plot example strength curves for train set
# -------------------------------------------------------------------------
strength_list, alphaD_list = helper_mod.data_table(train_set)
for arr in strength_list:
    mask = arr[:,0] > 1.0
    plt.plot(arr[mask,0], arr[mask,1], alpha=0.1, color='black')

plt.xlim(3, 25)
plt.xlabel(r'$\omega$ (MeV)', size=16)
plt.ylabel(r'$S$ ($e^2$fm$^2$/MeV)', size=16)
plt.tight_layout()
plt.show()




#sys.exit(-1)




'''
How many parameters you want ?
'''
n = 20
weight=100.0
#fold=0.8

D, S1, S2 = helper_mod.initial_matrix(n)
print(D.shape, S1.shape, S2.shape)


# initialize the external field vector
v0 = rng.random(n)


strength_train, alphaD_train = helper_mod.data_table(train_set)
strength_cv, alphaD_cv = helper_mod.data_table(cv_set)

alphaD_train = np.vstack(alphaD_train)
alphaD_cv = np.vstack(alphaD_cv)

param_count = 4 + n + n + 2*int(n*(n+1)/2)
print('Number of parameters: ', param_count)


random_initial_guess = rng.uniform(0, 1, param_count)

params_shape = [D.shape, S1.shape, S2.shape]

params = tf.Variable(
    random_initial_guess,
    dtype=tf.float64
)




''' Optimization '''

num_iterations = 40000
early_stop_rel = 5e-4
num_check =  300

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)


@tf.function
def optimization():
    with tf.GradientTape() as tape:
        cost_train, Lor, Lor_true, omega, alphaD_pred_train = helper_mod.cost_function(
            params, n,
            train_set, strength_train, alphaD_train,
            weight
            )
    grads = tape.gradient(cost_train, [params])
    optimizer.apply_gradients(zip(grads, [params]))
    return cost_train, Lor, Lor_true, omega #, alphaD_pred_train

@tf.function
def cv():
    '''Loss function for cross-validation'''
    cost_cv, Lor, Lor_true, omega, alphaD_pred_cv = helper_mod.cost_function(
        params, n,
        cv_set, strength_cv, alphaD_cv,
        weight
    )
    return cost_cv, alphaD_pred_cv


cost_train_loop = []
cost_cv_loop = []

for i in range(num_iterations):
    cost_train, Lor, Lor_true, omega = optimization()
    current_learning_rate = optimizer.learning_rate.numpy()
    cost_train_loop.append(cost_train.numpy())
    
    cost_cv, alphaD_pred_cv = cv()
    cost_cv_loop.append(cost_cv.numpy())
    
    rel_err = np.abs(np.array(alphaD_pred_cv) - np.array(alphaD_cv[:,2])) / np.array(alphaD_cv[:,2])
    #rel_err = (tf.abs(np.array(alphaD_pred_cv) - np.array(alphaD_cv[:,2])) / np.array(alphaD_cv[:,2]))
    #rel = np.mean(rel)
    
    if (np.mean(rel_err) < 0.0005): # was 0.00015
        print('Stopped iterations at: ', np.mean(rel_err))
        break
    
    if i % num_check == 0:
        print(f'Iteration {i}, Cost: {cost_train.numpy()}, Rate: {current_learning_rate}')
        print('CV cost: ', cost_cv.numpy())
        print('r: ', np.mean(rel_err))
        
        plt.figure(i)
        
        plt.subplot(121)
        #rel = np.abs(np.array(alphaD_check_cv)-np.array(alphaD_cv[:,2]))/np.array(alphaD_cv[:,2])
        plt.plot([j for j in range(len(cv_set))], rel_err, marker = '.', label = 'QRPA calc', ls = '--') 
        plt.axhline(np.mean(rel_err), marker = '.', label = 'QRPA calc', ls = '--', color = 'black') 
        plt.yscale('log')
        plt.title('iter = '+str(i))
        
        plt.subplot(122)
        plt.plot(omega, Lor)
        plt.plot(omega, Lor_true)
        plt.show()






plt.plot(range(len(cost_train_loop)), cost_train_loop, label = 'Training set')
plt.plot(range(len(cost_cv_loop)), cost_cv_loop, label = 'Cross-validation set')
plt.yscale('log')
print('Final: ', cost_train.numpy())
plt.xlabel('Number of training iterations', size = 16)
plt.ylabel('Cost function', size = 16)
plt.legend()





np.savetxt('params_'+str(n)+'.txt', params.numpy())



