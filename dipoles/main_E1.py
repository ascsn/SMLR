#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 21:33:09 2025

@author: anteravlic
"""
import helper
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial
import random as rn
import re, os
import sys

from matplotlib.colors import LogNorm





# if __name__ == "__main_E1__":
#     print("Running main.py as a script")


rn.seed(42)



'''
Constructing data set for strength_dir and alphaD_dir
'''
strength_dir = '../dipoles_data_all/total_strength/'
alphaD_dir = '../dipoles_data_all/total_alphaD/'

# Pattern for strength files: strength_beta_alpha.out
pattern = re.compile(r'strength_([0-9.]+)_([0-9.]+)\.out')
pattern_alphaD = re.compile(r'alphaD_([0-9.]+)_([0-9.]+)\.out')

all_pairs = []
for fname in os.listdir(strength_dir):
    match = pattern.match(fname)
    if match:
        beta_val = match.group(1)
        alpha_val = match.group(2)
        all_pairs.append((beta_val, alpha_val))
            
filtered = [
    (beta, alpha) for (beta, alpha) in all_pairs
    if 1.5 <= float(beta_val := beta) <= 4.0
    if 0.4 <= float(alpha_val := alpha) <= 1.8
]

for beta, alpha in filtered:
    fstr = os.path.join(strength_dir, f'strength_{beta}_{alpha}.out')
    data = np.loadtxt(fstr)
    mask = data[:,0] > 1.0
    plt.plot(data[mask, 0], data[mask, 1], alpha=0.1, color='black')
    #print(f"Plotted strength for beta = {beta}, alpha={alpha}")



plt.xlim(3, 25)
plt.xlabel('$\omega$ (MeV)', size = 16)
plt.ylabel('$S$ ($e^2$fm$^2$/MeV)', size = 16)
plt.tight_layout()
plt.show()





'''Dividing train_set, cv_set, test_set''' # perhaps using sklearn.model_selection.train_test_split?

rn.shuffle(filtered)
n_total = len(filtered)
n_train = int(0.6*n_total)
n_cv = n_total - n_train

train_set = filtered[:n_train]
cv_set = filtered[n_train:]

print(f"train_set length: {len(train_set)}, cv_set length: {len(cv_set)}"
      )




#scatter plot showing data points

x_all = [float(alpha) for beta, alpha in all_pairs]
y_all = [float(beta)  for beta, alpha in all_pairs]

x_tr = [float(alpha) for beta, alpha in train_set]
y_tr = [float(beta)  for beta, alpha in train_set]

x_cv = [float(alpha) for beta, alpha in cv_set]
y_cv = [float(beta)  for beta, alpha in cv_set]

plt.figure(figsize=(6,4))
plt.scatter(x_all, y_all,
            c='lightgray', marker='.',
            label=f'all ({len(all_pairs)})')
plt.scatter(x_tr, y_tr,
            c='C0', marker='o',
            label=f'train ({len(train_set)})')
plt.scatter(x_cv, y_cv,
            c='C2', marker='s',
            label=f'cv    ({len(cv_set)})')
plt.xlabel(r'$\alpha$ (dtv)', fontsize=14)
plt.ylabel(r'$\beta$ (btv)', fontsize=14)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
           ncol=3, frameon=False)
plt.tight_layout()
plt.show()


#sys.exit(-1)




'''
How many parameters you want ?
'''
n = 10
weight=100.0
#fold=0.8

D, S1, S2 = helper.initial_matrix(n)
print(D.shape, S1.shape, S2.shape)


# initialize the external field vector
v0 = np.random.rand(n)


strength_train, alphaD_train = helper.data_table(train_set)
strength_cv, alphaD_cv = helper.data_table(cv_set)

alphaD_train = np.vstack(alphaD_train)
alphaD_cv = np.vstack(alphaD_cv)

param_count = 4 + n + n + 2*int(n*(n+1)/2)
print('Number of parameters: ', param_count)


random_initial_guess = np.random.uniform(0, 1, param_count)

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
        cost_train, Lor, Lor_true, omega, alphaD_pred_train = helper.cost_function(
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
    cost_cv, Lor, Lor_true, omega, alphaD_pred_cv = helper.cost_function(
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
    
    rel_err = tf.abs(np.array(alphaD_pred_cv) - np.array(alphaD_cv[:,2]) / np.array(alphaD_cv[:,2]))
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




# save parameters in a file
np.savetxt('params_'+str(n)+'.txt', params.numpy())

with open("train_set.txt", "w") as f:
    for tup in train_set:
        f.write(",".join(map(str, tup)) + "\n")  # Convert tuple to comma-separated string
with open("cv_set.txt", "w") as f:
    for tup in cv_set:
        f.write(",".join(map(str, tup)) + "\n")  # Convert tuple to comma-separated string


