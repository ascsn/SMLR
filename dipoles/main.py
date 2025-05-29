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
import os
import re
import sys


'''
Main data regarding the decay
'''



'''
Construct the data set
'''
rn.seed(42)

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
rn.shuffle(combined)

# Define split ratios (e.g., 60% train, 20% cv, 20% test)
train_ratio = 0.9
cv_ratio = 0.1
test_ratio = 0.0

# Calculate the number of elements for each set
n = len(combined)
n_train = int(n * train_ratio)
n_cv = int(n * cv_ratio)
n_test = 0 #n - n_train - n_cv  # Ensure all elements are used

# Split the combined list
train_set = combined[:n_train]
cv_set = combined[n_train:n_train + n_cv]
test_set = combined[n_train + n_cv:]



print(len(test_set), len(cv_set), len(train_set))

x_train = [float(x) for x, y in train_set]
y_train = [float(y) for x, y in train_set]
x_test = [float(x) for x, y in test_set]
y_test = [float(y) for x, y in test_set]
x_cv = [float(x) for x, y in cv_set]
y_cv = [float(y) for x, y in cv_set]
x_all = [float(x) for x, y in all_points]
y_all = [float(y) for x, y in all_points]

# Scatter plot
plt.figure(figsize=(6, 4))
plt.scatter(x_train, y_train, color='blue', marker='o', label = 'train('+str(len(train_set))+')')
#plt.scatter(x_test, y_test, color='red', marker='o', label = 'test('+str(len(test_set))+')')
plt.scatter(x_cv, y_cv, color='green', marker='o', label = 'cv('+str(len(cv_set))+')')
plt.scatter(x_all, y_all, color='green', marker='x', label = 'cv('+str(len(cv_set))+')')
plt.xlabel(r"$\alpha (d_{tv}$)", size = 18)
plt.ylabel(r"$\beta (b_{tv})$", size = 18)

plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
            ncol=3, fancybox=True, shadow=True)

sys.exit(-1)



'''
How many parameters you want ?
'''
n = 20
weight=100.0
fold=1.0

D, S1, S2 = helper.nec_mat(n)
print(D.shape, S1.shape, S2.shape)


# initialize the external field vector
v0 = np.random.rand(n)

'''
    data table is now constructed for alpha & beta parameters
'''

strength, alphaD = helper.data_table(train_set)
strength_cv, alphaD_cv = helper.data_table(cv_set)

alphaD = np.vstack(alphaD)
alphaD_cv = np.vstack(alphaD_cv)

nec_num_param = n + 2*int(n * (n + 1) / 2) + n + 1 # add v0 to the mix (last n)
# last parameter is fold
print('Number of parameters: ', nec_num_param)
random_initial_guess = np.random.uniform(0, 1, nec_num_param)
random_initial_guess[0] = fold

params_shape = [D.shape, S1.shape, S2.shape]

params = tf.Variable(random_initial_guess, dtype=tf.float64)


optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.01)


# Optimization step function
@tf.function
def optimization_step():
    with tf.GradientTape() as tape:
        cost, Lor, Lor_true,x, alphaD_train = helper.cost_function(params, n, train_set, strength, alphaD, weight)
    gradients = tape.gradient(cost, [params])
    optimizer.apply_gradients(zip(gradients, [params]))
    return cost, Lor, Lor_true, x

@tf.function
def cv_step():
    '''
        Loss function for cross-validation
    '''
    cost, Lor, Lor_true,x, alphaD_train = helper.cost_function(params, n, cv_set, strength_cv, alphaD_cv, weight)
  
    return cost, alphaD_train

# Run the optimization
num_iterations = 40000 # Total number of optimization iterations
num_check = 300 #Print output after this many steps
cost_loop = []
cost_loop_cv = []
for i in range(num_iterations):
    cost, Lor, Lor_true, x = optimization_step()
    current_learning_rate = optimizer._decayed_lr(tf.float32).numpy()
    cost_loop.append(cost.numpy())
    
    cost_cv, alphaD_check_cv = cv_step()
    cost_loop_cv.append(cost_cv.numpy())
    rel =  np.abs(np.array(alphaD_check_cv)-np.array(alphaD_cv[:,2]))/np.array(alphaD_cv[:,2])
    if ( np.mean(rel)< 0.0005): # was 0.00015
        print('Stopped iterations at: ', np.mean(rel))
        break
    
    
    
    
    if i % num_check == 0:  # Print cost every 100 iterations
        
        
        print(f"Iteration {i}, Cost: {cost.numpy()}, Rate: {current_learning_rate}")
        print('CV cost: ', cost_cv.numpy())
        print('r: ', np.mean(rel))
        plt.figure(i)
        plt.subplot(121)
        rel =  np.abs(np.array(alphaD_check_cv)-np.array(alphaD_cv[:,2]))/np.array(alphaD_cv[:,2])
        plt.plot([j for j in range(len(cv_set))],rel, marker = '.', label = 'QRPA calc', ls = '--') 
        plt.axhline(np.mean(rel), marker = '.', label = 'QRPA calc', ls = '--', color = 'black') 
        plt.yscale('log')
        plt.title('iter = '+str(i))
        plt.subplot(122)
        plt.plot(x, Lor)
        plt.plot(x, Lor_true)
        plt.show()
        
        # plot CV half-lives 

plt.plot(range(len(cost_loop)), cost_loop, label = 'Training set')
plt.plot(range(len(cost_loop_cv)), cost_loop_cv, label = 'Cross-validation set')
plt.yscale('log')
print('Final: ', cost.numpy())
plt.xlabel('Number of training iterations', size = 16)
plt.legend()
plt.ylabel('Cost function', size = 16)



# save parameters in a file
np.savetxt('params_'+str(n)+'.txt', params.numpy())

with open("test_set.txt", "w") as f:
    for tup in train_set:
        f.write(",".join(map(str, tup)) + "\n")  # Convert tuple to comma-separated string
with open("cv_set.txt", "w") as f:
    for tup in cv_set:
        f.write(",".join(map(str, tup)) + "\n")  # Convert tuple to comma-separated string


