#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 17:03:32 2025

@author: anteravlic
"""
import helper
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial
import random as rn


'''
Main data regarding the decay
'''



'''
Construct the data set
'''
rn.seed(42)

# split the data into test set and so on
alpha_values = np.linspace(0.400,0.850,10)
formatted_alpha_values = [f"{num:.4f}" for num in alpha_values]

beta_values = np.linspace(1.4,2.0,13)
formatted_beta_values = [f"{num:.4f}" for num in beta_values]


# Example lists
alpha = formatted_alpha_values
beta = formatted_beta_values

# Combine the lists into pairs
combined = combined = [(x, y) for x in alpha for y in beta]
# Shuffle the combined list
rn.shuffle(combined)

# Define split ratios (e.g., 60% train, 20% cv, 20% test)
train_ratio = 0.3
cv_ratio = 0.1
test_ratio = 0.6

# Calculate the number of elements for each set
n = len(combined)
n_train = int(n * train_ratio)
n_cv = int(n * cv_ratio)
n_test = n - n_train - n_cv  # Ensure all elements are used

# Split the combined list
train_set = combined[:n_train]
cv_set = combined[n_train:n_train + n_cv]
test_set = combined[n_train + n_cv:]



print(len(test_set), len(cv_set), len(train_set))





'''
How many parameters you want ?
'''
n = 6


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

nec_num_param = n + 2*int(n * (n + 1) / 2) + n # add v0 to the mix (last n)
print('Number of parameters: ', nec_num_param)
random_initial_guess = np.random.uniform(0, 10, nec_num_param)

params_shape = [D.shape, S1.shape, S2.shape]

params = tf.Variable(random_initial_guess, dtype=tf.float64)


optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.01)


# Optimization step function
@tf.function
def optimization_step():
    with tf.GradientTape() as tape:
        cost, alphaD_train = helper.cost_function_only_alphaD(params, n, train_set, alphaD)
    gradients = tape.gradient(cost, [params])
    optimizer.apply_gradients(zip(gradients, [params]))
    return cost

@tf.function
def cv_step():
    '''
        Loss function for cross-validation
    '''
    cost, alphaD_train = helper.cost_function_only_alphaD(params, n, cv_set, alphaD_cv)
  
    return cost, alphaD_train

# Run the optimization
num_iterations = 20000 # Total number of optimization iterations
num_check = 300 #Print output after this many steps
cost_loop = []
cost_loop_cv = []
for i in range(num_iterations):
    cost = optimization_step()
    current_learning_rate = optimizer._decayed_lr(tf.float32).numpy()
    cost_loop.append(cost.numpy())
    
    cost_cv, alphaD_check_cv = cv_step()
    cost_loop_cv.append(cost_cv.numpy())
    rel =  np.abs(np.array(alphaD_check_cv)-np.array(alphaD_cv[:,2]))/np.array(alphaD_cv[:,2])
    if (np.mean(rel) < 0.0025):
        print('Stopped iterations at: ', np.mean(rel))
        break
    
    
    
    
    if i % num_check == 0:  # Print cost every 100 iterations
        
        
        print(f"Iteration {i}, Cost: {cost.numpy()}, Rate: {current_learning_rate}")
        print('CV cost: ', cost_cv.numpy())
        print('r: ', np.mean(rel))
        plt.figure(i)
        rel =  np.abs(np.array(alphaD_check_cv)-np.array(alphaD_cv[:,2]))/np.array(alphaD_cv[:,2])
        plt.plot([j for j in range(len(cv_set))],rel, marker = '.', label = 'QRPA calc', ls = '--') 
        plt.axhline(np.mean(rel), marker = '.', label = 'QRPA calc', ls = '--', color = 'black') 
        plt.yscale('log')
        plt.title('iter = '+str(i))
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
np.savetxt('params_'+str(n)+'_only_alphaD.txt', params.numpy())

with open("test_set.txt", "w") as f:
    for tup in test_set:
        f.write(",".join(map(str, tup)) + "\n")  # Convert tuple to comma-separated string




