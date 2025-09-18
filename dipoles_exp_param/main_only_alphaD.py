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
import random as rn
import re
import os



'''
Construct the data set
'''
rn.seed(495)
# Seed for reproducibility
np.random.seed(58)

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
        
        if ((float(beta_val) <= 4.0 and float(beta_val) >= 1.5) and (float(alpha_val) <= 1.8 and float(alpha_val) >= 0.4)):
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
# rn.shuffle(combined)

# # Define split ratios (e.g., 60% train, 20% cv, 20% test)
train_ratio = 1.0
cv_ratio = 0.0
test_ratio = 0.0

# # Calculate the number of elements for each set
n = len(combined)
n_train = int(n * train_ratio)
n_cv = int(n * cv_ratio)
n_test = n - n_train - n_cv  # Ensure all elements are used

# Split the combined list
train_set = combined[:n_train]
cv_set = combined[n_train:n_train + n_cv]
test_set = combined[n_train + n_cv:]

'''
Always include 4 points of the edge of the train region
'''
# Convert combined to array of floats
alpha_float = np.array([float(a) for (a,b) in combined])
beta_float  = np.array([float(b) for (a,b) in combined])

alpha_min = np.min(alpha_float)
alpha_max = np.max(alpha_float)
beta_min  = np.min(beta_float)
beta_max  = np.max(beta_float)

'''
This is added to compute a central data point
'''
# combined: list of (alpha, beta) points
Amin = min(float(a) for a, b in combined)
Amax = max(float(a) for a, b in combined)
Bmin = min(float(b) for a, b in combined)
Bmax = max(float(b) for a, b in combined)

cx = (Amin + Amax) / 2.0
cy = (Bmin + Bmax) / 2.0

central_point = min(combined, key=lambda t: (float(t[0]) - cx)**2 + (float(t[1]) - cy)**2)
print('Central data point in train set:', central_point)





'''
How many parameters you want ?
'''
n = 10


D, S1, S2 = helper.nec_mat(n)
print(D.shape, S1.shape, S2.shape)


# initialize the external field vector
v0 = np.random.rand(n)

'''
    data table is now constructed for alpha & beta parameters
'''

strength, alphaD = helper.data_table(train_set)
#strength_cv, alphaD_cv = helper.data_table(cv_set)

alphaD = np.vstack(alphaD)
#alphaD_cv = np.vstack(alphaD_cv)

'''
Put alphaD in a list
'''
alphaD_list = [float(a[2]) for a in alphaD]
#alphaD_cv_list = [float(a[2]) for a in alphaD_cv]

'''
First n is for diagonal matrix
3* ... for 3 symmetric matrices
and last one is for x1 in  the exponential
'''
nec_num_param = n + 3*int(n * (n + 1) / 2) + 1 
print('Number of parameters: ', nec_num_param)
random_initial_guess = np.random.uniform(0, 2, nec_num_param)


params = tf.Variable(random_initial_guess, dtype=tf.float32)


# optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.1)
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.05)



# Optimization step function
@tf.function
def optimization_step():
    with tf.GradientTape() as tape:
        cost, alphaD_train = helper.cost_function_only_alphaD_batched(params, n, train_set, alphaD_list, central_point)
    gradients = tape.gradient(cost, [params])
    optimizer.apply_gradients(zip(gradients, [params]))
    return cost, alphaD_train

# @tf.function
# def cv_step():
#     '''
#         Loss function for cross-validation
#     '''
#     cost, alphaD_train = helper.cost_function_only_alphaD_batched(params, n, cv_set, alphaD_cv_list, central_point)
  
#     return cost, alphaD_train

# Run the optimization
num_iterations = 40000 # Total number of optimization iterations
num_check = 1000 #Print output after this many steps
cost_loop = []
cost_loop_cv = []
cv_cost_min = 1e5
train_cost_min = 1e3
params_min = params
params_in_train = params
for i in range(num_iterations):
    cost, alphaD_check_train = optimization_step()
    # current_learning_rate = optimizer._decayed_lr(tf.float32).numpy()
    current_learning_rate = optimizer.lr.numpy()
    cost_loop.append(cost.numpy())
    
    #cost_cv, alphaD_check_cv = cv_step()
    #cost_loop_cv.append(cost_cv.numpy())
    rel =  np.abs(np.array(alphaD_check_train)-np.array(alphaD[:,2]))/np.array(alphaD[:,2])
    
    # if (cost_cv < cv_cost_min):
    #     cv_cost_min = cost_cv
    #     params_min = params
    #     min_cv_iter = i
        
    if (cost < train_cost_min):
        train_cost_min = cost
        params_min_train = params
        min_train_iter = i
    
    '''
    Keep the best configuration
    '''
    
    # if (np.mean(rel) < 0.0025):
    #     print('Stopped iterations at: ', np.mean(rel))
    #     break
    
    
    
    
    if i % num_check == 0:  # Print cost every 100 iterations
        
        
        print(f"Iteration {i}, Cost: {cost.numpy()}, Rate: {current_learning_rate}")
        #print('CV cost: ', cost_cv.numpy())
        print('r: ', np.mean(rel))
        plt.figure(i)
        rel =  np.abs(np.array(alphaD_check_train)-np.array(alphaD[:,2]))/np.array(alphaD[:,2])
        plt.plot([j for j in range(len(train_set))],alphaD_check_train, marker = '.', label = 'Emu', ls = '--', color = 'red') 
        plt.plot([j for j in range(len(train_set))],np.array(alphaD[:,2]), marker = '.', label = 'QRPA', ls = '--',color = 'black') 
        # plt.axhline(np.mean(rel), marker = '.', label = 'QRPA calc', ls = '--', color = 'black') 
        plt.yscale('log')
        plt.title('iter = '+str(i))
        plt.savefig('it_out.png')
        plt.show()
        
        # plot CV half-lives 

plt.plot(range(len(cost_loop)), cost_loop, label = 'Training set')
#plt.plot(range(len(cost_loop_cv)), cost_loop_cv, label = 'Cross-validation set')
plt.yscale('log')
print('Final: ', cost.numpy())
plt.xlabel('Number of training iterations', size = 16)
plt.legend()
plt.ylabel('Cost function', size = 16)
#plt.savefig('cost_function.png')



# save parameters in a file
#print('Iteration with min CV:', cv_cost_min, min_cv_iter)
print('Iteration with min cost:', train_cost_min, min_train_iter)
np.savetxt('params_'+str(n)+'_only_alphaD.txt', params_min_train.numpy())

with open('train_set.txt', "w") as f:
    for tup in train_set:
        f.write(",".join(map(str, tup)) + "\n")  # Convert tuple to comma-separated string
# with open('test_set.txt', "w") as f:
#     for tup in test_set:
#         f.write(",".join(map(str, tup)) + "\n")  # Convert tuple to comma-separated string
# with open('cv_set.txt', "w") as f:
#     for tup in cv_set:
#         f.write(",".join(map(str, tup)) + "\n")  # Convert tuple to comma-separated string
# with open('central_point.txt', "w") as f:
#         f.write(",".join(map(str, central_point)) + "\n")  # Convert tuple to comma-separated string



