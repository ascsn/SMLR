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
# Set number of intra-op (within an op) threads
#tf.config.threading.set_intra_op_parallelism_threads(4)

# Set number of inter-op (between ops) threads
#tf.config.threading.set_inter_op_parallelism_threads(4)


'''
Construct the data set
'''
rn.seed(142)
# Seed for reproducibility
np.random.seed(30)

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

    
'''
Find the central point on the dataset
'''

# combined: list of (alpha, beta) points
Amin = min(float(a) for a, b in combined)
Amax = max(float(a) for a, b in combined)
Bmin = min(float(b) for a, b in combined)
Bmax = max(float(b) for a, b in combined)

cx = (Amin + Amax) / 2.0
cy = (Bmin + Bmax) / 2.0

central_point = min(combined, key=lambda t: (float(t[0]) - cx)**2 + (float(t[1]) - cy)**2)

# # Define split ratios (e.g., 60% train, 20% cv, 20% test)
train_ratio = 1.0
cv_ratio = 0.0
test_ratio = 0.0

# # Calculate the number of elements for each set
n = len(combined)
n_train = int(n * train_ratio)
n_cv = int(n * cv_ratio)
n_test = 0 #n - n_train - n_cv  # Ensure all elements are used

# Split the combined list
train_set = combined[:n_train]
cv_set = combined[n_train:n_train + n_cv]
test_set = combined[n_train + n_cv:]
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


print('Central data point in train set:', central_point)


print(len(test_set), len(cv_set), len(train_set))

# x_train = [float(x) for x, y in train_set]
# y_train = [float(y) for x, y in train_set]
# x_test = [float(x) for x, y in test_set]
# y_test = [float(y) for x, y in test_set]
# x_cv = [float(x) for x, y in cv_set]
# y_cv = [float(y) for x, y in cv_set]
# x_all = [float(x) for x, y in all_points]
# y_all = [float(y) for x, y in all_points]

# #Scatter plot
# plt.figure(figsize=(6, 4))
# plt.scatter(x_train, y_train, color='steelblue', marker='o', label = 'train('+str(len(train_set))+')')
# #plt.scatter(x_test, y_test, color='red', marker='o', label = 'test('+str(len(test_set))+')')
# plt.scatter(x_cv, y_cv, color='green', marker='d', label = 'cv('+str(len(cv_set))+')')
# plt.xlabel(r"$\alpha (d_{tv}$)", size = 18)
# plt.ylabel(r"$\beta (b_{tv})$", size = 18)

# plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
#             ncol=3, fancybox=True, shadow=True)

# '''
# Indicate the central point in the train set
# '''
# plt.scatter(float(central_point[0]), float(central_point[1]), color = 'red', marker = 'X')

# sys.exit(-1)



'''
- How many parameters you want ?

- weight for the alpha_d in cost function

- ratio of spectrum retained

'''
n = 8
retain = 0.6



fold=2.0

D, S1, S2 = helper.nec_mat(n)
print(D.shape, S1.shape, S2.shape)


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
Last two parameters are for x1 and x2 added to the ansatz
'''
nec_num_param = 1 + 3*n + n + 4 * int(n * (n + 1) / 2) + 4
# last parameter is fold
print('Number of parameters: ', nec_num_param)
random_initial_guess = np.random.uniform(0,1 , nec_num_param)

'''
This re-initializes some of the parameters so that they are not just random
'''
random_initial_guess[0] = fold
random_initial_guess[-4] = 1  # x1
random_initial_guess[-3] = 1  # x1
random_initial_guess[-2] = 1  # x2
random_initial_guess[-1] = 1  # x3



'''
Here also re-initialize matrix elements of M0 and v0 !
'''
keep = round(retain*n)
data = np.loadtxt(strength_dir+'strength_'+central_point[1]+'_'+central_point[0]+'.out')
omega = data[:,0]
y = data[:,1]
omega_tf = tf.convert_to_tensor(omega, dtype=tf.float32)
y_tf = tf.convert_to_tensor(y, dtype=tf.float32)
eta_tf = tf.convert_to_tensor(fold, dtype=tf.float32)
E_hat, B_hat, y_hat = helper.fit_strength_with_tf_lorentzian(omega_tf, y_tf, keep, eta_tf, min_spacing=0.01)

plt.plot(data[:,0], y_hat)
plt.stem(E_hat, B_hat)
plt.plot(data[:,0], data[:,1])





initial_guess = helper.encode_initial_guess(random_initial_guess, E_hat, B_hat, n, retain)



params = tf.Variable(initial_guess, dtype=tf.float32)
helper.print_all_matrices_from_params(params,n)
print(E_hat)
print(np.sqrt(B_hat))
#sys.exit(-1)


optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.01)


# Optimization step function
@tf.function
def optimization_step():
    with tf.GradientTape() as tape:
        #cost, Lor, Lor_true,x, alphaD_train, B, eigenvalues = helper.cost_function_batched_mixed(params, n, train_set, strength, alphaD_list, weight, central_point,retain)
        cost, strength_cost,alphaD_cost,m1_cost, Lor, Lor_true,x, alphaD_train, B, eigenvalues = \
            helper.cost_function_batched_mixed(params, n, train_set,  \
            strength, alphaD_list, central_point,retain, 100,200,0,875.0,1e-8)
    gradients = tape.gradient(cost, [params])
    optimizer.apply_gradients(zip(gradients, [params]))
    return cost,strength_cost,alphaD_cost,m1_cost, Lor, Lor_true, x, alphaD_train, B, eigenvalues

# @tf.function
# def cv_step():
#     '''
#         Loss function for cross-validation
#     '''
#     cost, Lor, Lor_true,x, alphaD_train = helper.cost_function_batched_mixed(params, n, cv_set, strength_cv, alphaD_cv_list, weight, central_point)
  
#     return cost, alphaD_train

# Run the optimization
num_iterations = 30000 # Total number of optimization iterations
num_check = 1000 #Print output after this many steps
cost_loop = []
#cost_loop_cv = []
#cv_cost_min = 1e10
train_cost_min = 1e10
params_min = params
params_min_train = params
min_train_iter = num_iterations+10
#min_cv_iter = num_iterations+10
for i in range(num_iterations):
    cost,strength_cost,alphaD_cost,m1_cost, Lor, Lor_true, x, alphaD_train, B, eigs = optimization_step()
    current_learning_rate = optimizer._decayed_lr(tf.float32).numpy()
    cost_loop.append(cost.numpy())
    
    #print(m1_cost)
    
    # cost_cv, alphaD_check_cv = cv_step()
    # cost_loop_cv.append(cost_cv.numpy())
    rel =  np.abs(np.array(alphaD_train)-np.array(alphaD[:,2]))/np.array(alphaD[:,2])
    
    # if (cost_cv < cv_cost_min):
    #     cv_cost_min = cost_cv
    #     params_min = params
    #     min_cv_iter = i
    #     min_cv_r = np.mean(rel)
        
    if (cost < train_cost_min):
        train_cost_min = cost
        params_min_train = params
        min_train_iter = i
        
    
    
    
    
    if i % num_check == 0:  # Print cost every 100 iterations
        
        
        print(f"Iteration {i}, Cost: {cost.numpy()}, Rate: {current_learning_rate}")
        print(f"Iteration {i}, str_Cost: {strength_cost.numpy()/cost.numpy()},alphaD_Cost: {alphaD_cost.numpy()/cost.numpy()}, m1_Cost: {m1_cost.numpy()/cost.numpy()}")
        # print('CV cost: ', cost_cv.numpy())
        print('r: ', np.mean(rel))
        plt.figure(i)
        plt.subplot(121)
        rel =  np.abs(np.array(alphaD_train)-np.array(alphaD[:,2]))/np.array(alphaD[:,2])
        plt.plot([j for j in range(len(train_set))],rel, marker = '.', label = 'QRPA calc', ls = '--') 
        plt.axhline(np.mean(rel), marker = '.', label = 'QRPA calc', ls = '--', color = 'black') 
        plt.yscale('log')
        plt.title('iter = '+str(i))
        plt.subplot(122)
        plt.plot(x, Lor)
        plt.plot(x, Lor_true)
        plt.stem(eigs, B)
        plt.show()
        
        # plot CV half-lives 

plt.plot(range(len(cost_loop)), cost_loop, label = 'Training set')
#plt.plot(range(len(cost_loop_cv)), cost_loop_cv, label = 'Cross-validation set')
plt.yscale('log')
print('Final: ', cost.numpy())
plt.xlabel('Number of training iterations', size = 16)
plt.legend()
plt.ylabel('Cost function', size = 16)



# save parameters in a file
#print('Iteration with min CV:', cv_cost_min, min_cv_iter, min_cv_r, ' cost normalized: ', cv_cost_min/len(cv_set))
print('Iteration with min cost:', train_cost_min, min_train_iter, ' cost normalized: ', train_cost_min/len(train_set))
np.savetxt('params_'+str(n)+'_'+str(retain)+'.txt', params_min_train.numpy())

with open('train_set.txt', "w") as f:
    for tup in train_set:
        f.write(",".join(map(str, tup)) + "\n")  # Convert tuple to comma-separated string
# with open('test_set.txt', "w") as f:
#     for tup in test_set:
#         f.write(",".join(map(str, tup)) + "\n")  # Convert tuple to comma-separated string
# with open('cv_set.txt', "w") as f:
#     for tup in cv_set:
#         f.write(",".join(map(str, tup)) + "\n")  # Convert tuple to comma-separated string


