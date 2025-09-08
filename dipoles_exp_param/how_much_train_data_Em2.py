#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 22 21:29:33 2025

@author: anteravlic
"""

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

# Set number of intra-op (within an op) threads
tf.config.threading.set_intra_op_parallelism_threads(4)

# Set number of inter-op (between ops) threads
tf.config.threading.set_inter_op_parallelism_threads(4)

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
        
        #if ((float(beta_val) <= 4.0 and float(beta_val) >= 1.5) and (float(alpha_val) <= 1.8 and float(alpha_val) >= 0.4)):
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
train_ratio = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
cv_cost_list = []
train_cost_list = []
test_cost_list = []
for t in range(len(train_ratio)):
    cv_ratio = 0.1
    test_ratio = 1.0 - cv_ratio - train_ratio[t]
    
    # # Calculate the number of elements for each set
    n = len(combined)
    n_train = int(n * train_ratio[t])
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
    # alpha_float = np.array([float(a) for (a,b) in combined])
    # beta_float  = np.array([float(b) for (a,b) in combined])
    
    # alpha_min = np.min(alpha_float)
    # alpha_max = np.max(alpha_float)
    # beta_min  = np.min(beta_float)
    # beta_max  = np.max(beta_float)
    
    # # Define corner points (as string tuples to match combined format)
    # corners = [
    #     (f"{alpha_min:.4f}", f"{beta_min:.4f}"),
    #     (f"{alpha_min:.4f}", f"{beta_max:.4f}"),
    #     (f"{alpha_max:.4f}", f"{beta_min:.4f}"),
    #     (f"{alpha_max:.4f}", f"{beta_max:.4f}")
    # ]
    
    # #Separate corner points
    # corner_set = [p for p in combined if p in corners]
    
    # # Remaining points
    # non_corner = [p for p in combined if p not in corners]
    
    # # Shuffle the remaining points
    # rn.shuffle(non_corner)
    
    # # Calculate number of elements
    # n_non_corner = len(non_corner)
    # n_train = int(n_non_corner * train_ratio[t])
    
    # # Build final splits
    # train_set = corner_set + non_corner[:n_train]
    # cv_set    = non_corner[n_train:n_train + n_cv]
    # test_set  = non_corner[n_train + n_cv:]
    print('Train:' , len(train_set))
    print('Total:' , len(combined))
    '''
    This is added to compute a central data point
    '''
    combined_ar = np.array(combined, dtype = float)
    centroid = combined_ar.mean(axis=0)
    
    # Compute distances from each point to centroid
    distances = np.linalg.norm(combined_ar - centroid, axis=1)
    
    # Find index of closest point
    central_index = np.argmin(distances)
    central_point = tuple(combined[central_index])
    print('Central data point in train set:', central_point)
    
    
    
    
    
    '''
    How many parameters you want ?
    '''
    n = 4
    
    
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
    '''
    Put alphaD in a list
    '''
    alphaD_list = [float(a[2]) for a in alphaD]
    alphaD_cv_list = [float(a[2]) for a in alphaD_cv]
    
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
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.01)
    
    
    
    # Optimization step function
    @tf.function
    def optimization_step():
        with tf.GradientTape() as tape:
            cost, alphaD_train = helper.cost_function_only_alphaD_batched(params, n, train_set, alphaD_list, central_point)
        gradients = tape.gradient(cost, [params])
        optimizer.apply_gradients(zip(gradients, [params]))
        return cost
    
    @tf.function
    def cv_step():
        '''
            Loss function for cross-validation
        '''
        cost, alphaD_train = helper.cost_function_only_alphaD_batched(params, n, cv_set, alphaD_cv_list, central_point)
      
        return cost, alphaD_train
    
    # Run the optimization
    num_iterations = 25000 # Total number of optimization iterations
    num_check = 1000 #Print output after this many steps
    cost_loop = []
    cost_loop_cv = []
    cv_cost_min = 1e5
    train_cost_min = 1e3
    params_min = params
    params_in_train = params
    for i in range(num_iterations):
        cost = optimization_step()
        # current_learning_rate = optimizer._decayed_lr(tf.float32).numpy()
        current_learning_rate = optimizer.lr.numpy()
        cost_loop.append(cost.numpy())
        
        cost_cv, alphaD_check_cv = cv_step()
        cost_loop_cv.append(cost_cv.numpy())
        rel =  np.abs(np.array(alphaD_check_cv)-np.array(alphaD_cv_list))/np.array(alphaD_cv_list)
        
        if (cost_cv < cv_cost_min):
            cv_cost_min = cost_cv
            params_min = params
            min_cv_iter = i
            
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
            print('CV cost: ', cost_cv.numpy())
            print('r: ', np.mean(rel))
            plt.figure(i)
            rel =  np.abs(np.array(alphaD_check_cv)-np.array(alphaD_cv_list))/np.array(alphaD_cv_list)
            plt.plot([j for j in range(len(cv_set))],alphaD_check_cv, marker = '.', label = 'Emu', ls = '--', color = 'red') 
            plt.plot([j for j in range(len(cv_set))],np.array(alphaD_cv_list), marker = '.', label = 'QRPA', ls = '--',color = 'black') 
            # plt.axhline(np.mean(rel), marker = '.', label = 'QRPA calc', ls = '--', color = 'black') 
            plt.yscale('log')
            plt.title('iter = '+str(i))
            plt.savefig('it_out.png')
            plt.show()
            
            # plot CV half-lives 
    
    plt.plot(range(len(cost_loop)), cost_loop, label = 'Training set')
    plt.plot(range(len(cost_loop_cv)), cost_loop_cv, label = 'Cross-validation set')
    plt.yscale('log')
    print('Final: ', cost.numpy())
    plt.xlabel('Number of training iterations', size = 16)
    plt.legend()
    plt.ylabel('Cost function', size = 16)
    #plt.savefig('cost_function.png')
    
    
    
    # save parameters in a file
    print('Iteration with min CV:', cv_cost_min, min_cv_iter)
    print('Iteration with min cost:', train_cost_min, min_train_iter)
    np.savetxt('params_'+str(n)+'_only_alphaD.txt', params_min.numpy())
    
    cv_cost_list.append(cv_cost_min/len(cv_set))
    train_cost_list.append(train_cost_min/len(train_set))

    
plt.figure(3)
plt.plot(train_ratio, [i.numpy() for i in train_cost_list], label = 'Train')
plt.plot(train_ratio, [i.numpy() for i in cv_cost_list], label = 'CV')

plt.legend()



