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
import re
import os


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
        
        if ((float(beta_val) <= 0.9 and float(beta_val) >= 0.1) and (float(alpha_val) <= 1.8 and float(alpha_val) >= 0.2)):
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
    
# Define split ratios (e.g., 60% train, 20% cv, 20% test)
train_ratio = 1.0
cv_ratio = 0.0
test_ratio = 0.0

# Calculate the number of elements for each set
n = len(combined)
n_train = int(n * train_ratio)
n_cv = int(n * cv_ratio)
n_test = n - n_train - n_cv  # Ensure all elements are used

train_set = combined[:n_train]
cv_set = combined[n_train:n_train + n_cv]
test_set = combined[n_train + n_cv:]

'''
This is added to compute a central data point
'''
# Compute centroid
combined_ar = np.array(combined, dtype = float)
centroid = combined_ar.mean(axis=0)

# Compute distances from each point to centroid
distances = np.linalg.norm(combined_ar - centroid, axis=1)

# Find index of closest point
central_index = np.argmin(distances)
central_point = tuple(combined[central_index])
print('Central data point in train set:', central_point)



print(len(test_set), len(cv_set), len(train_set))

# x_train = [float(x) for x, y in train_set]
# y_train = [float(y) for x, y in train_set]
# x_test = [float(x) for x, y in test_set]
# y_test = [float(y) for x, y in test_set]
# x_cv = [float(x) for x, y in cv_set]
# y_cv = [float(y) for x, y in cv_set]


# #Scatter plot
# plt.figure(figsize=(6, 4))
# plt.scatter(x_train, y_train, color='steelblue', marker='o', label = 'train('+str(len(train_set))+')')
# plt.scatter(x_test, y_test, color='red', marker='o', label = 'test('+str(len(test_set))+')')
# plt.scatter(x_cv, y_cv, color='green', marker='d', label = 'cv('+str(len(cv_set))+')')
# plt.xlabel(r"$\alpha (d_{tv}$)", size = 18)
# plt.ylabel(r"$\beta (b_{tv})$", size = 18)

# plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
#             ncol=3, fancybox=True, shadow=True)

# '''
# Indicate the central point in the train set
# '''
# #plt.scatter(float(central_point[0]), float(central_point[1]), color = 'red', marker = 'X')

# sys.exit(-1)

#plt.scatter(*zip(*train_set))
# Seed for reproducibility
np.random.seed(42)

'''
How many parameters you want ?
'''
n = 13
retain = 0.9
weight= 1
width = 2.0

D, S1, S2 = helper.nec_mat(n)
print(D.shape, S1.shape, S2.shape)



'''
    data table is now constructed for alpha & beta parameters
'''

Lors, HLs = helper.data_table(train_set, coeffs, g_A, nucnam)
#Lors_cv, HLs_cv = helper.data_table(cv_set, coeffs, g_A)
'''
n - diagonal matrix
2*n(n+1)/2 - symmetric matrices
n - v0
4 - eta, x1, x2, x3

'''
nec_num_param = n + 2*int(n * (n + 1) / 2) + n + 4 # add v0 to the mix (last n)
print('Number of parameters: ', nec_num_param)
random_initial_guess = np.random.uniform(0, 1, nec_num_param)

'''
Here also re-initialize matrix elements of M0 and v0 !
'''
# keep = round(retain*n)
# data = np.loadtxt(strength_dir+'lorm_'+nucnam+'_'+central_point[1]+'_'+central_point[0]+'.out')
# data = data[data[:,0]>-10]
# data = data[data[:,0]<0.782]
# omega = data[:,0]
# y = data[:,1]
# omega_tf = tf.convert_to_tensor(omega, dtype=tf.float64)
# y_tf = tf.convert_to_tensor(y, dtype=tf.float64)
# eta_tf = tf.convert_to_tensor(width, dtype=tf.float64)
# E_hat, B_hat, y_hat = helper.fit_strength_with_tf_lorentzian(omega_tf, y_tf, keep, eta_tf, min_spacing=0.01)

# plt.plot(data[:,0], y_hat)
# plt.stem(E_hat, B_hat)
# plt.plot(data[:,0], data[:,1])





initial_guess = random_initial_guess #helper.encode_initial_guess(random_initial_guess, E_hat, B_hat, n, retain)

params_shape = [D.shape, S1.shape, S2.shape]

params = tf.Variable(random_initial_guess, dtype=tf.float64)

# lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate=0.01,
#     decay_steps=3000,
#     decay_rate=0.96,
#     staircase=False  # set to False for smooth decay
# )


optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.01)


# Optimization step function
@tf.function
def optimization_step():
    with tf.GradientTape() as tape:
        cost, Lor, Lor_true,x, HLs_calc, B_last, E_last = helper.cost_function(params, n, train_set\
                                              , Lors,HLs, coeffs, g_A, weight, central_point, retain)
    gradients = tape.gradient(cost, [params])
    optimizer.apply_gradients(zip(gradients, [params]))
    return cost, Lor, Lor_true, x, HLs_calc, B_last, E_last



# Run the optimization
num_iterations = 30000 # Total number of optimization iterations
num_check = 300 #Print output after this many steps
cost_loop = []
#cost_loop_cv = []
train_cost_min = 1e10
params_min = params
params_min_train = params
min_train_iter = num_iterations+10
for i in range(num_iterations):
    cost, Lor, Lor_true, x, HLs_check_train, B_last, E_last = optimization_step()
    current_learning_rate = optimizer._decayed_lr(tf.float64).numpy()
    cost_loop.append(cost.numpy())
    
    # cost_cv, HLs_check_cv = cv_step()
    # cost_loop_cv.append(cost_cv.numpy())
    rel =  np.abs(np.array(HLs_check_train)-np.array(HLs))/np.array(HLs)
    
    if (cost < train_cost_min):
        train_cost_min = cost
        params_min_train = params
        min_train_iter = i

    
    
    
    
    if i % num_check == 0:  # Print cost every 100 iterations
        
        
        print(f"Iteration {i}, Cost: {cost.numpy()}, Rate: {current_learning_rate}")
        #print('CV cost: ', cost_cv.numpy())
        print('r: ', np.mean(rel))
        plt.figure(i)
        plt.subplot(121)
        rel =  np.abs(np.array(HLs_check_train)-np.array(HLs))/np.array(HLs)
        plt.plot([j for j in range(len(train_set))],rel, marker = '.', label = 'QRPA calc', ls = '--') 
        
        plt.axhline(np.mean(rel), marker = '.', label = 'QRPA calc', ls = '--', color = 'black') 
        plt.yscale('log')
        plt.title('iter = '+str(i))
        plt.subplot(122)
        plt.stem(E_last, B_last)
        plt.plot(x, Lor)
        plt.plot(x, Lor_true)
        plt.ylim(0)
        plt.xlim(-10,np.max(x))
        plt.axvline(0.8, ls = '--')
        plt.show()
        
        # plot CV half-lives 

plt.plot(range(len(cost_loop)), cost_loop, label = 'Training set')
plt.yscale('log')
print('Final: ', cost.numpy())
plt.xlabel('Number of training iterations', size = 16)
plt.legend()
plt.ylabel('Cost function', size = 16)



# save parameters in a file
np.savetxt('params_'+str(n)+'_'+str(retain)+'.txt', params_min_train.numpy())

with open("train_set.txt", "w") as f:
    for tup in train_set:
        f.write(",".join(map(str, tup)) + "\n")  # Convert tuple to comma-separated string



