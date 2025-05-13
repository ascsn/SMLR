#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 10:23:52 2025

@author: anteravlic
"""

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


'''
Main data regarding the decay
'''
A = 74
Z = 28
g_A = 1.2


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
alpha_values = np.linspace(0,1.5,16)
formatted_alpha_values = [f"{num:.3f}" for num in alpha_values]
beta_values = np.linspace(0,1.,11)
formatted_beta_values = [f"{num:.3f}" for num in beta_values]


# Example lists
alpha = formatted_alpha_values
beta = formatted_beta_values

# Combine the lists into pairs
combined = combined = [(x, y) for x in alpha for y in beta]
# Shuffle the combined list
rn.shuffle(combined)

# Define split ratios (e.g., 60% train, 20% cv, 20% test)
train_ratio = 0.1
cv_ratio = 0.2
test_ratio = 0.7

# Calculate the number of elements for each set
n = len(combined)
n_train = int(n * train_ratio)
n_cv = int(n * cv_ratio)
n_test = n - n_train - n_cv  # Ensure all elements are used

train_set = combined[:n_train]
cv_set = combined[n_train:n_train + n_cv]
test_set = combined[n_train + n_cv:]



print(len(test_set), len(cv_set), len(train_set))

#plt.scatter(*zip(*train_set))


'''
How many parameters you want ?
'''
n = 6

D, S1, S2 = helper.nec_mat(n)
print(D.shape, S1.shape, S2.shape)


'''
    data table is now constructed for alpha & beta parameters
'''

print('Reading data ...')
HLs = helper.data_table_only_HL(train_set,coeffs, g_A)
HLs_cv = helper.data_table_only_HL(cv_set,coeffs, g_A)
print('Data read ...')
nec_num_param = n + 2*int(n * (n + 1) / 2) 
print('Number of parameters: ', nec_num_param)
random_initial_guess = np.random.uniform(0, 1, nec_num_param)

params_shape = [D.shape, S1.shape, S2.shape]

params = tf.Variable(random_initial_guess, dtype=tf.float64)


optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.01)


# Optimization step function
@tf.function
def optimization_step():
    with tf.GradientTape() as tape:
        cost, HLs_calc = helper.cost_function_only_HL(params, n, train_set,HLs)
    gradients = tape.gradient(cost, [params])
    optimizer.apply_gradients(zip(gradients, [params]))
    return cost, HLs_calc

@tf.function
def cv_step():
    '''
        Loss function for cross-validation
    '''
    cost, HLs_calc = helper.cost_function_only_HL(params, n, cv_set, HLs_cv)
  
    return cost, HLs_calc

# Run the optimization
num_iterations = 20000 # Total number of optimization iterations
num_check = 300 #Print output after this many steps
cost_loop = []
cost_loop_cv = []
for i in range(num_iterations):
    cost, HLs_train = optimization_step()
    current_learning_rate = optimizer._decayed_lr(tf.float32).numpy()
    cost_loop.append(cost.numpy())
    
    cost_cv, HLs_check_cv = cv_step()
    cost_loop_cv.append(cost_cv.numpy())
    rel =  np.abs(np.array(HLs_check_cv)-np.array(HLs_cv))/np.array(HLs_cv)
    if (np.mean(rel) < 0.008):
        break
    
    
    
    
    if i % num_check == 0:  # Print cost every 100 iterations
        
        
        print(f"Iteration {i}, Cost: {cost.numpy()}, Rate: {current_learning_rate}")
        print('CV cost: ', cost_cv.numpy())
        print('r: ', np.mean(rel))
        plt.figure(i)
        rel =  np.abs(np.array(HLs_check_cv)-np.array(HLs_cv))/np.array(HLs_cv)
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
np.savetxt('params_'+str(n)+'_only_HL.txt', params.numpy())

with open("test_set.txt", "w") as f:
    for tup in test_set:
        f.write(",".join(map(str, tup)) + "\n")  # Convert tuple to comma-separated string



