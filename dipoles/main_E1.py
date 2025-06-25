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




if __name__ == "__main_E1__":
    print("Running main.py as a script")



'''
Construct the data set
'''
rn.seed(42)



'''
The values of parameters should be read directly from the file name
'''
strength_dir = '../dipoles_data_all/total_strength/'
alphaD_dir = '../dipoles_data_all/total_alphaD/'

# Pattern for strength files: strength_beta_alpha.out
pattern = re.compile(r'strength_([0-9.]+)_([0-9.]+)\.out')

# formatted_alpha_values = []
# formatted_beta_values = []

all_pairs = []
for fname in os.listdir(strength_dir):
    match = pattern.match(fname)
    if match:
        beta_val = match.group(1)
        alpha_val = match.group(2)
        all_pairs.append((beta_val, alpha_val))
        
filtered = [
    (beta, alpha) for (beta, alpha) in all_pairs
    if 0.5 <= float(alpha_val := alpha) <= 0.8
]

for beta, alpha in filtered:
    fstr = os.path.join(strength_dir, f'strength_{beta}_{alpha}.out')
    data = np.loadtxt(fstr)
    mask = data[:,0] > 1.0
    plt.plot(data[mask, 0], data[mask, 1], alpha=0.1, color='black')
    print(f"Plotted strength for beta = {beta}, alpha={alpha}")



plt.xlim(3, 25)
plt.xlabel('$\omega$ (MeV)', size = 16)
plt.ylabel('$S$ ($e^2$fm$^2$/MeV)', size = 16)
plt.tight_layout()
plt.show()

rn.shuffle(filtered)
n_total = len(filtered)
n_train = int(0.9*n_total)
n_cv = n_total - n_train

train_set = filtered[:n_train]
cv_set = filtered[n_train:]

print(f"train_set length: {len(train_set)},"
      f"cv_set length: {len(cv_set)}"
      )





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




# sys.exit(-1)






'''
How many parameters you want ?
'''
# n = 20
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





num_iterations = 40000
early_stop_rel = 5e-4
num_check =  300

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

def optimization_for_n(n,
                       train_set, strength_train, alphaD_train,
                       cv_set, strength_cv, alphaD_cv,
                       weight,
                       num_iterations,
                       early_stop_rel,
                       num_check):
    
    with tf.GradientTape() as tape:
        cost, Lor, Lor_true, omega, alphaD_train = helper.cost_function(
            params, n,
            train_set, strength_train, alphaD_train,
            weight
        )
    grads = tape.gradient(cost, [params])
    optimizer.apply_gradients(zip(grads, [params]))
    return cost, Lor, Lor_true, omega, alphaD_train

    def cv_step():
        cost_cv, Lor, Lor_true, omega, alphaD_cv = helper.cost_function(
            params, n,
            cv_set, strength_cv, alphaD_cv,
            weight
        )
    
    final_cost = None
    for i in range(num_iterations):
        cost, _, _, _, _ = optimization_for_n() #####
        cost_cv, alphaD_pred_cv = cv_step()
        
        rel = tf.abs(alphaD_pred_cv - alphaD_cv[:,2]) / alphaD_cv[:,2]
        if tf.reduce_mean(rel) < early_stop_rel:
            final_cost = cost.numpy()
            break
        
        final_cost = cost.numpy()
        
        if i % num_check ==0:
            mean_rel = tf.reduce_mean(rel).numpy()
            print(f"[n={n}] iter={i}, train cost={cost.numpy():.4e}, cv cost={cost_cv.numpy():.4e}, mean rel.err={mean_rel:.4e}")

    return final_cost

# Sweep over n = 1…20
ns = list(range(1, 21))
costs = []

for n in ns:
    print(f"=== Running optimization for n = {n} ===")
    c = optimization_for_n(
        n,
        train_set, strength_train, alphaD_train,
        cv_set, strength_cv, alphaD_cv,
        weight,
        num_iterations,
        early_stop_rel,
        num_check
    )
    costs.append(c)

# Plotting
plt.figure(figsize=(8,5))
plt.plot(ns, costs, 'o-')
plt.xlabel('Number of Lorentzians $n$')
plt.ylabel('Final training cost')
plt.title('Optimization cost vs. number of Lorentzians')
plt.grid(True)
plt.show()







# save parameters in a file
np.savetxt('params_'+str(n)+'.txt', params.numpy())

with open("test_set.txt", "w") as f:
    for tup in train_set:
        f.write(",".join(map(str, tup)) + "\n")  # Convert tuple to comma-separated string
with open("cv_set.txt", "w") as f:
    for tup in cv_set:
        f.write(",".join(map(str, tup)) + "\n")  # Convert tuple to comma-separated string


