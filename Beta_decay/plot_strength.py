#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 17:42:19 2025

@author: anteravlic
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import helper
from numpy.polynomial.polynomial import Polynomial
import tensorflow as tf

Z = 28
A = 74
del_np = 1.293 # MeV


# split the data into test set and so on
# split the data into test set and so on
alpha_values = np.linspace(0,1.5,16)
formatted_alpha_values = [f"{num:.3f}" for num in alpha_values]
beta_values = np.linspace(0,1.,11)
formatted_beta_values = [f"{num:.3f}" for num in beta_values]

for alpha in formatted_alpha_values:
    for beta in formatted_beta_values:
        
        # first open the file with the data
        file_strength = np.loadtxt('../beta_decay_data/lorm_Ni_74_'+beta+'_'+alpha+'.out')
        
        file_strength = file_strength[file_strength[:,0] < 0.782]
        
        plt.plot(file_strength[:,0], file_strength[:,1], alpha = 0.1, color = 'black')


plt.xlim(-4,0.782)
plt.ylim(0)

plt.xlabel('$\omega$ (MeV)', size = 16)
plt.ylabel('$S$ (1/MeV)', size = 16)
plt.tight_layout()

plt.annotate('${}^{74}$Ni', (0.2,0.7), xycoords='axes fraction', size = 22)

plt.gca().tick_params(axis="y",direction="in", which = 'both', labelsize = 12)
plt.gca().tick_params(axis="x",direction="in", which = 'both', labelsize = 12)

'''
In the same figure I would also like to plot the phase-space
'''
poly = helper.fit_phase_space(0, Z, A, 15)
coeffs = Polynomial(poly).coef

# Example coefficients: f(x) = 2 - x + 0*x^2 + 3*x^3
coefficients = tf.constant(coeffs[::-1], dtype=tf.float64)

# Range of x values for plotting
x_tensor = tf.constant(del_np - file_strength[:,0], dtype=tf.float64)

# Evaluate the polynomial at all x values
y_values = helper.evaluate_polynomial_tf(coefficients, x_tensor).numpy()

ax2 = plt.gca().twinx()
ax2.plot(file_strength[:,0], y_values, color = 'red')
ax2.set_yscale('log')
ax2.set_ylim(1e-3,1e4)
ax2.set_ylabel('$f(E_0,Z,A)$', color='red', size = 16)
ax2.tick_params(axis='y', which='both', labelcolor='red', colors='red')  # Tick and label color
plt.gca().tick_params(axis="y",direction="in", which = 'both', labelsize = 12)





# plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(1))
# plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(0.2))

#plt.savefig('gamow_teller_variation.pdf', bbox_inches='tight')

