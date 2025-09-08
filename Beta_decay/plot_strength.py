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
import re
import os

Z = 28
A = 80
g_A = 1.2
del_np = 1.293
nucnam='Ni_80'

poly = helper.fit_phase_space(0, Z, A, 15)
coeffs = Polynomial(poly).coef


strength_dir = '../beta_decay_data_Ni_80/'

# Pattern for strength files: strength_beta_alpha.out
pattern = re.compile(r'lorm_Ni_80_([0-9.]+)_([0-9.]+)\.out')

formatted_alpha_values = []
formatted_beta_values = []

all_points = []

for fname in os.listdir(strength_dir):
    match = pattern.match(fname)
    
    if match:
        beta_val = match.group(1)
        alpha_val = match.group(2)
        all_points.append((alpha_val, beta_val))
        
        #if ((float(beta_val) <= 0.8 and float(beta_val) >= 0.2) and (float(alpha_val) <= 1.3 and float(alpha_val) >= 0.2)):
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
    
    


Lors, HLs = helper.data_table(combined, coeffs, g_A, nucnam)

for i in range(len(Lors)):
    plt.plot(Lors[i][:,0], Lors[i][:,1])

plt.xlim(-10,0.782)
plt.ylim(0)

plt.xlabel('$\omega$ (MeV)', size = 16)
plt.ylabel('$S$ (1/MeV)', size = 16)
plt.tight_layout()

plt.annotate('${}^{80}$Ni', (0.2,0.7), xycoords='axes fraction', size = 22)

plt.gca().tick_params(axis="y",direction="in", which = 'both', labelsize = 12)
plt.gca().tick_params(axis="x",direction="in", which = 'both', labelsize = 12)

'''
In the same figure I would also like to plot the phase-space
'''
# poly = helper.fit_phase_space(0, Z, A, 15)
# coeffs = Polynomial(poly).coef

# # Example coefficients: f(x) = 2 - x + 0*x^2 + 3*x^3
# coefficients = tf.constant(coeffs[::-1], dtype=tf.float64)

# # Range of x values for plotting
# x_tensor = tf.constant(del_np - Lors[0][:,0], dtype=tf.float64)

# # Evaluate the polynomial at all x values
# y_values = helper.evaluate_polynomial_tf(coefficients, x_tensor).numpy()

# ax2 = plt.gca().twinx()
# ax2.plot(Lors[:,0], y_values, color = 'red')
# ax2.set_yscale('log')
# ax2.set_ylim(1e-3,1e4)
# ax2.set_ylabel('$f(E_0,Z,A)$', color='red', size = 16)
# ax2.tick_params(axis='y', which='both', labelcolor='red', colors='red')  # Tick and label color
# plt.gca().tick_params(axis="y",direction="in", which = 'both', labelsize = 12)





# plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(1))
# plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(0.2))

#plt.savefig('gamow_teller_variation.pdf', bbox_inches='tight')

