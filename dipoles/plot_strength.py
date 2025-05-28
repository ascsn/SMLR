#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 11:30:06 2025

@author: anteravlic
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import re


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

for fname in os.listdir(strength_dir):
    match = pattern.match(fname)
    if match:
        beta = match.group(1)
        alpha = match.group(2)

        formatted_alpha_values.append(alpha)
        formatted_beta_values.append(beta)


# Example lists
alpha = formatted_alpha_values
beta = formatted_beta_values

count = 0
for i in range(len(formatted_alpha_values)):
    
    alpha = formatted_alpha_values[i]
    beta = formatted_beta_values[i]
    
    if ((float(alpha) >= 0.5 and float(alpha) <= 0.8)):
        
        plt.figure(i)
        
        # first open the file with the data
        file_strength = np.loadtxt('../dipoles_data_all/total_strength/strength_'+beta+'_'+alpha+'.out')
        file_alphaD = np.loadtxt('../dipoles_data_all/total_alphaD/alphaD_'+beta+'_'+alpha+'.out')
        
        file_strength = file_strength[file_strength[:,0] > 1]
        
        plt.plot(file_strength[:,0], file_strength[:,1], alpha = 0.1, color = 'black')
        #print(alpha, beta)
        plt.title(alpha+', '+ beta)
        count += 1
        
        plt.show()
    
print(count)

plt.xlim(3,25)
plt.ylim(0)

plt.xlabel('$\omega$ (MeV)', size = 16)
plt.ylabel('$S$ ($e^2$fm$^2$/MeV)', size = 16)
plt.tight_layout()

plt.annotate('${}^{180}$Yb', (0.7,0.7), xycoords='axes fraction', size = 18)

plt.gca().tick_params(axis="y",direction="in", which = 'both', labelsize = 12)
plt.gca().tick_params(axis="x",direction="in", which = 'both', labelsize = 12)

#plt.savefig('isovector_dipole_variation.pdf', bbox_inches='tight')

