#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 11:30:06 2025

@author: anteravlic
"""

import numpy as np
import matplotlib.pyplot as plt


# split the data into test set and so on
alpha_values = np.linspace(0.400,0.850,10)
formatted_alpha_values = [f"{num:.4f}" for num in alpha_values]

beta_values = np.linspace(1.4,2.0,13)
formatted_beta_values = [f"{num:.4f}" for num in beta_values]

for alpha in formatted_alpha_values:
    for beta in formatted_beta_values:
        
        # first open the file with the data
        file_strength = np.loadtxt('../total_strength/strength_'+beta+'_'+alpha+'.out')
        file_alphaD = np.loadtxt('../total_alphaD/alphaD_'+beta+'_'+alpha+'.out')
        
        file_strength = file_strength[file_strength[:,0] > 1]
        
        plt.plot(file_strength[:,0], file_strength[:,1], alpha = 0.1, color = 'black')


plt.xlim(3,25)
plt.ylim(0)

plt.xlabel('$\omega$ (MeV)', size = 16)
plt.ylabel('$S$ ($e^2$fm$^2$/MeV)', size = 16)
plt.tight_layout()

plt.annotate('${}^{180}$Yb', (0.7,0.7), xycoords='axes fraction', size = 18)

plt.gca().tick_params(axis="y",direction="in", which = 'both', labelsize = 12)
plt.gca().tick_params(axis="x",direction="in", which = 'both', labelsize = 12)

#plt.savefig('isovector_dipole_variation.pdf', bbox_inches='tight')

