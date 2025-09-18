#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 14:13:38 2025

@author: anteravlic
"""

import numpy as np
import matplotlib.pyplot as plt
import helper
import re
import os

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']


        
# central_point = []
# with open("central_point.txt", "r") as f:
#     for line in f:
#         tup = tuple(map(str, line.strip().split(",")))  # Convert back to tuple of integers
#         central_point = tup


test_set = []
with open("train_set.txt", "r") as f:
    for line in f:
        tup = tuple(map(str, line.strip().split(",")))  # Convert back to tuple of integers
        test_set.append(tup)
        
'''
This is added to compute a central data point
'''
Amin = min(float(a) for a, b in test_set)
Amax = max(float(a) for a, b in test_set)
Bmin = min(float(b) for a, b in test_set)
Bmax = max(float(b) for a, b in test_set)

cx = (Amin + Amax) / 2.0
cy = (Bmin + Bmax) / 2.0

central_point = min(test_set, key=lambda t: (float(t[0]) - cx)**2 + (float(t[1]) - cy)**2)


print('Central data point in train set:', central_point)

strength = []
alphaD = []
fmt_data = []

formatted_alpha_values = []
formatted_beta_values = []

strength_dir = '../dipoles_data_all/total_strength/'
alphaD_dir = '../dipoles_data_all/total_alphaD/'

# Pattern for strength files: strength_beta_alpha.out
pattern = re.compile(r'strength_([0-9.]+)_([0-9.]+)\.out')

for fname in os.listdir(strength_dir):
    match = pattern.match(fname)
    if match:
        beta = match.group(1)
        alpha = match.group(2)
        #if ((alpha, beta)) in test_set:
        if ((alpha, beta)) not in test_set:
            #if ((float(beta_val) <= 4.0 and float(beta_val) >= 1.5) and (float(alpha_val) <= 1.8 and float(alpha_val) >= 0.4)):
        #if (float(alpha) > 0.5) :
            strength_file = os.path.join(strength_dir, fname)
            alphaD_file = os.path.join(alphaD_dir, f'alphaD_{beta}_{alpha}.out')

            if os.path.exists(alphaD_file):
                # Read data
                file_strength = np.loadtxt(strength_file)
                file_alphaD = np.loadtxt(alphaD_file)
    
    
                # Store
                strength.append(file_strength)
                alphaD.append(file_alphaD)
                fmt_data.append((alpha, beta))
                
                formatted_alpha_values.append(alpha)
                formatted_beta_values.append(beta)



# Example lists
alpha = formatted_alpha_values
beta = formatted_beta_values

# Combine the lists into pairs
#combined = [(x, y) for x in alpha for y in beta]
combined = []
for i in range(len(alpha)):
    combined.append((alpha[i], beta[i]))

'''
Emulator 1 goes here 
'''


num_par = [7,12,16]
retain = 0.6

print('Total number of data points: ', len(combined))

for n in range(len(num_par)):
    params = np.loadtxt('params_'+str(num_par[n])+'_'+str(retain)+'.txt')
    params = params.astype(np.float32)
    
    
    
    
    #print(test_set)
    
    
    alphaD, alphaD_test, times = helper.plot_alphaD(combined,params,num_par[n], central_point, retain)
    
    
    plt.figure(3, dpi = 150)
    rel =  np.abs(np.array(alphaD_test)-np.array(alphaD))/np.array(alphaD_test)
    plt.scatter(np.array(times)*1e3, rel, label = 'n(Alg 1) = '+str(num_par[n]), alpha = 0.8, color = colors[n])
    print(n, np.log10(np.mean(rel)))
    
    dat = np.concatenate((np.array(times).reshape(len(times),1), rel.reshape(len(times),1)), axis=1)
    np.savetxt('CAT_Emulator1_'+str(n)+'.txt', dat)
    
    plt.axhline((np.mean(rel)), color = colors[n], ls = '-')
    
    




'''
Optional on the same CAT plot also add the simple emulator
'''
num_par = [3,5,7]


for n in range(len(num_par)):
    params = np.loadtxt('params_'+str(num_par[n])+'_only_alphaD.txt')
    params = params.astype(np.float32)
    
    test_set = []
    with open("train_set.txt", "r") as f:
        for line in f:
            tup = tuple(map(str, line.strip().split(",")))  # Convert back to tuple of integers
            test_set.append(tup)

    
    #print(test_set)
    
    
    alphaD, alphaD_test, times = helper.plot_alphaD_simple(combined,params,num_par[n], central_point)
    
    
    plt.figure(3, dpi = 150)
    rel =  np.abs(np.array(alphaD_test)-np.array(alphaD))/np.array(alphaD_test)
    plt.scatter(np.array(times)*1e3, rel, label = 'n(Alg 2) = '+str(num_par[n]), alpha = 0.8, marker = 'x',
                color = colors[n])
    
    print(n, np.log10(np.mean(rel)))
    
    dat = np.concatenate((np.array(times).reshape(len(times),1), rel.reshape(len(times),1)), axis=1)
    np.savetxt('CAT_Emulator2_'+str(n)+'.txt', dat)
    


    plt.axhline((np.mean(rel)), color = colors[n], ls = '--')
    
plt.legend(frameon = False, ncol = 2, fontsize = 11)

plt.yscale('log')

#plt.xscale('log')
#plt.xlim(3e-4,1e-3)
#plt.xlim(0.3,0.5)
plt.annotate('QRPA $\sim 10^3$ s', xy=(1.1, 5e-6), xytext=(0.8, 5e-6),
              arrowprops=dict(arrowstyle='->'), fontsize=12)
#plt.axvline(1e3)

plt.xlabel('$time$ (ms)', size = 16)
plt.ylabel('Relative error', size = 16)
plt.tight_layout()

plt.gca().tick_params(axis="y",direction="in", which = 'both', labelsize = 12)
plt.gca().tick_params(axis="x",direction="in", which = 'both', labelsize = 12)



#plt.ylim(1e-6,1e-1)
plt.xlim(.5,2.0)
plt.show()
#plt.xscale('log')
#plt.savefig('CAT_dipole.pdf', bbox_inches='tight')






