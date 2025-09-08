#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 10:21:01 2025

@author: anteravlic
"""

import numpy as np
import matplotlib.pyplot as plt
import helper 
import tensorflow as tf

n = 10
params = np.loadtxt('params_'+str(n)+'.txt')

central_point = ('1.2290', '2.7500')
alpha, beta = central_point

'''
Load the strength
'''
strength = np.loadtxt(f'../dipoles_data_all/total_strength/strength_{beta}_{alpha}.out')



opt_D, opt_S1, opt_S2, opt_v0,opt_v1, opt_v2, fold = helper.modified_DS_affine_v(params, n)
eigvals, eigvecs = helper.generalized_eigen(
    opt_D.numpy(), opt_S1.numpy(), opt_S2.numpy(), (alpha, beta), central_point
)

# Save full eigenvalues BEFORE discarding
eigvals_full = eigvals

disc = int((1 - 0.7) * n / 2)
eigvals_trunc = eigvals_full[disc:-disc]
eigvecs = eigvecs[:, disc:-disc]

v_eff = opt_v0 + (float(alpha) - float(central_point[0])) * opt_v1 + (float(beta) - float(central_point[1])) * opt_v2
projections = tf.linalg.matvec(tf.transpose(eigvecs), v_eff)
B = tf.square(projections)

x = strength[:, 0]
orig = strength[:, 1]
lor = helper.give_me_Lorentzian(x, eigvals_trunc, B, fold)

plt.figure(figsize=(8,6))
plt.plot(x, strength[:,1], label='High Fidelity', linewidth=2)
plt.plot(x, lor, '--', label='PMM Emulated', linewidth=2)
plt.xlabel('Energy')
plt.ylabel('Strength')
plt.title('Lorentzian at Central Point')
plt.legend()
plt.grid()
plt.show()
