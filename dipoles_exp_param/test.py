#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 16:14:50 2025

@author: anteravlic
"""

import tensorflow as tf
import helper
import numpy as np

n = 20
weight=10

D, S1, S2 = helper.nec_mat(n)
print(D.shape, S1.shape, S2.shape)


# initialize the external field vector
v0 = np.random.rand(n)

'''
    data table is now constructed for alpha & beta parameters
'''


nec_num_param = n + 2*int(n * (n + 1) / 2) + n # add v0 to the mix (last n)
print('Number of parameters: ', nec_num_param)
random_initial_guess = np.random.uniform(0, 1, nec_num_param)

params_shape = [D.shape, S1.shape, S2.shape]

params = tf.Variable(random_initial_guess, dtype=tf.float64)

D_mod, S1_mod, S2_mod, v0_mod = helper.modified_DS(params,  n)

M_true = D_mod + float(1.2) * S1_mod + float(0.5) * S2_mod


# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = tf.linalg.eigh(M_true)

# Print all eigenvalues
print("All eigenvalues:", eigenvalues.numpy())
print("All eigenvectors:", eigenvectors.numpy())

# Filter: Keep only positive eigenvalues and their corresponding eigenvectors
positive_mask = eigenvalues > 0
eigenvalues_pos = tf.boolean_mask(eigenvalues, positive_mask)
eigenvectors_pos = tf.boolean_mask(eigenvectors, positive_mask, axis=1)

# Output results
print("Positive eigenvalues:", eigenvalues_pos.numpy(), eigenvalues_pos.shape)
print("Corresponding eigenvectors:\n", eigenvectors_pos.numpy(), eigenvectors_pos.shape)

B = [tf.square(tf.tensordot(eigenvectors_pos[:, i], v0_mod, axes=1)) for i in range(eigenvectors_pos.shape[1])]




