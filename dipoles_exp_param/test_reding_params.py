#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 10:39:52 2025

@author: anteravlic
"""
import numpy as np
import helper
import tensorflow as tf


n = 5
nec_num_param = 1 + 3*n + n + 2 * int(n * (n + 1) / 2)

params_test = np.random.uniform(0, 10, nec_num_param)
params_test_tf = tf.convert_to_tensor(params_test, dtype=tf.float64)

D_mod, S1_mod, S2_mod, v0_mod, v1_mod, v2_mod, eta = helper.modified_DS_affine_v(params_test_tf, n)

print("eta:", eta.numpy())
print("v0_mod:", v0_mod.numpy())
print("v1_mod:", v1_mod.numpy())
print("v2_mod:", v2_mod.numpy())
print("D_mod shape:", D_mod.shape)
print("S1_mod shape:", S1_mod.shape)
print("S2_mod shape:", S2_mod.shape)

# Test v(alpha,beta)
alpha = 1.5
beta = 2.0
alpha_0 = 1.0
beta_0 = 1.0

v_eff = v0_mod + (alpha - alpha_0) * v1_mod + (beta - beta_0) * v2_mod
print("v_eff:", v_eff.numpy())
