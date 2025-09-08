#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 17:34:32 2025

@author: anteravlic
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test just the central point

This is just a demonstration that has to be implemented into the main code.

"""


'''
This is added to compute a central data point
'''

import numpy as np
import helper
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import tensorflow as tf
import os
import re
from matplotlib.colors import LogNorm
import matplotlib.patches as patches
from scipy.optimize import least_squares, nnls

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
        #if ((alpha, beta)) not in test_set and ((alpha, beta)) not in cv_set:
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
Find the central point on the dataset
'''

# combined: list of (alpha, beta) points
Amin = min(float(a) for a, b in combined)
Amax = max(float(a) for a, b in combined)
Bmin = min(float(b) for a, b in combined)
Bmax = max(float(b) for a, b in combined)

cx = (Amin + Amax) / 2.0
cy = (Bmin + Bmax) / 2.0

innermost = min(combined, key=lambda t: (float(t[0]) - cx)**2 + (float(t[1]) - cy)**2)
print(innermost)

data = np.loadtxt(strength_dir+'strength_'+innermost[1]+'_'+innermost[0]+'.out')
plt.plot(data[:,0], data[:,1])

'''
Here you will do the fit
'''



# ---------- small helpers ----------
def _softplus(x):            # numpy-friendly for init
    x = np.asarray(x, dtype=float)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)

def _inv_softplus(y):
    y = np.maximum(np.asarray(y, dtype=float), 1e-12)
    return np.log(np.expm1(y))

# Enforce E strictly increasing and B >= 0 via reparameterization (done in TF)
def _unpack_params_tf(z, n, wmin, min_spacing):
    z = tf.convert_to_tensor(z, tf.float32)
    zE, zB = z[:n], z[n:]

    e0 = wmin + tf.nn.softplus(zE[0])                       # E1 >= wmin
    gaps = tf.nn.softplus(zE[1:]) + min_spacing             # positive gaps
    E = tf.concat([e0[None], e0 + tf.cumsum(gaps)], axis=0) # (n,)

    B = tf.nn.softplus(zB) ** 2                             # >=0 (extra margin)
    return E, B

def _pack_init_np(E0, B0, wmin, min_spacing):
    zE = np.empty_like(E0, dtype=float)
    zE[0] = _inv_softplus(E0[0] - wmin)
    gaps = np.diff(E0)
    zE[1:] = _inv_softplus(np.maximum(gaps - min_spacing, 1e-6))
    zB = _inv_softplus(np.sqrt(np.maximum(B0, 1e-12)))
    return np.concatenate([zE, zB])

# ---------- main fit (uses TF inside the residual) ----------
def fit_strength_with_tf_lorentzian(omega, y, n, eta,
                                    grid_M=None, min_spacing=0.2, l2=0.0):
    """
    omega, y : (G,) arrays (measured strength)
    n        : number of Lorentzians
    eta      : fixed width (float)
    grid_M   : number of NNLS grid centers (default=len(omega))
    min_spacing : minimal spacing between centers during refine (MeV)
    l2       : small Tikhonov on raw z (0 disables)

    Returns: E_hat (n,), B_hat (n,), y_hat (G,)
    """
    omega_np = np.asarray(omega, float)
    y_np = np.asarray(y, float)
    wmin, wmax = float(omega_np.min()), float(omega_np.max())
    if grid_M is None:
        grid_M = len(omega_np)

    # ---- Stage 1: NNLS seed on a dense center grid ----
    E_grid = np.linspace(wmin + 1e-6, wmax - 1e-6, grid_M)
    A = 1.0 / ((omega[:, None] - E_grid[None, :])**2 + (eta**2)/4.0) * (eta/(2*np.pi))
    coeff, _ = nnls(A, y)
    idx = np.argsort(coeff)[-n:]
    E0 = np.sort(E_grid[idx])
    B0 = coeff[idx][np.argsort(E_grid[idx])]

    # ensure minimal spacing in seed
    for k in range(1, n):
        if E0[k] - E0[k-1] < min_spacing:
            E0[k] = E0[k-1] + min_spacing
    z0 = _pack_init_np(E0, B0, wmin, min_spacing)

    # ---- Stage 2: nonlinear refine using TF lorentzian in the residual ----
    def residuals(z):
        # unpack in TF, evaluate model with your TF lorentzian, return numpy residuals
        E_tf, B_tf = _unpack_params_tf(z, n, tf.constant(wmin, tf.float32),
                                       tf.constant(min_spacing, tf.float32))
        yhat_tf = helper.give_me_Lorentzian(omega, E_tf, B_tf, tf.constant(eta, tf.float32))
        r = (yhat_tf.numpy() - y)
        if l2 > 0:
            r = np.concatenate([r, np.sqrt(l2) * np.asarray(z, dtype=float)])
        return r

    res = least_squares(residuals, z0, method="trf",
                        max_nfev=5000, xtol=1e-10, ftol=1e-10, gtol=1e-10)

    # unpack final parameters and compute final curve (all through TF)
    E_tf, B_tf = _unpack_params_tf(res.x, n, tf.constant(wmin, tf.float32),
                                   tf.constant(min_spacing, tf.float32))
    yhat_tf = helper.give_me_Lorentzian(omega, E_tf, B_tf, tf.constant(eta, tf.float32))

    E_hat = E_tf.numpy()
    B_hat = B_tf.numpy()
    y_hat = yhat_tf.numpy()
    return E_hat, B_hat, y_hat


omega = data[:,0]
y = data[:,1]
n, eta = 12, 2.0
omega_tf = tf.convert_to_tensor(omega, dtype=tf.float32)
y_tf = tf.convert_to_tensor(y, dtype=tf.float32)
eta_tf = tf.convert_to_tensor(eta, dtype=tf.float32)
E_hat, B_hat, y_hat = fit_strength_with_tf_lorentzian(omega_tf, y_tf, n, eta_tf, min_spacing=0.01)

plt.plot(data[:,0], y_hat)
plt.stem(E_hat, B_hat)


