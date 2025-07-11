# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
from scipy.linalg import eigh, eig
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time
import tensorflow as tf
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import sys
import random as rn
from scipy.interpolate import lagrange
from numpy.polynomial.polynomial import Polynomial
import scipy.integrate as integrate
from scipy.special import gamma
import scipy.integrate as integrate
import os, re


if __name__ == "____":
    print("Running main.py as a script")


# definition of the constants
hqc = 197.33

# Seed for reproducibility
np.random.seed(42)

@tf.function
def give_me_Lorentzian(energy, poles, strength, width):
    if isinstance(energy, np.ndarray):
        energy = tf.convert_to_tensor(energy, dtype=tf.float64)
    
    poles = tf.convert_to_tensor(poles, dtype=tf.float64)
    strength = tf.convert_to_tensor(strength, dtype=tf.float64)
    width = tf.convert_to_tensor(width, dtype=tf.float64)
    
    mask = tf.cast((poles > 3) & (poles < 40), dtype=tf.float64)
    strength = strength * mask
    
    energy_expanded = tf.expand_dims(energy, axis=-1)

    numerator = strength * (width / 2 / np.pi)
    denominator = ((energy_expanded - poles)**2 + (width**2 / 4))
    
    lorentzian = numerator / denominator

    #value = tf.reduce_sum(lorentzian, axis=-1)
    
    return tf.reduce_sum(lorentzian, axis=-1)


# nec_mat for M_true(a) = D + a * S1 + b * S2
def initial_matrix(n):
    D = np.diag(np.random.uniform(1, 10, n))
    A = np.random.uniform(1, 10, (n, n))
    S1 = np.abs(A + A.T) / 2
    S2 = np.abs(A + A.T) / 2
    return D, S1, S2


# def data_table(fmt_data):
#     # Build absolute paths relative to this helper.py file
#     this_dir = os.path.dirname(__file__)
#     data_base = os.path.join(this_dir, 'dipoles_data_all')
#     strength_dir = os.path.join(data_base, 'total_strength')
#     alphaD_dir   = os.path.join(data_base, 'total_alphaD')

#     # Pattern to parse beta and alpha values from filenames
#     pattern = re.compile(r'strength_([0-9.]+)_([0-9.]+)\.out')

#     # Collect all (beta, alpha) pairs
#     for fname in sorted(os.listdir(strength_dir)):
#         match = pattern.match(fname)
#         if match:
#             beta_str, alpha_str = match.groups()
#             fmt_data.append((beta_str, alpha_str))

#     # Load data based on the collected pairs
#     strength = []
#     alphaD   = []
#     for beta_str, alpha_str in fmt_data:
#         file_strength = os.path.join(
#             strength_dir, f'strength_{beta_str}_{alpha_str}.out'
#         )
#         file_alphaD = os.path.join(
#             alphaD_dir,   f'alphaD_{beta_str}_{alpha_str}.out'
#         )
#         strength.append(np.loadtxt(file_strength))
#         alphaD.append(np.loadtxt(file_alphaD))

#     return strength, alphaD

'''
constructing data table for alpha & beta parameters
'''


def data_table(fmt_data):    
    strength_dir = '../dipoles_data_all/total_strength/'
    alphaD_dir = '../dipoles_data_all/total_alphaD/'
    pattern = re.compile(r'strength_([0-9.]+)_([0-9.]+)\.out')
        
    strength = []
    alphaD = []
    
    for fname in sorted(os.listdir(strength_dir)):
        match = pattern.match(fname)
        
        if match:     
            beta_str, alpha_str = match.group(1), match.group(2)
            fmt_data.append((beta_str, alpha_str))
            
            #print(beta_str, alpha_str)
            
    for beta_str, alpha_str in fmt_data:        
        file_strength = os.path.join(strength_dir,
                              f'strength_{beta_str}_{alpha_str}.out')
        file_alphaD = os.path.join(alphaD_dir,
                              f'alphaD_{beta_str}_{alpha_str}.out')

        strength.append(np.loadtxt(file_strength))
        alphaD.append(np.loadtxt(file_alphaD))
        
    return strength, alphaD





def emulator_params(params, n, alpha, beta):
    '''
    
    Parameters:
        - eta0, p1, p2, p3
        - v0: n
        - D: n
        - S1, S2: n(n+1)/2
    
    Returns:
        - eta0, p1, p2, p3 (width params)
        - v0 (external field)
        - D_mod, S1_mod, S2_mod (emulator matrices)
     
    params: tf.Variable, randomized
    
    '''
    # eta0 = params[0]
    # p1, p2, p3 = params[1], params[2], params[3]
    
    eta0 = tf.cast(params[0], dtype=tf.float64)
    p1 = tf.cast(params[1], dtype=tf.float64)
    p2 = tf.cast(params[2], dtype=tf.float64)
    p3 = tf.cast(params[3], dtype=tf.float64)
    
    alpha_t = tf.cast(alpha, dtype=tf.float64)
    beta_t = tf.cast(beta,  dtype=tf.float64)
    term = p1 + p2 * alpha_t + p3 * beta_t
    width = tf.sqrt(eta0**2 + term**2)
    
    D_shape = (n,n)
    S1_shape = (n,n)
    S2_shape = (n,n)
    
    
    v0_mod = tf.convert_to_tensor(params[4 : n+4], dtype = tf.float64)
    
    D_mod = tf.linalg.diag(params[n+4 : 2*n+4])
   
    S1_mod = tf.zeros(S1_shape, dtype=tf.float64)
    upper_tri_indices1 = np.triu_indices(n)
    indices1 = tf.constant(list(zip(upper_tri_indices1[0], upper_tri_indices1[1])))
    S1_mod = tf.tensor_scatter_nd_update(S1_mod,
                                         indices1,
                                         params[2*n+4
                                                : 2*n+4 + len(upper_tri_indices1[0])])
    S1_mod = S1_mod + tf.linalg.band_part(tf.transpose(S1_mod), -1, 0) \
        - tf.linalg.diag(tf.linalg.diag_part(S1_mod))
        
    S2_mod = tf.zeros(S2_shape, dtype=tf.float64)
    upper_tri_indices2 = np.triu_indices(n)
    indices2 = tf.constant(list(zip(upper_tri_indices2[0], upper_tri_indices2[1])))
    S2_mod = tf.tensor_scatter_nd_update(S2_mod,
                                         indices2,
                                         params[2*n+4 + len(upper_tri_indices1[0])
                                                            : 2*n+4+len(upper_tri_indices1[0]) + len(upper_tri_indices2[0])])
    S2_mod = S2_mod + tf.linalg.band_part(tf.transpose(S2_mod), -1, 0) \
        - tf.linalg.diag(tf.linalg.diag_part(S2_mod))
    
    # tf.print("n =", n,
    #      "S1_shape =", S1_shape,
    #      "len(indices1) =", tf.shape(indices1),
    #      "updates1.shape =", tf.shape(params[2*n+4 : 2*n+4 + len(indices1)]))
    
    
    return eta0, p1, p2, p3, width, v0_mod, D_mod, S1_mod, S2_mod






@tf.function
def calculate_alphaD(eigvals, B):
    
    mask = tf.cast((eigvals > 3) & (eigvals < 40), dtype=tf.float64)
    B = B * mask
    
    fac = tf.constant(8.0*np.pi*(7.29735e-3)*hqc/9.0 , dtype = tf.float64)
    val = tf.reduce_sum(B / eigvals)*fac
    
    return val
    






# cost_function
def cost_function(params, n, fmt_data, strength, alphaD, weight):
    
    '''
    params: tf.Variable
    D_shape, S1_shape, S2_shape : int
    alpha_values, beta_values: list
    
    calculates the cost function by subtracting two Lorentzians
    and also alphaD
    
    '''
        
    cost = tf.constant(0.0, dtype = tf.float64)
    
    idx = 0
    
    alphaD_calc = []
    
    for idx, (beta_str, alpha_str) in enumerate(fmt_data):
        beta = float(beta_str)
        alpha = float(alpha_str)
        
        eta0, p1, p2, p3, width, v0_mod, D_mod, S1_mod, S2_mod = \
            emulator_params(params, n, alpha, beta)
        
        
        M_true = D_mod + beta*S1_mod + alpha*S2_mod
    
        eigvals, eigvecs = tf.linalg.eigh(M_true)
        
        # Compute dot product of each eigenvector (columns) with v0_mod
        proj = tf.linalg.matvec(tf.transpose(eigvecs), v0_mod)
        strengths = tf.square(proj)
        mask = tf.cast((eigvals > 3) & (eigvals < 40), dtype=tf.float64)
        #B = B * mask

        # values from QFAM
        omega = tf.constant(strength[idx][:,0], dtype=tf.float64)
        Lor_true = tf.constant(strength[idx][:,1], dtype=tf.float64)

        # Use tf.map_fn to apply the give_me_Lorentzian function over the x values
        Lor = give_me_Lorentzian(omega, eigvals, strengths, width)
        cost += tf.reduce_sum((Lor - Lor_true) ** 2)
        
        val_true = alphaD[idx][2]
        val = calculate_alphaD(eigvals, strengths)
        cost += (val_true - val)**2*weight
        alphaD_calc.append(val)

        #idx+=1
            
    return cost, Lor, Lor_true, omega, alphaD_calc




# generalized eigen for M_true(a)
#@tf.function
def generalized_eigen(D, S1, S2, alpha):
    M_true = D + float(alpha[0]) * S1 + float(alpha[1]) * S2
    eigvals, eigvecs = eigh(M_true)
    return eigvals, eigvecs




def plot_Lorentzian_for_idx(idx, train_set, n, params):

    # beta = float(train_set[idx][0])
    # alpha = float(train_set[idx][1])
    
    beta, alpha = map(float, train_set[idx])
    
    Lor_train, alphaD_train = data_table(train_set)
    omega_fom, S_fom = Lor_train[idx][:,0], Lor_train[idx][:,1]
    
    
    
    #Lors_orig = Lors_train[idx]
    
    #opt_D, opt_S1, opt_S2, opt_v0, fold = modified_DS(params, n)
    eta0, p1, p2, p3, width, v0_mod, D_mod, S1_mod, S2_mod = emulator_params(params, n, alpha, beta)
    
    eigvals, eigvecs = generalized_eigen(
        D_mod.numpy(), S1_mod.numpy(), S2_mod.numpy(), (beta, alpha)
    )
    
    proj = tf.linalg.matvec(tf.transpose(eigvecs), v0_mod)
    strengths = tf.square(proj)
    mask = tf.cast((eigvals > 3) & (eigvals < 40), dtype=tf.float64)
    #B = strengths #* mask; are we still doing this mask?
    
    #fig, ax = plt.subplots()
    
    # plot the Lorentzian for the original data
    #omega = Lors_orig[:,0]
    
    
    emu_Lor = give_me_Lorentzian(
        omega_fom,
        eigvals,
        strengths,
        width
    )
    
#     #beta, alpha = train_set[idx]

# # Unpack emulator parameters including computed width
#     eta0, p1, p2, p3, width, v0_mod, D_mod, S1_mod, S2_mod = emulator_params(params, n, alpha, beta)
    
#     # Solve the generalized eigenproblem
#     eigvals, eigvecs = generalized_eigen(
#         D_mod.numpy(), S1_mod.numpy(), S2_mod.numpy(), train_set[idx]
#     )
    
#     # Compute transition strengths
#     projections = tf.linalg.matvec(tf.transpose(eigvecs), v0_mod)
#     strengths = tf.square(projections) * tf.cast((eigvals > 3) & (eigvals < 40), tf.float64)
    
#     # Retrieve original QRPA data
#     Lors_train, alphaD_train = data_table(train_set)
#     omega_orig = Lors_train[idx][:,0]
#     S_orig     = Lors_train[idx][:,1]
    
#     # Generate emulated Lorentzian
#     emu_L = give_me_Lorentzian(omega_orig, eigvals, strengths, width)
    
    # Plot
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(omega_fom, S_fom, 'b--', label='FAM QRPA calculation')
    ax.plot(omega_fom, emu_Lor.numpy(), 'r-', label='Emulated Lorentzian')
    ax.set_title(r'$b_{{TV}}={:.1f},\; d_{{TV}}={:.1f}$'.format(alpha, beta), size=14)
    ax.set_xlabel(r'$\omega$ (MeV)', size=12)
    ax.set_ylabel(r'$S$ (e$^2$ fm$^2$/MeV)', size=12)
    ax.legend(frameon=False)
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 20)
    plt.show()

    
    
    
    
    # plt.plot(omega, Lors_orig[:,1], 'b--',label='FAM QRPA calculation')
    
    # plt.plot(omega, opt_Lor, 'r-',label='emulated Lorentzian')    
    # ax.set_title(r'$b_{TV}$ = '+str(round(alpha,1))+r', $d_{TV} = $'+str(round(beta,1)), size = 18)
    # ax.legend(frameon = False)
        
    # plt.xlabel('$\omega$ (MeV)', size = 18)
    # plt.ylabel('$S$ (e$^2$ fm$^2$/MeV)', size = 18)
    # plt.annotate('${}^{180}$Yb', (0.7,0.7), xycoords='axes fraction', size = 18)

    # plt.gca().tick_params(axis="y",direction="in", which = 'both', labelsize = 12)
    # plt.gca().tick_params(axis="x",direction="in", which = 'both', labelsize = 12)
    
    # plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    # plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(1))
        
    # plt.xlim(0, 40)
    # plt.ylim(0, 10)
    
    return




def data_Lorentzian_for_idx(idx, train_set, n, params):
    # alpha = float(test_set[idx][0])
    # beta = float(test_set[idx][1])
    
    beta, alpha = map(float, train_set[idx])
    
    Lor_train, alphaD_train = data_table(train_set)
    omega_fom, S_fom = Lor_train[idx][:,0], Lor_train[idx][:,1]
    #opt_D, opt_S1, opt_S2, opt_v0, fold = modified_DS(params, n)
    eta0, p1, p2, p3, width, v0_mod, D_mod, S1_mod, S2_mod = emulator_params(params, n, alpha, beta)
    eigvals, eigvecs = generalized_eigen(
        D_mod.numpy(), S1_mod.numpy(), S2_mod.numpy(), (beta, alpha)
    )
    #opt_eigvals, opt_eigvecs = generalized_eigen(opt_D.numpy(), opt_S1.numpy(), opt_S2.numpy(), test_set[idx])
    
    proj = tf.linalg.matvec(tf.transpose(eigvecs), v0_mod)
    strengths = tf.square(proj)    
    mask = tf.cast((eigvals > 3) &  (eigvals < 40), dtype=tf.float64)

    # Apply the mask to zero out B where eigenvalue is negative
    #opt_dot_products = B #* mask
    
    #omega = Lors_orig[:,0]
        
    emu_Lor = give_me_Lorentzian(
        omega_fom,
        eigvals,
        strengths,
        width
    )
    
    return omega_fom, S_fom, emu_Lor
    




 
def plot_alphaD(idx, train_set, params, n): 
    emu_alphaD = []
    times = []
    
    beta, alpha = map(float, train_set[idx])
    
    Lor_train, alphaD_train = data_table(train_set)
    alphaD_train = np.vstack(alphaD_train)
    alphaD_train = alphaD_train[:,2]
    
    for idx in range(len(train_set)):
        start = time.time()  # Start time
        #opt_D, opt_S1, opt_S2, opt_v0, fold = modified_DS(params, n)
        eta0, p1, p2, p3, width, v0_mod, D_mod, S1_mod, S2_mod = emulator_params(params, n, alpha, beta)
        
        # M_true = D_mod + float(train_set[idx][0]) * S1_mod + float(train_set[idx][1]) * S2_mod
        # opt_eigvals, opt_eigvecs = tf.linalg.eigh(M_true)
        
        eigvals, eigvecs = generalized_eigen(
            D_mod.numpy(), S1_mod.numpy(), S2_mod.numpy(), (beta, alpha)
        )
        
        #opt_eigenvalues, opt_eigenvectors = generalized_eigen(opt_D.numpy(), opt_S1.numpy(), opt_S2.numpy(), test_set[idx])
        proj = tf.linalg.matvec(tf.transpose(eigvecs), v0_mod)
        strengths = tf.square(proj)
        #mask = tf.cast((opt_eigenvalues > 1) &  (opt_eigenvalues < 30), dtype=tf.float64)
        #B = B * mask
        end = time.time()  # Start time
        emu_alphaD.append(calculate_alphaD(eigvals, strengths))
        times.append(end - start)
    return emu_alphaD, alphaD_train, times
