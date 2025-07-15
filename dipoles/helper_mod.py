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
from numpy.random import default_rng
from scipy.interpolate import lagrange
from numpy.polynomial.polynomial import Polynomial
import scipy.integrate as integrate
from scipy.special import gamma
import scipy.integrate as integrate
import os, re
from pathlib import Path


if __name__ == "____":
    print("Running main.py as a script")


# definition of the constants
hqc = 197.33

# Seed for reproducibility
rng = default_rng(42)

@tf.function
def give_me_Lorentzian(energy, poles, strength, width):
    if isinstance(energy, np.ndarray):
        energy = tf.convert_to_tensor(energy, dtype=tf.float64)
    
    poles = tf.convert_to_tensor(poles, dtype=tf.float64)
    strength = tf.convert_to_tensor(strength, dtype=tf.float64)
    #width = tf.convert_to_tensor(width, dtype=tf.float64) #####
    
    mask = tf.cast((poles > 3) & (poles < 40), dtype=tf.float64)
    strength = strength * mask
    
    energy_expanded = tf.expand_dims(energy, axis=-1)

    numerator = strength * (width / 2 / np.pi)
    denominator = ((energy_expanded - poles)**2 + (width**2 / 4))
    
    lorentzian = numerator / denominator

    value = tf.reduce_sum(lorentzian, axis=-1)
    
    return value


# nec_mat for M_true(a) = D + a * S1 + b * S2
def initial_matrix(n):
    D = np.diag(rng.uniform(1, 10, n))
    A = rng.uniform(1, 10, (n, n))
    S1 = np.abs(A + A.T) / 2
    S2 = np.abs(A + A.T) / 2
    return D, S1, S2



'''
constructing data table for alpha & beta parameters
'''
#strength_list, alphaD_list = data_table()

# def data_table(fmt_data):    
#     strength_dir = '../dipoles_data_all/total_strength/'
#     alphaD_dir = '../dipoles_data_all/total_alphaD/'
#     pattern = re.compile(r'strength_([0-9.]+)_([0-9.]+)\.out')
        
#     strength = []
#     alphaD = []
    
#     for fname in sorted(os.listdir(strength_dir)):
#         match = pattern.match(fname)
        
#         if match:     
#             beta_str, alpha_str = match.group(1), match.group(2)
#             fmt_data.append((beta_str, alpha_str))
            
#             #print(beta_str, alpha_str)
            
#     for beta_str, alpha_str in fmt_data:        
#         file_strength = os.path.join(strength_dir,
#                               f'strength_{beta_str}_{alpha_str}.out')
#         file_alphaD = os.path.join(alphaD_dir,
#                               f'alphaD_{beta_str}_{alpha_str}.out')

#         strength.append(np.loadtxt(file_strength))
#         alphaD.append(np.loadtxt(file_alphaD))
        
#     return strength, alphaD



_strength_dir = Path("../dipoles_data_all/total_strength")
_alphaD_dir   = Path("../dipoles_data_all/total_alphaD")
_pattern      = re.compile(r"strength_([0-9.]+)_([0-9.]+)\.out")

def beta_alpha_pairs(min_beta=None, max_beta=None,
                     min_alpha=None, max_alpha=None):
    """
    Discover all (beta, alpha) filename pairs in the data directories,
    optionally filtering by numeric ranges.

    Returns
    -------
    List of tuples of strings: [(beta_str, alpha_str), ...]
    """
    pairs = []
    for fname in sorted(os.listdir(_strength_dir)):
        m = _pattern.match(fname)
        if not m:
            continue
        b, a = float(m.group(1)), float(m.group(2))
        if min_beta is not None and not (min_beta <= b <= max_beta): continue
        if min_alpha is not None and not (min_alpha <= a <= max_alpha): continue
        pairs.append((m.group(1), m.group(2)))
    return pairs


def data_table(pairs=None):
    """
    Load dipole strength and alphaD data arrays for each (beta, alpha) pair.

    Parameters
    ----------
    pairs : list of (beta_str, alpha_str) tuples; if None, loads all.

    Returns
    -------
    strength_list : list of ndarray
    alphaD_list   : list of ndarray
    """
    if pairs is None:
        pairs = beta_alpha_pairs()
    strength_list, alphaD_list = [], []
    for b_str, a_str in pairs:
        f1 = _strength_dir / f"strength_{b_str}_{a_str}.out"
        f2 = _alphaD_dir   / f"alphaD_{b_str}_{a_str}.out"
        strength_list.append(np.loadtxt(f1))
        alphaD_list.append(np.loadtxt(f2))
    return strength_list, alphaD_list


def load_omega_and_strength(max_files=None):
    """
    Quickly load the first-column (omega) and stack the strength columns into a matrix.
    Parameters
    ----------
    max_files : int or None, number of spectra to load (in sorted order)
    Returns
    -------
    omega : ndarray, shape (L,)
    S     : ndarray, shape (K, L)
    """
    files = sorted(_strength_dir.glob("*.out"))
    if max_files is not None:
        files = files[:max_files]
    omega = np.loadtxt(files[0])[:, 0]
    strength = np.vstack([np.loadtxt(f)[:, 1] for f in files])
    return omega, strength


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
def cost_function(params, n, pairs, strength_list, alphaD_list, weight):
    
    '''
    params: tf.Variable
    D_shape, S1_shape, S2_shape : int
    alpha_values, beta_values: list
    
    calculates the cost function by subtracting two Lorentzians
    and also alphaD
    
    '''
    
    print("Entering cost_function with", len(pairs), "samples.")

        
    #cost = tf.constant(0.0, dtype = tf.float64)
    cost = 0
    
    idx = 0
    
    alphaD_calc = []
    
    for idx, (beta_str, alpha_str) in enumerate(pairs):
        beta = float(beta_str)
        alpha = float(alpha_str)
        
        # print(f"idx={idx},"
        #       f"β={beta}, α={alp"ha},"
        #       f"strength_list[idx].shape={strength_list[idx].shape},"
        #       f"alphaD_list[idx].shape={alphaD_list[idx].shape}")
        
        
        eta0, p1, p2, p3, width, v0_mod, D_mod, S1_mod, S2_mod = \
            emulator_params(params, n, alpha, beta)
        
        M_true = D_mod + alpha*S1_mod + beta*S2_mod
    
        eigvals, eigvecs = tf.linalg.eigh(M_true)
        
        # Compute dot product of each eigenvector (columns) with v0_mod
        proj = tf.linalg.matvec(tf.transpose(eigvecs), v0_mod)
        strengths = tf.square(proj)
        mask = tf.cast((eigvals > 3) & (eigvals < 40), dtype=tf.float64)
        #strengths = strengths * mask

        # values from QFAM
        omega = tf.constant(strength_list[idx][:,0], dtype=tf.float64)
        Lor_true = tf.constant(strength_list[idx][:,1], dtype=tf.float64)

        # Use tf.map_fn to apply the give_me_Lorentzian function over the x values
        Lor = give_me_Lorentzian(omega, eigvals, strengths, width)
        cost += tf.reduce_sum((Lor - Lor_true) ** 2)
        
        val_true = alphaD_list[idx][2]
        val = calculate_alphaD(eigvals, strengths)
        cost += (val_true - val)**2*weight
        alphaD_calc.append(val)

        idx+=1
            
    return cost, Lor, Lor_true, omega, alphaD_calc




# generalized eigen for M_true(a)
#@tf.function
def generalized_eigen(D, S1, S2, alpha):
    M_true = D + float(alpha[1]) * S1 + float(alpha[0]) * S2
    eigvals, eigvecs = eigh(M_true)
    return eigvals, eigvecs




def plot_Lorentzian_for_idx(idx, train_set, n, params):

    beta, alpha = map(float, train_set[idx])
    
    Lor_train, alphaD_train = data_table(train_set)
    omega_fom, S_fom = Lor_train[idx][:,0], Lor_train[idx][:,1]
    
    eta0, p1, p2, p3, width, v0_mod, D_mod, S1_mod, S2_mod = emulator_params(params, n, alpha, beta)
    
    eigvals, eigvecs = generalized_eigen(
        D_mod.numpy(), S1_mod.numpy(), S2_mod.numpy(), (beta, alpha)
    )
    
    proj = tf.linalg.matvec(tf.transpose(eigvecs), v0_mod)
    strengths = tf.square(proj)
    mask = tf.cast((eigvals > 3) & (eigvals < 40), dtype=tf.float64)
    #B = strengths #* mask; are we still doing this mask?
    
    emu_Lor = give_me_Lorentzian(
        omega_fom,
        eigvals,
        strengths,
        width
    )

    # Plot
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(omega_fom, S_fom, 'black', label='FAM QRPA calculation')
    ax.plot(omega_fom, emu_Lor.numpy(), 'r--', label='Emulated Lorentzian')
    
    
    # — green poles at each eigenvalue with height = discrete strength —
    ev = eigvals
    st = strengths.numpy() if hasattr(strengths, "numpy") else strengths
    sel = (ev > 3) & (ev < 40)
    ev = ev[sel]
    st = st[sel]
    ax.vlines(ev, 0, st, color='g', alpha=0.7, linewidth=1)
    ax.scatter(ev, st, color='g', s=30, zorder=5)
    
    #ax.set_title(r'$alpha b_{{TV}}={:.4f},\; beta d_{{TV}}={:.4f}$'.format(alpha, beta), size=14)
    ax.set_title(r'$\alpha={:.4f},\; \beta={:.4f}$'.format(alpha, beta), size=14)
    ax.set_xlabel(r'$\omega$ (MeV)', size=12)
    ax.set_ylabel(r'$S$ (e$^2$ fm$^2$/MeV)', size=12)
    ax.legend(frameon=False)
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 20)
    plt.show()
    
    return




def data_Lorentzian_for_idx(idx, train_set, n, params):

    beta, alpha = map(float, train_set[idx])
    
    Lor_train, alphaD_train = data_table(train_set)
    omega_fom, S_fom = Lor_train[idx][:,0], Lor_train[idx][:,1]

    eta0, p1, p2, p3, width, v0_mod, D_mod, S1_mod, S2_mod = emulator_params(params, n, alpha, beta)
    eigvals, eigvecs = generalized_eigen(
        D_mod.numpy(), S1_mod.numpy(), S2_mod.numpy(), (beta, alpha)
    )
    
    proj = tf.linalg.matvec(tf.transpose(eigvecs), v0_mod)
    strengths = tf.square(proj)    
    mask = tf.cast((eigvals > 3) &  (eigvals < 40), dtype=tf.float64)

    #opt_dot_products = B #* mask
            
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

        eta0, p1, p2, p3, width, v0_mod, D_mod, S1_mod, S2_mod = emulator_params(params, n, alpha, beta)
        
        eigvals, eigvecs = generalized_eigen(
            D_mod.numpy(), S1_mod.numpy(), S2_mod.numpy(), (beta, alpha)
        )
        
        proj = tf.linalg.matvec(tf.transpose(eigvecs), v0_mod)
        strengths = tf.square(proj)
        #mask = tf.cast((opt_eigenvalues > 1) &  (opt_eigenvalues < 30), dtype=tf.float64)
        #B = B * mask
        end = time.time()  # Start time
        emu_alphaD.append(calculate_alphaD(eigvals, strengths))
        times.append(end - start)
        
    return emu_alphaD, alphaD_train, times
