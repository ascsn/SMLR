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
import scipy.integrate as integrate
import os
import re
from scipy.optimize import least_squares, nnls


# definition of the constants
hqc = 197.33


def print_all_matrices_from_params(params, n):
    """
    Given a packed parameter vector (as in modified_DS_affine_v),
    reconstruct and print all matrices and vectors.
    """
    from helper import modified_DS_affine_v  # adjust import if needed

    D, S1, S2, S3, S4, v0, v1, v2, eta, x1, x2, x3, x4 = modified_DS_affine_v(params, n)

    # Convert tensors to numpy for clean printing
    def to_np(x): return x.numpy() if tf.is_tensor(x) else x

    print("eta0 =", to_np(eta))
    # print("x1 =", to_np(x1))
    # print("x2 =", to_np(x2))
    # print("x3 =", to_np(x3))
    # print("x4 =", to_np(x4))
    print()

    print("v0 =", to_np(v0))
    # print("v1 =", to_np(v1))
    # print("v2 =", to_np(v2))
    print()

    print("D =\n", to_np(D))
    # print("S1 =\n", to_np(S1))
    # print("S2 =\n", to_np(S2))
    # print("S3 =\n", to_np(S3))
    # print("S4 =\n", to_np(S4))


# def encode_initial_guess(random_initial_guess, E, B, n):
    
#     num_upper = n * (n + 1) // 2
    
#     # --- overwrite only v0 and D with your fitted values ---
#     idx = 0
#     # eta (leave as randomized)
#     idx += 1
    
#     # v0 := sqrt(B_fit)
#     random_initial_guess[idx:idx+n] = np.sqrt(np.maximum(B, 0.0)).astype(np.float32)
#     idx += n
    
#     # v1 (leave)
#     idx += n
    
#     # v2 (leave)
#     idx += n
    
#     # D diagonal := E_fit
#     random_initial_guess[idx:idx+n] = E
#     idx += n
    
#     # S1 upper (leave)
#     idx += num_upper
    
#     # S2 upper (leave)
#     idx += num_upper
    
#     # S3 upper (leave)
#     idx += num_upper
    
#     # S4 upper (leave)
#     idx += num_upper
    
#     # x1,x2,x3,x4 (leave)
#     # params[idx:idx+4] already randomized
    
#     # Now pass `params` to your training / model code
#     return random_initial_guess


def encode_initial_guess(random_initial_guess, E, B, n, retain):
    """
    Centered retain: place k=round(retain*n) fitted (E,B) in the middle of the n-diagonal.
    D outside the kept block is filled by +/-2 stepping:
      left side:  min(E_sel) - 2, -4, ...
      right side: max(E_sel) + 2, +4, ...
    v0 outside kept block is zero.

    Parameters
    ----------
    random_initial_guess : np.ndarray (float32)
        Base packed parameter vector.
    E, B : array-like
        Fitted energies and strengths (length >= k_keep).
    n : int
        Matrix dimension.
    retain : float in (0,1]
        Fraction to retain.

    Returns
    -------
    params : np.ndarray (float32)
    """
    n = int(n)
    params = np.asarray(random_initial_guess, dtype=np.float32).copy()

    # how many to keep (centered)
    k_keep = int(round(float(retain) * n))
    k_keep = max(1, min(k_keep, n))
    left  = (n - k_keep) // 2
    right = left + k_keep

    # sort by energy and take centered slice of fitted set
    E = np.asarray(E, dtype=np.float32).reshape(-1)
    B = np.asarray(B, dtype=np.float32).reshape(-1)
    order = np.argsort(E)
    E, B = E[order], B[order]

    if len(E) < k_keep:
        raise ValueError(f"E,B need at least k_keep={k_keep} entries (got {len(E)}).")

    m = len(E)
    start = (m - k_keep) // 2
    E_sel = E[start:start + k_keep]
    B_sel = B[start:start + k_keep]

    # ---- build full D: kept center + +/-2 stepping on ends ----
    D_full = np.empty(n, dtype=np.float32)
    D_full[left:right] = E_sel

    step = 2.0
    # left side: decreasing by step from min(E_sel)
    cur = E_sel[0]
    for i in range(left - 1, -1, -1):
        cur -= step
        D_full[i] = cur
    # right side: increasing by step from max(E_sel)
    cur = E_sel[-1]
    for i in range(right, n):
        cur += step
        D_full[i] = cur

    # ---- v0: sqrt(B) in kept block, zeros elsewhere ----
    v0_full = np.zeros(n, dtype=np.float32)
    v0_full[left:right] = np.sqrt(np.maximum(B_sel, 0.0)).astype(np.float32)

    # ---- write back into packed vector (your layout) ----
    num_upper = n * (n + 1) // 2
    idx = 0
    idx += 1                                   # eta (keep as-is)
    params[idx:idx + n] = v0_full; idx += n    # v0
    idx += n                                   # v1 (leave)
    idx += n                                   # v2 (leave)
    params[idx:idx + n] = D_full; idx += n     # D diagonal
    idx += num_upper                           # S1 (leave)
    idx += num_upper                           # S2 (leave)
    idx += num_upper                           # S3 (leave)
    idx += num_upper                           # S4 (leave)
    # x1..x4 unchanged

    return params




'''
List of helper functions for the central point
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
        yhat_tf = give_me_Lorentzian(omega, E_tf, B_tf, tf.constant(eta, tf.float32))
        r = (yhat_tf.numpy() - y)
        if l2 > 0:
            r = np.concatenate([r, np.sqrt(l2) * np.asarray(z, dtype=float)])
        return r

    res = least_squares(residuals, z0, method="trf",
                        max_nfev=5000, xtol=1e-10, ftol=1e-10, gtol=1e-10)

    # unpack final parameters and compute final curve (all through TF)
    E_tf, B_tf = _unpack_params_tf(res.x, n, tf.constant(wmin, tf.float32),
                                    tf.constant(min_spacing, tf.float32))
    yhat_tf = give_me_Lorentzian(omega, E_tf, B_tf, tf.constant(eta, tf.float32))

    E_hat = E_tf.numpy()
    B_hat = B_tf.numpy()
    y_hat = yhat_tf.numpy()
    return E_hat, B_hat, y_hat



@tf.function
def give_me_Lorentzian(energy, poles, strength, width):
    if isinstance(energy, np.ndarray):
        energy = tf.convert_to_tensor(energy, dtype=tf.float32)

    poles = tf.convert_to_tensor(poles, dtype=tf.float32)
    strength = tf.convert_to_tensor(strength, dtype=tf.float32)
    
    #mask = tf.cast((poles > 5) & (poles < 40), dtype=tf.float32)

    # Apply the mask to zero out B where eigenvalue is negative
    strength = strength #* mask
    
    #width = tf.constant(width, dtype=tf.float32)

    energy_expanded = tf.expand_dims(energy, axis=-1)

    numerator = strength * (width / 2 / np.pi)
    denominator = ((energy_expanded - poles) ** 2 + (width ** 2 / 4))
    
    lorentzian = numerator / denominator

    value = tf.reduce_sum(lorentzian, axis=-1)
    
    
    return value

@tf.function
def give_me_Lorentzian_batched(omega, poles_batch, B_batch, width):
    '''
    omega: (1, grid_size)
    poles_batch: (batch, n)
    B_batch: (batch, n)
    width: scalar
    '''
    # Broadcast omega: (batch, grid_size, n)
    omega_exp = tf.expand_dims(omega, axis=1)  # (1, 1, grid_size)
    omega_exp = tf.tile(omega_exp, [tf.shape(poles_batch)[0], 1, 1])  # (batch, 1, grid_size)

    poles_exp = tf.expand_dims(poles_batch, axis=-1)  # (batch, n, 1)
    B_exp = tf.expand_dims(B_batch, axis=-1)  # (batch, n, 1)

    #mask = tf.cast((poles_batch > 1.0) & (poles_batch < 30.0), dtype=tf.float32)
    #B_batch = B_batch * mask

    gamma = width
    # numerator = B_exp * (gamma / 2.0) / np.pi  # (batch, n, 1)

    # denom = tf.square(omega_exp - poles_exp) + tf.square(gamma / 2.0)  # (batch, n, grid_size)
    # lorentz = numerator / denom  # (batch, n, grid_size)
    gamma_half = gamma / 2.0
    gamma_half = tf.reshape(gamma_half, (-1, 1, 1))  # shape (batch, 1, 1)
    
    numerator = B_exp * gamma_half / np.pi  # (batch, n, 1)
    
    denom = tf.square(omega_exp - poles_exp) + tf.square(gamma_half)  # (batch, n, grid_size)
    
    lorentz = numerator / denom  # (batch, n, grid_size)

    # Sum over poles (axis=1)
    value = tf.reduce_sum(lorentz, axis=1)  # (batch, grid_size)

    return value


# nec_mat for M_true(a) = D + a * S1 + b * S2
def nec_mat(n):
    D = np.diag(np.random.uniform(1, 10, n))
    A = np.random.uniform(1, 10, (n, n))
    S1 = np.abs(A + A.T) / 2
    S2 = np.abs(A + A.T) / 2
    return D, S1, S2




def data_table(fmt_data):
    '''
    
    this loads Lorentzian and alphaD files
    
    '''
    strength = []
    alphaD = [] 
    
    m0 = []
    alphaD_test = []
    m_square = []
    
    fac = 8.0*np.pi*(7.29735e-3)*hqc/9.0
    

    for frmt in fmt_data:
        
        alpha = frmt[0]
        beta = frmt[1]

        # first open the file with the data
        file_strength = np.loadtxt('../dipoles_data_all/total_strength/strength_'+beta+'_'+alpha+'.out')
        file_alphaD = np.loadtxt('../dipoles_data_all/total_alphaD/alphaD_'+beta+'_'+alpha+'.out')
        
        '''
        This is extending the strength function to negative values
        and assigns them with zero strength
        '''
        

        # neg_energy = -np.flip(file_strength[:,0])  # from -40 to 0
        # zero_strength = np.zeros_like(neg_energy)
        
        # # Stack negative part
        # neg_part = np.column_stack((neg_energy, zero_strength))
        
        # # Combine with original
        # extended_strength = np.vstack((neg_part, file_strength))
        
        # #file_strength = file_strength[file_strength[:,0] > 1]


        # strength.append(extended_strength)  
        strength.append(file_strength)
        alphaD.append(file_alphaD)
        
        m0_tmp = np.trapz(file_strength[:,1]*file_strength[:,0], file_strength[:,0])
        m0.append(m0_tmp)
        alphaD_tmp =  fac*np.trapz(file_strength[:,1]/file_strength[:,0], file_strength[:,0])
        alphaD_test.append(alphaD_tmp)
        
        m_square.append(np.trapz(file_strength[:,1]**2, file_strength[:,0]))
        
        #print(alpha,beta, file_alphaD, alphaD_tmp)
        
    alphaD = np.array(alphaD)
    m_square = np.array(m_square)
    print(m_square.argsort())
    m0 = np.array(m0)
    print(np.min(m0), np.max(m0), np.max(m0)-np.min(m0))
    print(np.min(alphaD[:,2]), np.max(alphaD[:,2]), np.max(alphaD[:,2])-np.min(alphaD[:,2]))
    print(np.min(m_square), np.max(m_square), np.max(m_square)-np.min(m_square))
    
    #sys.exit(-1)
    return strength, alphaD





'''
    data table is now constructed for alpha & beta parameters
'''

def modified_DS_simple(params, n):
    '''

     Takes parameters and puts them into the PMM
     for Emulator 2
    
    '''
    D_shape = (n, n)
    S1_shape = (n, n)
    S2_shape = (n, n)
    S3_shape = (n, n)

    # Indices in param vector
    idx = 0

    D_mod = tf.linalg.diag(params[idx:idx+n])
    idx += n

    # S1 upper triangle
    S1_mod = tf.zeros(S1_shape, dtype=tf.float32)
    upper_tri_indices1 = np.triu_indices(S1_shape[0])
    indices1 = tf.constant(list(zip(upper_tri_indices1[0], upper_tri_indices1[1])))

    num_upper1 = len(upper_tri_indices1[0])
    S1_mod = tf.tensor_scatter_nd_update(S1_mod, indices1,
                                         params[idx:idx+num_upper1])
    idx += num_upper1

    S1_mod = S1_mod + tf.linalg.band_part(tf.transpose(S1_mod), -1, 0) - tf.linalg.diag(tf.linalg.diag_part(S1_mod))

    # S2 upper triangle
    S2_mod = tf.zeros(S2_shape, dtype=tf.float32)
    upper_tri_indices2 = np.triu_indices(S2_shape[0])
    indices2 = tf.constant(list(zip(upper_tri_indices2[0], upper_tri_indices2[1])))

    num_upper2 = len(upper_tri_indices2[0])
    S2_mod = tf.tensor_scatter_nd_update(S2_mod, indices2,
                                         params[idx:idx+num_upper2])
    idx += num_upper2

    S2_mod = S2_mod + tf.linalg.band_part(tf.transpose(S2_mod), -1, 0) - tf.linalg.diag(tf.linalg.diag_part(S2_mod))
    
    # S3 upper triangle
    S3_mod = tf.zeros(S3_shape, dtype=tf.float32)
    upper_tri_indices3 = np.triu_indices(S3_shape[0])
    indices3 = tf.constant(list(zip(upper_tri_indices3[0], upper_tri_indices3[1])))

    num_upper3 = len(upper_tri_indices3[0])
    S3_mod = tf.tensor_scatter_nd_update(S3_mod, indices3,
                                         params[idx:idx+num_upper3])
    idx += num_upper3

    S3_mod = S3_mod + tf.linalg.band_part(tf.transpose(S3_mod), -1, 0) - tf.linalg.diag(tf.linalg.diag_part(S3_mod))
    


    # Add new learned params x1 and x2 and x3
    x1 = tf.convert_to_tensor(params[idx])
    idx += 1


    return D_mod, S1_mod, S2_mod,S3_mod, x1
    
    
    





def modified_DS_affine_v(params, n):
    ''''
    Build PMM matrices:
    M = D + (alpha-alpha_0)*exp(-(beta-beta_0)*x1)*S1 + (alpha-alpha_0)*exp(-(beta-beta_0)*x2)*S2
    and external field:
    v(alpha,beta) = v0 + (alpha-alpha_0)*v1 + (beta-beta_0)*v2
    '''
    D_shape = (n, n)
    S1_shape = (n, n)
    S2_shape = (n, n)
    S3_shape = (n, n)
    S4_shape = (n, n)

    # Indices in param vector
    idx = 0
    eta = tf.convert_to_tensor(params[idx])
    idx += 1

    v0_mod = tf.convert_to_tensor(params[idx:idx+n])
    idx += n

    v1_mod = tf.convert_to_tensor(params[idx:idx+n])
    idx += n

    v2_mod = tf.convert_to_tensor(params[idx:idx+n])
    idx += n

    D_mod = tf.linalg.diag(params[idx:idx+n])
    idx += n

    # S1 upper triangle
    S1_mod = tf.zeros(S1_shape, dtype=tf.float32)
    upper_tri_indices1 = np.triu_indices(S1_shape[0])
    indices1 = tf.constant(list(zip(upper_tri_indices1[0], upper_tri_indices1[1])))

    num_upper1 = len(upper_tri_indices1[0])
    S1_mod = tf.tensor_scatter_nd_update(S1_mod, indices1,
                                         params[idx:idx+num_upper1])
    idx += num_upper1

    S1_mod = S1_mod + tf.linalg.band_part(tf.transpose(S1_mod), -1, 0) - tf.linalg.diag(tf.linalg.diag_part(S1_mod))

    # S2 upper triangle
    S2_mod = tf.zeros(S2_shape, dtype=tf.float32)
    upper_tri_indices2 = np.triu_indices(S2_shape[0])
    indices2 = tf.constant(list(zip(upper_tri_indices2[0], upper_tri_indices2[1])))

    num_upper2 = len(upper_tri_indices2[0])
    S2_mod = tf.tensor_scatter_nd_update(S2_mod, indices2,
                                         params[idx:idx+num_upper2])
    idx += num_upper2

    S2_mod = S2_mod + tf.linalg.band_part(tf.transpose(S2_mod), -1, 0) - tf.linalg.diag(tf.linalg.diag_part(S2_mod))
    
    # S3 upper triangle
    S3_mod = tf.zeros(S3_shape, dtype=tf.float32)
    upper_tri_indices3 = np.triu_indices(S3_shape[0])
    indices3 = tf.constant(list(zip(upper_tri_indices3[0], upper_tri_indices3[1])))

    num_upper3 = len(upper_tri_indices3[0])
    S3_mod = tf.tensor_scatter_nd_update(S3_mod, indices3,
                                         params[idx:idx+num_upper3])
    idx += num_upper3

    S3_mod = S3_mod + tf.linalg.band_part(tf.transpose(S3_mod), -1, 0) - tf.linalg.diag(tf.linalg.diag_part(S3_mod))
    
    # S3 upper triangle
    S4_mod = tf.zeros(S4_shape, dtype=tf.float32)
    upper_tri_indices4 = np.triu_indices(S4_shape[0])
    indices4 = tf.constant(list(zip(upper_tri_indices4[0], upper_tri_indices4[1])))

    num_upper4 = len(upper_tri_indices4[0])
    S4_mod = tf.tensor_scatter_nd_update(S4_mod, indices4,
                                         params[idx:idx+num_upper4])
    idx += num_upper4

    S4_mod = S4_mod + tf.linalg.band_part(tf.transpose(S4_mod), -1, 0) - tf.linalg.diag(tf.linalg.diag_part(S4_mod))

    # Add new learned params x1 and x2 and x3
    x1 = tf.convert_to_tensor(params[idx])
    idx += 1
    x2 = tf.convert_to_tensor(params[idx])
    idx += 1
    x3 = tf.convert_to_tensor(params[idx])
    idx += 1
    x4 = tf.convert_to_tensor(params[idx])
    idx += 1

    return D_mod, S1_mod, S2_mod,S3_mod,S4_mod, v0_mod, v1_mod, v2_mod, eta, x1, x2, x3, x4


@tf.function
def calculate_alphaD(eigenvalues, B):
    
    '''
    Mask is added to alphaD to avoid singularities
    use some small value, and this will prevent optimizer from going crazy
    '''
    
    mask = tf.cast((eigenvalues > 1.0), dtype=tf.float32)

    # Apply the mask to zero out B where eigenvalue is negative
    B = B  * mask
    
    fac = tf.constant(8.0*np.pi*(7.29735e-3)*hqc/9.0 , dtype = tf.float32)
    val = tf.reduce_sum(B / eigenvalues)*fac
    
    return val
    



# @tf.function
# def cost_function_batched_mixed(params, n, fmt_data, strength_true, alphaD_true,weight, central_point, retain):
#     '''
#     params: tf.Variable
#     fmt_data: list of (alpha, beta)
#     strength_true: list of arrays (omega, Lorentzian_true)
#     alphaD_true: list of true alphaD values
#     alpha_0, beta_0: centroids

#     Returns:
#     total_cost, alphaD_calc
#     '''

#     # Unpack PMM params
#     D_mod, S1_mod, S2_mod,S3_mod,S4_mod, v0_mod, v1_mod, v2_mod, eta, x1, x2,x3, x4 = modified_DS_affine_v(params, n)

#     # Build alpha, beta tensors
#     alpha_list = [float(alpha[0]) for alpha in fmt_data]
#     beta_list  = [float(alpha[1]) for alpha in fmt_data]

#     alpha_tensor = tf.constant(alpha_list, dtype=tf.float32)  # (batch,)
#     beta_tensor  = tf.constant(beta_list, dtype=tf.float32)   # (batch,)
    
#     alpha_c = tf.constant(float(central_point[0]), dtype=tf.float32)
#     beta_c  = tf.constant(float(central_point[1]), dtype=tf.float32)
    

#     alpha_shift = alpha_tensor - alpha_c
#     beta_shift  = beta_tensor - beta_c
    
#     exp1 = tf.exp( -alpha_shift * x1 )



#     #Build batched M
#     M_batch = D_mod[None,:,:] \
#             + alpha_shift[:,None,None] * S1_mod[None,:,:] \
#             + beta_shift[:,None,None] * S2_mod[None,:,:] \
#             + beta_shift[:,None,None] * exp1[:,None,None] * S3_mod[None,:,:] \
            
#     #eta_new = tf.sqrt(eta**2 + (x2*alpha_tensor + x3*beta_tensor)**2)
#     eta_broadcast = tf.broadcast_to(eta, tf.shape(alpha_tensor))  # (batch,)

#     eta_new = tf.sqrt(
#     tf.square(eta_broadcast) + tf.square(x2 + x3 * alpha_shift + x4 * beta_shift)
#     )
        


#     # Batched eigen-decomposition
#     eigenvalues, eigenvectors = tf.linalg.eigh(M_batch)  # (batch, n), (batch, n, n)
    

#     # Build batched v_eff
#     v_eff_batch = v0_mod[None,:] \
#                   + alpha_shift[:,None] * v1_mod[None,:] \
#                   + beta_shift[:,None]  * v2_mod[None,:]  # (batch, n)


#     # discard = int((1-retain)*n/2)
    
#     # if discard > 0:
#     #     eigenvalues = eigenvalues[:, discard:-discard]
#     #     eigenvectors_T = tf.transpose(eigenvectors, perm=[0,2,1])
#     #     eigenvectors_T = eigenvectors_T[:, discard:-discard, :]
#     # else:
#     # #     # keep everything
#     #     eigenvalues = eigenvalues
#     #     eigenvectors_T = tf.transpose(eigenvectors, perm=[0,2,1])
    
#     # eigenvalues:   (B, n)  ascending from tf.linalg.eigh
#     # eigenvectors:  (B, n, n)
#     n_i = tf.shape(eigenvalues)[1]                 # n (int32)
#     r   = tf.convert_to_tensor(retain, tf.float32)
    
#     # how many to keep
#     k_keep = tf.cast(tf.round(r * tf.cast(n_i, tf.float32)), tf.int32)
#     k_keep = tf.clip_by_value(k_keep, 1, n_i)      # ensure [1, n]
    
#     # centered slice [left:right)
#     left  = (n_i - k_keep) // 2
#     right = left + k_keep
    
#     # slice the centered block
#     eigenvalues = eigenvalues[:, left:right]    # (B, k_keep)
#     eigvecsT       = tf.transpose(eigenvectors, [0, 2, 1])  # (B, n, n)
#     eigenvectors_T= eigvecsT[:, left:right, :]    # (B, k_keep, n)
    
#     # Then compute projections on reduced eigenspace
#     projections = tf.matmul(eigenvectors_T, v_eff_batch[:,:,None])
#     projections = tf.squeeze(projections, axis=-1)  # (batch, n)
#     projections = tf.transpose(projections, perm=[0,1])  # (batch, n)

#     B_batch = tf.square(projections)  # (batch, n)

#     # Now loop over batch for Lorentzian + alphaD cost
#     total_cost = 0.0
#     alphaD_calc = []

#     # Omega grid (you said it is always the same)
#     #omega_tensor = tf.constant(strength_true[0][:,0], dtype=tf.float32)  # (grid_size,)
#     omega_tensor = tf.cast(strength_true[0][:,0], dtype=tf.float32)

#     # Stack strength_true into tensor
#     strength_true_tensor = tf.stack([
#     tf.cast(s[:,1], dtype=tf.float32) for s in strength_true
#     ], axis=0)

#     # Batched Lorentzian
#     Lor_batch = give_me_Lorentzian_batched(omega_tensor[None,:], eigenvalues, B_batch, eta_new)

#     # Lorentzian cost
#     diff = Lor_batch - strength_true_tensor
#     cost_Lor = tf.reduce_sum( tf.square(diff), axis=1 )  # (batch,)
#     total_cost_Lor = tf.reduce_sum(cost_Lor)

#     # AlphaD batch
#     mask = tf.cast( (eigenvalues > 1.0), dtype=tf.float32 )
#     B_masked = B_batch * mask 

#     fac = tf.constant(8.0*np.pi*(7.29735e-3)*hqc/9.0, dtype=tf.float32)

#     alphaD_calc = tf.reduce_sum( B_masked / eigenvalues, axis=1 ) * fac  # (batch,)

#     #alphaD_true_tensor = tf.constant( [a[2] for a in alphaD_true], dtype=tf.float32 )
#     alphaD_true_tensor = tf.constant(alphaD_true, dtype=tf.float32)

#     rel_error = (alphaD_calc - alphaD_true_tensor)
#     total_cost_alphaD = tf.reduce_sum( weight * tf.square(rel_error) )
    
    
#     '''
#     Impose sum rule as well !
#     '''
#     m1_calc = tf.reduce_sum( B_masked * eigenvalues, axis=1 )  # (batch,)
#     rel_error = 0.01*(m1_calc - 875)
#     total_cost_m1 = tf.reduce_sum( tf.square(rel_error) )
    
    

#     # Final total cost
#     total_cost = total_cost_Lor + total_cost_alphaD  + total_cost_m1
    
#     Lor_last = Lor_batch[-1,:]
#     Lor_true_last = strength_true_tensor[-1,:]
    
#     B_last = B_masked[-1,:]
#     eigenvalues_last = eigenvalues[-1,:]

#     return total_cost, Lor_last, Lor_true_last, omega_tensor, alphaD_calc, B_last, eigenvalues_last


@tf.function
def cost_function_batched_mixed(
    params,
    n,
    fmt_data,
    strength_true,          # list of (G,2): [omega, S_true]
    alphaD_true,            # list/array of shape (B,)
    central_point,          # (alpha0, beta0)
    retain,                 # fraction in (0,1]
    w_strength,         # weight for strength term
    w_mminus1,          # weight for m_{-1} (alpha_D) term
    w_mplus1,           # weight for m_{+1} (TRK-like) term
    m1_target=875.0,        # scalar target for m_{+1}; or pass a tensor later
    eps=1e-8
):
    """
    Returns
    -------
    total_cost, Lor_last, Lor_true_last, omega_tensor, alphaD_calc, B_last, eigenvalues_last

    Notes on normalizations:
      - Strength term uses integrated L2 normalized by ∫ S_true^2 dω (per sample).
      - m_{-1} term uses relative squared error.
      - m_{+1} term uses relative squared error to m1_target.
    """

    # ---------- Unpack PMM params ----------
    D_mod, S1_mod, S2_mod, S3_mod, S4_mod, v0_mod, v1_mod, v2_mod, eta, x1, x2, x3, x4 = modified_DS_affine_v(params, n)

    # ---------- Batch (alpha, beta) ----------
    alpha_list = [float(a[0]) for a in fmt_data]
    beta_list  = [float(a[1]) for a in fmt_data]
    alpha_tensor = tf.constant(alpha_list, tf.float32)   # (B,)
    beta_tensor  = tf.constant(beta_list,  tf.float32)   # (B,)
    Bsize = tf.shape(alpha_tensor)[0]

    alpha_c = tf.constant(float(central_point[0]), tf.float32)
    beta_c  = tf.constant(float(central_point[1]), tf.float32)

    alpha_shift = alpha_tensor - alpha_c
    beta_shift  = beta_tensor  - beta_c

    exp1 = tf.exp(-alpha_shift * x1)

    # ---------- Build M (B, n, n) ----------
    M_batch = (
        D_mod[None, :, :]
        + alpha_shift[:, None, None] * S1_mod[None, :, :]
        + beta_shift[:,  None, None] * S2_mod[None, :, :]
        + beta_shift[:,  None, None] * exp1[:, None, None] * S3_mod[None, :, :]
    )

    # η_new per sample
    eta_b = tf.broadcast_to(eta, tf.shape(alpha_tensor))  # (B,)
    eta_new = tf.sqrt(tf.square(eta_b) + tf.square(x2 + x3 * alpha_shift + x4 * beta_shift))  # (B,)

    # ---------- Eigendecomposition ----------
    eigenvalues, eigenvectors = tf.linalg.eigh(M_batch)   # (B, n), (B, n, n)

    # ---------- v_eff (B, n) ----------
    v_eff_batch = v0_mod[None, :] + alpha_shift[:, None] * v1_mod[None, :] + beta_shift[:, None] * v2_mod[None, :]

    # ---------- Retain centered fraction ----------
    n_i = tf.shape(eigenvalues)[1]
    r   = tf.convert_to_tensor(retain, tf.float32)
    k_keep = tf.cast(tf.round(r * tf.cast(n_i, tf.float32)), tf.int32)
    k_keep = tf.clip_by_value(k_keep, 1, n_i)
    left  = (n_i - k_keep) // 2
    right = left + k_keep

    eigenvalues   = eigenvalues[:, left:right]            # (B, k)
    eigvecsT_full = tf.transpose(eigenvectors, [0, 2, 1]) # (B, n, n)
    eigenvectors_T= eigvecsT_full[:, left:right, :]       # (B, k, n)

    # ---------- Projections & B (B, k) ----------
    proj = tf.matmul(eigenvectors_T, v_eff_batch[:, :, None])   # (B, k, 1)
    proj = tf.squeeze(proj, axis=-1)                             # (B, k)
    B_batch = tf.square(proj)                                    # (B, k)

    # ---------- Data tensors ----------
    # Shared ω-grid:
    omega_tensor = tf.cast(strength_true[0][:, 0], tf.float32)      # (G,)
    # Stack strengths: (B, G)
    strength_true_tensor = tf.stack([tf.cast(s[:, 1], tf.float32) for s in strength_true], axis=0)

    # ---------- Lorentzian model (B, G) ----------
    Lor_batch = give_me_Lorentzian_batched(omega_tensor[None, :], eigenvalues, B_batch, eta_new)  # (B, G)

    # ---------- Trapezoidal weights on ω (G,) ----------
    d = omega_tensor[1:] - omega_tensor[:-1]                    # (G-1,)
    w0 = d[0] / 2.0
    wN = d[-1] / 2.0
    w_mid = (omega_tensor[2:] - omega_tensor[:-2]) / 2.0 if tf.shape(omega_tensor)[0] > 2 else tf.zeros([0], tf.float32)
    w = tf.concat([[w0], w_mid, [wN]], axis=0)                  # (G,)
    w = tf.maximum(w, 0.0)

    # ---------- Strength loss: normalized integrated L2 ----------
    diff = Lor_batch - strength_true_tensor                        # (B, G)
    num  = tf.reduce_sum(tf.square(diff) * w[None, :], axis=1)     # (B,)
    #den  = tf.reduce_sum(tf.square(strength_true_tensor) * w[None, :], axis=1) + eps
    den = tf.cast(285.66404867541524, tf.float32)
    L_strength_per = num / den                                     # (B,)
    L_strength = tf.reduce_mean(L_strength_per)                    # scalar

    # ---------- m_{-1} (αD) with mask & relative error ----------
    mask = tf.cast(eigenvalues > 1.0, tf.float32)                  # (B, k)
    denom = tf.where(mask > 0.0, eigenvalues, tf.ones_like(eigenvalues))  # avoid 0/0
    fac = tf.constant(8.0 * np.pi * (7.29735e-3) * hqc / 9.0, tf.float32)

    alphaD_calc = tf.reduce_sum((B_batch * mask) / denom, axis=1) * fac    # (B,)
    alphaD_true_tensor = tf.cast(alphaD_true, tf.float32)                  # (B,)
    w_alphaD = tf.cast(7.1110237745439, tf.float32)

    rel_mminus1 = (alphaD_calc - alphaD_true_tensor) / (w_alphaD + eps)   # (B,)
    L_mminus1 = tf.reduce_mean(tf.square(rel_mminus1))                     # scalar

    # ---------- m_{+1} (TRK-like) relative error ----------
    m1_calc = tf.reduce_sum(B_batch * eigenvalues * mask, axis=1)          # (B,)
    m1_tgt  = tf.cast(m1_target, tf.float32)           
    w_m1 = tf.cast(58.703055785984134, tf.float32)                    
    rel_mplus1 = (m1_calc - m1_tgt) / (w_m1 + eps)                       # (B,)
    L_mplus1 = tf.reduce_mean(tf.square(rel_mplus1))                       # scalar

    # ---------- Weighted total ----------
    total_cost = (
        tf.cast(w_strength,  tf.float32) * L_strength +
        tf.cast(w_mminus1,   tf.float32) * L_mminus1 +
        tf.cast(w_mplus1,    tf.float32) * L_mplus1
    )
    
    strength_cost = tf.cast(w_strength,  tf.float32) * L_strength
    alphaD_cost =  tf.cast(w_mminus1,   tf.float32) * L_mminus1
    m1_cost =  tf.cast(w_mplus1,    tf.float32) * L_mplus1

    # ---------- Extras for plotting/debug ----------
    Lor_last = Lor_batch[-1, :]
    Lor_true_last = strength_true_tensor[-1, :]
    B_last = (B_batch * mask)[-1, :]
    eigenvalues_last = eigenvalues[-1, :]

    return total_cost, strength_cost, alphaD_cost,m1_cost, Lor_last, Lor_true_last, omega_tensor, alphaD_calc, B_last, eigenvalues_last



def cost_function_only_alphaD(params, n, fmt_data, alphaD_true, central_point, d_alpha, d_beta):
    
    '''
    params: tf.Variable
    D_shape, S1_shape, S2_shape : int
    alpha_values, beta_values: list
    
    calculates the cost function by subtracting two Lorentzians
    and also alphaD
    
    fold: folding width for training
    
    '''
    
    D_mod, S1_mod, S2_mod, S3_mod, x1 = modified_DS_simple(params,  n)
    
    total_cost = 0
    
    count = 0
    alphaD_calc = []
    
    for idx, alpha in enumerate(fmt_data):
        
        exp1 = tf.exp( -(float(alpha[0])- float(central_point[0]))/d_alpha * x1 )

        M_true = D_mod + (float(alpha[0]) - float(central_point[0]))/d_alpha  * S1_mod \
                       + (float(alpha[1]) - float(central_point[1]))/d_beta  * S2_mod \
                       + (float(alpha[1]) - float(central_point[1]))/d_alpha * exp1 * S3_mod

        eigenvalues, eigenvectors = tf.linalg.eigh(M_true)
        
        

        val_true = alphaD_true[idx]
        # val = eigenvalues[1]
        # val = 0.0
        # for i in range(n):
        #     # val += tf.sqrt(eigenvalues[i]**2)
        #     val += eigenvalues[i]**2
        #     # val += eigenvalues[i]
            
        val = eigenvalues[int(n/2)]
        
    
        
        total_cost += (val_true - val)**2
        
        alphaD_calc.append(val)
        
        

        count+=1
            
    return total_cost, alphaD_calc

@tf.function
def cost_function_only_alphaD_batched(params, n, fmt_data, alphaD_true, central_point):
    
    '''
    params: tf.Variable
    D_shape, S1_shape, S2_shape : int
    alpha_values, beta_values: list
    
    calculates the cost function by subtracting two Lorentzians
    and also alphaD
    
    fold: folding width for training
    
    '''
    
    # Unpack PMM params
    D_mod, S1_mod, S2_mod,S3_mod, x1 = modified_DS_simple(params, n)

    # Build alpha, beta tensors
    alpha_list = [float(alpha[0]) for alpha in fmt_data]
    beta_list  = [float(alpha[1]) for alpha in fmt_data]

    alpha_tensor = tf.constant(alpha_list, dtype=tf.float32)  # (batch,)
    beta_tensor  = tf.constant(beta_list, dtype=tf.float32)   # (batch,)
    

    
    alpha_c = tf.constant(float(central_point[0]), dtype=tf.float32)
    beta_c  = tf.constant(float(central_point[1]), dtype=tf.float32)
    
    

    alpha_shift = (alpha_tensor - alpha_c)
    beta_shift  = (beta_tensor - beta_c)
    
    exp1 = tf.exp( -alpha_shift * x1 )
    #exp2 = tf.exp( -alpha_tensor * x2 )



    #Build batched M
    M_batch = D_mod[None,:,:] \
            + alpha_shift[:,None,None] * S1_mod[None,:,:] \
            + beta_shift[:,None,None] * S2_mod[None,:,:] \
            + beta_shift[:,None,None] * exp1[:,None,None] * S3_mod[None,:,:] 
            


    # Batched eigen-decomposition
    eigenvalues, eigenvectors = tf.linalg.eigh(M_batch)  # (batch, n), (batch, n, n)
    


    # Now loop over batch for Lorentzian + alphaD cost
    total_cost = 0.0
    alphaD_calc = []



    #alphaD_calc = tf.reduce_sum( tf.square(eigenvalues), axis=1 )   # (batch,)
    mid_index = n // 2
    alphaD_calc = tf.gather(eigenvalues, indices=mid_index, axis=1)


    alphaD_true_tensor = tf.constant(alphaD_true, dtype=tf.float32)

    rel_error = (alphaD_calc - alphaD_true_tensor)
    total_cost_alphaD = tf.reduce_sum(tf.square(rel_error) )

    # Final total cost
    total_cost =  total_cost_alphaD
    
            
    return total_cost, alphaD_calc



       # generalized_eigen for M_true(a)
#@tf.function
def generalized_eigen(D, S1, S2, alpha, central_point):
    M_true = D + (float(alpha[0]) - float(central_point[0])) * S1 \
        + (float(alpha[1]) - float(central_point[1])) * S2
    eigenvalues, eigenvectors = eigh(M_true)
    return eigenvalues, eigenvectors 






def data_Lorentzian_for_idx(idx, test_set,n,params, central_point):

    alpha = float(test_set[idx][0])
    beta = float(test_set[idx][1])
    
    
    Lors_test, alphaD_test = data_table(test_set)
    Lors_orig = Lors_test[idx]
    
    opt_D, opt_S1, opt_S2,opt_S3, opt_v0,opt_v1, opt_v2, fold, x1, x2, x3 = modified_DS_affine_v(params, n)
    #opt_eigenvalues, opt_eigenvectors = generalized_eigen(opt_D.numpy(), opt_S1.numpy(), opt_S2.numpy(), test_set[idx], central_point)
    exp1 = tf.exp( -(alpha) * x1 )
    #exp2 = tf.exp( -(alpha) * x2 )
    #exp3 = tf.exp( -(alpha) * x3 )
    
    # M_true = opt_D + (alpha - float(central_point[0])) * opt_S1 \
    #                + (beta - float(central_point[1])) * opt_S2 \
    #                + (beta) * exp1 * opt_S3 \
    M_true = opt_D + (alpha - float(central_point[0])) * opt_S1 \
                   + (beta - float(central_point[1])) * opt_S2 \
                   #+ (alpha - float(central_point[0]))*(beta - float(central_point[1])) * opt_S3 \
                    

    opt_eigenvalues, opt_eigenvectors = tf.linalg.eigh(M_true)
    
    disc = int((1-0.6)*n/2)
    opt_eigenvalues = opt_eigenvalues[disc:-disc]
    opt_eigenvectors = opt_eigenvectors[:, disc:-disc]
    
    v_eff = opt_v0 #+ beta * exp3 * opt_v1
    projections = tf.linalg.matvec(tf.transpose(opt_eigenvectors), v_eff)
    
    # Square each projection
    B = tf.square(projections)
    
   # mask = tf.cast((opt_eigenvalues > 5) &  (opt_eigenvalues < 40), dtype=tf.float32)

    # Apply the mask to zero out B where eigenvalue is negative
    opt_dot_products = B #* mask
    

   
    x = Lors_orig[:,0]
        
    opt_Lor = give_me_Lorentzian(x, opt_eigenvalues, opt_dot_products, fold)
    

    
    return x, Lors_orig[:,1], opt_Lor
    
 
def plot_alphaD(test_set,params,n, central_point, retain): 
    alphaD_guess = []
    times = []
    
    Lors_test, alphaD_test = data_table(test_set)
    alphaD_test = np.vstack(alphaD_test)
    
    
    
    for idx in range(len(test_set)):
        
        alpha = float(test_set[idx][0])
        beta = float(test_set[idx][1])
        
        alpha_tensor = tf.constant(alpha, dtype=tf.float32)  # (batch,)
        beta_tensor  = tf.constant(beta, dtype=tf.float32)
        
        start = time.time()  # Start time
        
        
        
        opt_D, opt_S1, opt_S2,opt_S3,opt_S4, opt_v0,opt_v1, opt_v2, fold, x1, x2, x3, x4 = modified_DS_affine_v(params, n)

        exp1 = tf.exp( -(alpha_tensor- float(central_point[0])) * x1)

        
        M_true = opt_D + (alpha_tensor- float(central_point[0])) * opt_S1 \
                     + (beta_tensor- float(central_point[1])) * opt_S2 \
                     + (beta_tensor- float(central_point[1])) * exp1 * opt_S3  \
                     #+ beta_tensor * exp2 * opt_S4
                     
        opt_eigenvalues, opt_eigenvectors = tf.linalg.eigh(M_true)
         
        n_i = opt_eigenvalues.shape[0]
        k_keep = int(round(retain * n_i))         # how many eigenvalues to keep
        k_keep = max(1, min(k_keep, n_i))         # safety: clamp between 1 and n
        
        left  = (n_i - k_keep) // 2               # starting index of the centered block
        right = left + k_keep                     # ending index (exclusive)
        
        opt_eigenvalues  = opt_eigenvalues[left:right]
        opt_eigenvectors = opt_eigenvectors[:, left:right]

         
        v_eff = opt_v0 \
              + (alpha_tensor- float(central_point[0])) * opt_v1 \
              + (beta_tensor- float(central_point[1])) * opt_v2 
              
        projections = tf.linalg.matvec(tf.transpose(opt_eigenvectors), v_eff)
        
        # Square each projection
        B = tf.square(projections)
        

        
        alphaD_guess.append(calculate_alphaD(opt_eigenvalues, B))
        
        end = time.time()  # Start time
        
        
        
        
        times.append(end-start)
        
        
    return alphaD_guess, alphaD_test[:,2], times


def plot_alphaD_simple(test_set,params,n, central_point): 
    alphaD_guess = []
    times = []
    
    Lors_test, alphaD_test = data_table(test_set)
    alphaD_test = np.vstack(alphaD_test)
    
    for idx in range(len(test_set)):
        
        start = time.time()  # Start time
        
        opt_D, opt_S1, opt_S2, opt_S3, x1 = modified_DS_simple(params, n)
        
        exp1 = tf.exp( -(float(test_set[idx][0]) - float(central_point[0])) * x1 )

        M_true = opt_D + (float(test_set[idx][0]) - float(central_point[0]))  * opt_S1 \
                       + (float(test_set[idx][1]) - float(central_point[1]))  * opt_S2 \
                       +  (float(test_set[idx][1]) - float(central_point[1])) * exp1 * opt_S3

        opt_eigenvalues, opt_eigenvectors = tf.linalg.eigh(M_true)
        
        #M_true = opt_D + float(test_set[idx][0]) * opt_S1 + float(test_set[idx][1]) * opt_S2

       # opt_eigenvalues, opt_eigenvectors = generalized_eigen(opt_D.numpy(), opt_S1.numpy(), opt_S2.numpy(), test_set[idx])#tf.linalg.eigh(M_true)
        
        end = time.time()  # Start time
        
        #alphaD_guess.append(opt_eigenvalues[1])
        # val = 0.0
        # for i in range(n):
        #     # val += tf.sqrt(eigenvalues[i]**2)
        #     val += opt_eigenvalues[i]**2
        #     # val += eigenvalues[i]
        val = opt_eigenvalues[int(n/2)]
            
        alphaD_guess.append(val)
        
        
        
        times.append(end-start)
        
        
    return alphaD_guess, alphaD_test[:,2], times



