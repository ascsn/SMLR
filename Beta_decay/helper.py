# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
from scipy.linalg import eigh, eig
import matplotlib.pyplot as plt
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
import matplotlib.ticker as ticker
from scipy.optimize import least_squares, nnls

emass = 0.511 # MeV
alpha_c = 1/137
hbarc = 197.33 # MeV fm
compton = hbarc/emass # fm
kappa = 6147 #s
del_np = 1.293 # MeV
del_nH = 0.782 # MeV



def encode_initial_guess(random_initial_guess, E, B, n, retain):
    """
    Centered retain: place k=round(retain*n) fitted (E,B) in the middle of the n-diagonal.
    D outside the kept block is filled by +/-2 stepping:
      left side:  min(E_sel) - 2, -4, ...
      right side: max(E_sel) + 2, +4, ...
    v0 outside kept block is zero.

    Parameters
    ----------
    random_initial_guess : np.ndarray (float64)
        Base packed parameter vector.
    E, B : array-like
        Fitted energies and strengths (length >= k_keep).
    n : int
        Matrix dimension.
    retain : float in (0,1]
        Fraction to retain.

    Returns
    -------
    params : np.ndarray (float64)
    """
    n = int(n)
    params = np.asarray(random_initial_guess, dtype=np.float64).copy()

    # how many to keep (centered)
    k_keep = int(round(float(retain) * n))
    k_keep = max(1, min(k_keep, n))
    left  = (n - k_keep) // 2
    right = left + k_keep

    # sort by energy and take centered slice of fitted set
    E = np.asarray(E, dtype=np.float64).reshape(-1)
    B = np.asarray(B, dtype=np.float64).reshape(-1)
    order = np.argsort(E)
    E, B = E[order], B[order]

    if len(E) < k_keep:
        raise ValueError(f"E,B need at least k_keep={k_keep} entries (got {len(E)}).")

    m = len(E)
    start = (m - k_keep) // 2
    E_sel = E[start:start + k_keep]
    B_sel = B[start:start + k_keep]

    # ---- build full D: kept center + +/-2 stepping on ends ----
    D_full = np.empty(n, dtype=np.float64)
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
    v0_full = np.zeros(n, dtype=np.float64)
    v0_full[left:right] = np.sqrt(np.maximum(B_sel, 0.0)).astype(np.float64)

    # ---- write back into packed vector (your layout) ----
    num_upper = n * (n + 1) // 2
    idx = 0
    idx += 1                                   # eta (keep as-is)
    params[idx:idx + n] = v0_full; idx += n    # v0
    params[idx:idx + n] = D_full; idx += n     # D diagonal
    idx += num_upper                           # S1 (leave)
    idx += num_upper                           # S2 (leave)
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
    z = tf.convert_to_tensor(z, tf.float64)
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
        E_tf, B_tf = _unpack_params_tf(z, n, tf.constant(wmin, tf.float64),
                                        tf.constant(min_spacing, tf.float64))
        yhat_tf = give_me_Lorentzian(omega, E_tf, B_tf, tf.constant(eta, tf.float64))
        r = (yhat_tf.numpy() - y)
        if l2 > 0:
            r = np.concatenate([r, np.sqrt(l2) * np.asarray(z, dtype=float)])
        return r

    res = least_squares(residuals, z0, method="trf",
                        max_nfev=5000, xtol=1e-10, ftol=1e-10, gtol=1e-10)

    # unpack final parameters and compute final curve (all through TF)
    E_tf, B_tf = _unpack_params_tf(res.x, n, tf.constant(wmin, tf.float64),
                                    tf.constant(min_spacing, tf.float64))
    yhat_tf = give_me_Lorentzian(omega, E_tf, B_tf, tf.constant(eta, tf.float64))

    E_hat = E_tf.numpy()
    B_hat = B_tf.numpy()
    y_hat = yhat_tf.numpy()
    return E_hat, B_hat, y_hat



# define the Fermi function as a function of electron energy
def Fermi(Z,A,W):
    
    R = 1.2*A**(1/3)
    R = R / compton
    
    gamma_1 = np.sqrt(1 - (alpha_c*Z)**2)
    p = np.sqrt(W**2 - 1)
    y = alpha_c*Z*W/p
    
    gamma_part = np.abs(gamma(gamma_1 + 1j*y))**2/gamma(2*gamma_1+1)**2
    part_1 = 4*(2*p*R)**(-2*(1-gamma_1))
    part_2 = np.exp(np.pi*y)
    L_0 = 0.5*(1+ gamma_1)
    
    return part_1*part_2*gamma_part*L_0



# define the phase factor integral
# p W ( W0 - W)**2 F(Z,W)

def phase_factor_integrand(kind,Z,A,W_0,W):
    '''
    Based on the kind argument I can have different types
    of phase-space factors

    kind = 0: pW(W_0-W)^2F(Z,W)


    '''
    
    if (W**2 - 1 < 0): return 0
    p = np.sqrt(W**2 - 1)

    if (kind == 0):
        return p*W*(W_0-W)**2*Fermi(Z+1,A,W)
    elif (kind == 1): # ka
        return p*W**2*(W_0-W)**2*Fermi(Z+1,A,W)
    elif (kind == 2): # kb
        return p*(W_0-W)**2*Fermi(Z+1,A,W)
    elif (kind == 3): # kc
        return p*W**3*(W_0-W)**2*Fermi(Z+1,A,W)
    else:
        print('Wrong kind in phase_factor_integrand!')
        exit(-1)


def phase_factor(kind,Z,A,W_0):

    '''
    Performs integration of the phase_factor_integrand function
    '''
    
    integrand = lambda W: phase_factor_integrand(kind,Z,A,W_0,W)
    
    res, err = integrate.quad(integrand, 1, W_0)
    

    return res

def theta2(K_val):
    if (K_val == 0):
        return 1
    else:
        return 2
    
    
    
def fit_phase_space(kind, Z, A, ala_np):

    ''' 
    
    Fit the phase-space factor to a N_fit-degree polynomial

    returns:

    popt - list of polynomial parameters for the fit
    
    '''

    # Number of Chebyshev grid points
    n_points = 20

    # Generate the Chebyshev grid points
    chebyshev_grid = np.cos(np.pi * (np.arange(n_points) + 0.5) / n_points)

    # Define the range for your x-coordinates (e.g., between -1 and 1)
    x_min, x_max = emass, ala_np+del_np

    # Transform the Chebyshev grid to a regular grid
    regular_grid = 0.5 * (x_min + x_max) + 0.5 * (x_max - x_min) * chebyshev_grid

    # calculate phase-space factor on chebyshev grid
    ydata = []
    for w0 in regular_grid:

        ydata.append(phase_factor(kind, Z+1,A,w0/emass))

    xdata = regular_grid
    ydata = np.array(ydata)


    # on the same figure perform a Lagrange fit
    poly = lagrange(xdata, ydata)



    return poly


@tf.function
def evaluate_polynomial_tf(coeffs, x):
    """
    TensorFlow function to evaluate a polynomial at a given point.
    
    Parameters:
        coeffs (tf.Tensor): Coefficients [a0, a1, ..., aN] 
        x (float or tf.Tensor): The x values to evaluate the polynomial at
    
    Returns:
        tf.Tensor: Polynomial values at x
    """
    N = tf.shape(coeffs)[0]
    powers = tf.range(N, dtype=coeffs.dtype)

    # Ensure x is properly broadcasted (even if a single value)
    x_tensor = tf.reshape(x, [-1, 1])  # Reshape to 2D tensor
    x_powers = tf.pow(x_tensor, powers)

    return tf.reduce_sum(coeffs * x_powers, axis=-1)
    



# TensorFlow routine
@tf.function
def half_life_loss(eigenvalues, B,coeffs,g_A):
    '''
        Routine which calculates the half-lives
    '''
    
    W_0 = (tf.constant(del_np, dtype=tf.float64) - eigenvalues) / tf.constant(emass, dtype=tf.float64)
    
    # Apply phase factor only if W_0 > 1
    valid = tf.greater(W_0, 1.0)
    
    #phase = tf.constant(1.0, dtype=tf.float64)
    
    # Compute phase factor and contributions
    coefficients = tf.reverse(coeffs, axis=[0])
    #coefficients = tf.constant(coeffs[::-1], dtype=tf.float64)
    
    
    x_tensor = W_0*emass

    # Evaluate the polynomial at all x values
    phase = evaluate_polynomial_tf(coefficients, x_tensor)
    
    # Add contributions only where W_0 > 1
    contributions = tf.where(valid, phase * B * tf.constant(g_A**2, dtype=tf.float64) \
* tf.math.log(tf.constant(2.0, dtype=tf.float64)) / tf.constant(kappa, dtype=tf.float64), tf.constant(0.0, dtype=tf.float64))
    
    # Sum the contributions
    suma = tf.math.log(tf.constant(2.0, dtype=tf.float64))/tf.reduce_sum(contributions)
    
    return suma

@tf.function
def give_me_Lorentzian(energy, poles, strength, width):
    if isinstance(energy, np.ndarray):
        energy = tf.convert_to_tensor(energy, dtype=tf.float64)

    poles = tf.convert_to_tensor(poles, dtype=tf.float64)
    strength = tf.convert_to_tensor(strength, dtype=tf.float64)

    energy_expanded = tf.expand_dims(energy, axis=-1)

    numerator = strength * (width / 2 / np.pi)
    denominator = ((energy_expanded - poles) ** 2 + (width ** 2 / 4))
    
    lorentzian = numerator / denominator

    value = tf.reduce_sum(lorentzian, axis=-1)
    
    
    return value



# nec_mat for M_true(a) = D + a * S1 + b * S2
def nec_mat(n):
    D = np.diag(np.random.uniform(1, 10, n))
    A = np.random.uniform(1, 10, (n, n))
    S1 = np.abs(A + A.T) / 2
    S2 = np.abs(A + A.T) / 2
    return D, S1, S2




def data_table(fmt_data, coeffs, g_A, nucnam):
    '''
    
    Here split the dataset from "beta_decay_data" folder
    into: training set, validation set and test set
    use any ratio you like (e.g. 0.8 0.1 0.1)
    
    For the optimizatio use only training set, and after you finish
    test it on validation set
    
    returns also number of QRPA poles n_QRPA
    
    '''

    Lors = []
    HLs = []
    

    for frmt in fmt_data:
        
        alpha = frmt[0]
        beta = frmt[1]

        # first open the file with the data
        file = np.loadtxt('../beta_decay_data_'+nucnam+'/lorm_'+nucnam+'_'+beta+'_'+alpha+'.out')
        
        # normalize the Lorentzians
        #norm = np.sum(file[:,1])
        #file[:,1] = file[:,1]/norm
        
        #print(alpha, beta, norm**2)

        file = file[file[:,0]<del_nH]
        #file = file[file[:,0]<10]
        file = file[file[:,0]>-10]

        Lors.append(file)  
        
        # now calculate half-lives the old way
        file = np.loadtxt('../beta_decay_data_'+nucnam+'/excm_'+nucnam+'_'+beta+'_'+alpha+'.out')
        file = file[file[:,0]<del_nH]
        file = file[file[:,0]>-10]
        HLs.append(half_life_loss(file[:,0], file[:,1],coeffs, g_A))

     
    return Lors, HLs



'''
    data table is now constructed for alpha & beta parameters
'''





def modified_DS(params, n):
    ''''
    Build PMM matrices:
    M = D + (alpha-alpha_0)*exp(-(beta-beta_0)*x1)*S1 + (alpha-alpha_0)*exp(-(beta-beta_0)*x2)*S2
    and external field:
    v(alpha,beta) = v0 + (alpha-alpha_0)*v1 + (beta-beta_0)*v2
    '''
    D_shape = (n, n)
    S1_shape = (n, n)
    S2_shape = (n, n)

    # Indices in param vector
    idx = 0
    eta = tf.convert_to_tensor(params[idx])
    idx += 1

    v0_mod = tf.convert_to_tensor(params[idx:idx+n])
    idx += n

    D_mod = tf.linalg.diag(params[idx:idx+n])
    idx += n

    # S1 upper triangle
    S1_mod = tf.zeros(S1_shape, dtype=tf.float64)
    upper_tri_indices1 = np.triu_indices(S1_shape[0])
    indices1 = tf.constant(list(zip(upper_tri_indices1[0], upper_tri_indices1[1])))

    num_upper1 = len(upper_tri_indices1[0])
    S1_mod = tf.tensor_scatter_nd_update(S1_mod, indices1,
                                         params[idx:idx+num_upper1])
    idx += num_upper1

    S1_mod = S1_mod + tf.linalg.band_part(tf.transpose(S1_mod), -1, 0) - tf.linalg.diag(tf.linalg.diag_part(S1_mod))

    # S2 upper triangle
    S2_mod = tf.zeros(S2_shape, dtype=tf.float64)
    upper_tri_indices2 = np.triu_indices(S2_shape[0])
    indices2 = tf.constant(list(zip(upper_tri_indices2[0], upper_tri_indices2[1])))

    num_upper2 = len(upper_tri_indices2[0])
    S2_mod = tf.tensor_scatter_nd_update(S2_mod, indices2,
                                         params[idx:idx+num_upper2])
    idx += num_upper2

    S2_mod = S2_mod + tf.linalg.band_part(tf.transpose(S2_mod), -1, 0) - tf.linalg.diag(tf.linalg.diag_part(S2_mod))
    
    
    # Add new learned params x1 and x2 and x3
    x1 = tf.convert_to_tensor(params[idx])
    idx += 1
    x2 = tf.convert_to_tensor(params[idx])
    idx += 1
    x3 = tf.convert_to_tensor(params[idx])
    idx += 1


    return D_mod, S1_mod, S2_mod, v0_mod, eta, x1, x2, x3

# cost_function
def cost_function(params, n, fmt_data, Lors_true, HLs_true,coeffs,g_A, weight, central_point, retain):
    
    '''
    params: tf.Variable
    D_shape, S1_shape, S2_shape : int
    alpha_values, beta_values: list
    data_table: pd.DataFrame
    
    calculates the cost function by subtracting two Lorentzians
    
    '''
    
    D_mod, S1_mod, S2_mod, v0_mod, eta, x1, x2, x3 = modified_DS(params, n)
    
    
    total_cost = 0
    
    count = 0
    HLs_calc = []
    for idx, alpha in enumerate(fmt_data):

        M_true = D_mod + (float(alpha[0])-float(central_point[0])) * S1_mod \
            + (float(alpha[1]) - float(central_point[1])) * S2_mod
        

        eigenvalues, eigenvectors = tf.linalg.eigh(M_true)
        
        n_i = eigenvalues.shape[0]
        k_keep = int(round(retain * n_i))         # how many eigenvalues to keep
        k_keep = max(1, min(k_keep, n_i))         # safety: clamp between 1 and n
        
        left  = (n_i - k_keep) // 2               # starting index of the centered block
        right = left + k_keep                     # ending index (exclusive)
        
        eigenvalues  = eigenvalues[left:right]
        eigenvectors = eigenvectors[:, left:right]
        
        projections = tf.linalg.matvec(tf.transpose(eigenvectors), v0_mod)
        
        # Square each projection
        B = tf.square(projections)
        
        mask = tf.cast((eigenvalues > -10) & (eigenvalues < 15), dtype=tf.float64)

        # Apply the mask to zero out B where eigenvalue is negative
        B = B * mask
        

        #B = [tf.square(tf.tensordot(eigenvectors[:, i], v0_mod, axes=1)) for i in range(eigenvectors.shape[1])]
        Lor_true = tf.constant(Lors_true[count][:,1], dtype=tf.float64)

        #Generate the x values
        x = tf.constant(Lors_true[count][:,0], dtype=tf.float64)
        
        width = tf.sqrt(tf.square(eta) + tf.square(x1 + x2*float(alpha[0]) + x3*float(alpha[1])))
        

        # Use tf.map_fn to apply the give_me_Lorentzian function over the x values
        Lor = give_me_Lorentzian(x, eigenvalues, B, width)
        
        
        

        total_cost += tf.reduce_sum((Lor - Lor_true) ** 2)
        ''' Total cost modified to match previous definitions'''
        
        
        ''' Add half-lives to optimization as well'''

        hls = half_life_loss(eigenvalues, B, coeffs, g_A)
        HLs_calc.append(hls)
        
        total_cost += tf.constant(weight,dtype=tf.float64)*tf.reduce_sum((tf.math.log(hls) - tf.math.log(HLs_true[idx])) ** 2)

        count+=1
            
    return total_cost, Lor, Lor_true, x, HLs_calc, B, eigenvalues



def cost_function_only_HL(params, n, fmt_data, HLs_true, central_point):
    
    '''
    params: tf.Variable
    D_shape, S1_shape, S2_shape : int
    alpha_values, beta_values: list
    data_table: pd.DataFrame
    
    calculates the cost function by subtracting only the half-lives !
    
    '''


    
    D_mod, S1_mod, S2_mod = modified_DS_only_HL(params, n)

    
    total_cost = 0
    HLs_calc = []
    

    for idx, alpha in enumerate(fmt_data):

        M_true = D_mod + (float(alpha[0]) - float(central_point[0])) * S1_mod \
                       + (float(alpha[1]) - float(central_point[1])) * S2_mod
        
        

        eigenvalues, eigenvectors = tf.linalg.eigh(M_true)


        
        ''' Add half-lives to optimization as well'''
        log_hls = eigenvalues[int(n/2)] #tf.reduce_sum((eigenvalues))
        
        total_cost += (log_hls - np.log10(HLs_true[idx])) ** 2

        
        # save the half lives for CV check
        HLs_calc.append(10**log_hls)
            
    return total_cost, HLs_calc

def data_table_only_HL(fmt_data,coeffs, g_A, nucnam):
    '''
    
    Here split the dataset from "beta_decay_data" folder
    into: training set, validation set and test set
    use any ratio you like (e.g. 0.8 0.1 0.1)
    
    For the optimizatio use only training set, and after you finish
    test it on validation set
    
    returns also number of QRPA poles n_QRPA
    
    '''

    HLs = []
    

    for frmt in fmt_data:
        
        alpha = frmt[0]
        beta = frmt[1]

        # now calculate half-lives the old way
        file = np.loadtxt('../beta_decay_data_'+nucnam+'/excm_'+nucnam+'_'+beta+'_'+alpha+'.out')
        file = file[file[:,0]<del_nH]
        file = file[file[:,0]>-10]
        HLs.append(half_life_loss(file[:,0], file[:,1],coeffs, g_A))

     
    return HLs


def modified_DS_only_HL(params, n):
    '''

     
     added S1_shape & S2_shape 
     
     params: tf.Variable
     D_shape: int, shape of diagonal matrix
     S1_shape : int
     S2_shape : int
     
     given params, construct D, S1 and S2 matrices , 

    
    '''
    D_shape = (n,n)
    S1_shape = (n,n)
    S2_shape = (n,n)
    
    # initialize D, S1 and S2
    D_mod = tf.linalg.diag(params[:D_shape[0]])
    S1_mod = tf.zeros(S1_shape, dtype=tf.float64)
    S2_mod = tf.zeros(S2_shape, dtype=tf.float64)
    
    # construct S1 and S2 matrices
    upper_tri_indices1 = np.triu_indices(S1_shape[0])
    indices1 = tf.constant(list(zip(upper_tri_indices1[0], upper_tri_indices1[1])))
    S1_mod = tf.tensor_scatter_nd_update(S1_mod, indices1,\
            params[D_shape[0]:D_shape[0] + len(upper_tri_indices1[0])])
    
    S1_mod = S1_mod + tf.linalg.band_part(tf.transpose(S1_mod), -1, 0) - tf.linalg.diag(tf.linalg.diag_part(S1_mod))
    
    
    upper_tri_indices2 = np.triu_indices(S2_shape[0])
    indices2 = tf.constant(list(zip(upper_tri_indices2[0], upper_tri_indices2[1])))
    S2_mod = tf.tensor_scatter_nd_update(S2_mod, indices2 \
    , params[D_shape[0] + len(upper_tri_indices1[0]):D_shape[0] \
        + len(upper_tri_indices1[0])+len(upper_tri_indices2[0])])
    
    S2_mod = S2_mod + tf.linalg.band_part(tf.transpose(S2_mod), -1, 0) - tf.linalg.diag(tf.linalg.diag_part(S2_mod))
    
    
    return D_mod, S1_mod, S2_mod


       # generalized_eigen for M_true(a)
def generalized_eigen(D, S1, S2, alpha):
    M_true = D + float(alpha[0]) * S1 + float(alpha[1]) * S2
    eigenvalues, eigenvectors = eigh(M_true)
    return np.real(eigenvalues), np.real(eigenvectors) 




def plot_Lorentzian_for_idx(idx, test_set,n,params, coeffs, g_A):

    alpha = float(test_set[idx][0])
    beta = float(test_set[idx][1])
    
    
    Lors_test, HLs_test = data_table(test_set, coeffs, g_A)
    Lors_orig = Lors_test[idx]
    
    opt_D, opt_S1, opt_S2, opt_v0 = modified_DS(params, n)
    opt_eigenvalues, opt_eigenvectors = generalized_eigen(opt_D.numpy(), opt_S1.numpy(), opt_S2.numpy(), test_set[idx])
    opt_dot_products = [np.square(np.dot(opt_eigenvectors[:, i], opt_v0.numpy())) for i in range(opt_eigenvectors.shape[1])]
    
    
    
    fig, ax = plt.subplots()
    
    
    
    # plot the Lorentzian for the original data
    x = Lors_orig[:,0]
    opt_Lor = []
    for en in x:
        opt_Lor.append(give_me_Lorentzian(en,opt_eigenvalues,opt_dot_products,0.5))
    
    plt.plot(x, Lors_orig[:,1], 'b--',label='QRPA calculation')    
    plt.plot(x, opt_Lor, 'r-',label='emulated Lorentzian')
        
    
    ax.set_title(r'$V_0^{is}$ = '+str(round(alpha,1))+r', $g_0 = $'+str(round(beta,1)), size = 18)
    ax.legend(frameon = False)
    
    
    plt.xlabel(r'$\omega$ (MeV)', size = 18)
    plt.ylabel('$S$ (1/MeV)', size = 18)
    
    plt.annotate('${}^{74}$Ni', (0.2,0.7), xycoords='axes fraction', size = 22)

    plt.gca().tick_params(axis="y",direction="in", which = 'both', labelsize = 12)
    plt.gca().tick_params(axis="x",direction="in", which = 'both', labelsize = 12)
    
    plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(0.25))
    
    plt.ylim(0)
    plt.xlim(-6,0.782)
    
    plt.savefig('gamow_teller_strength_emulator.pdf', bbox_inches='tight')
    
    
def data_Lorentzian_for_idx(idx, test_set,n,params, coeffs, g_A):

    
    
    Lors_test, HLs_test = data_table(test_set, coeffs, g_A)
    Lors_orig = Lors_test[idx]
    
    opt_D, opt_S1, opt_S2, opt_v0 = modified_DS(params, n)
    opt_eigenvalues, opt_eigenvectors = generalized_eigen(opt_D.numpy(), opt_S1.numpy(), opt_S2.numpy(), test_set[idx])
    opt_dot_products = [np.square(np.dot(opt_eigenvectors[:, i], opt_v0.numpy())) for i in range(opt_eigenvectors.shape[1])]
    
    
    
    fig, ax = plt.subplots()
    
    
    
    # plot the Lorentzian for the original data
    x = Lors_orig[:,0]
    opt_Lor = []
    for en in x:
        opt_Lor.append(give_me_Lorentzian(en,opt_eigenvalues,opt_dot_products,0.5))
    
    
    return x, Lors_orig[:,1], opt_Lor
    
 
# def plot_half_lives(test_set,params,n, coeffs, g_A, central_point, nucnam, retain):
#     '''
#     Calculate half-lives for type 1 alg
#     '''
#     hl_guess = []
#     times = []
    
#     Lors_test, HLs_test = data_table(test_set, coeffs, g_A, nucnam)
    
#     D_mod, S1_mod, S2_mod, v0_mod, eta, x1, x2, x3 = modified_DS(params, n)
    
    
#     for idx in range(len(test_set)):
        
#         start = time.time()  # Start time
        
#         M_true = D_mod + (float(test_set[idx][0])-float(central_point[0])) * S1_mod \
#             + (float(test_set[idx][1]) - float(central_point[1])) * S2_mod
        

#         eigenvalues, eigenvectors = tf.linalg.eigh(M_true)
        
#         n_i = eigenvalues.shape[0]
#         k_keep = int(round(retain * n_i))         # how many eigenvalues to keep
#         k_keep = max(1, min(k_keep, n_i))         # safety: clamp between 1 and n
        
#         left  = (n_i - k_keep) // 2               # starting index of the centered block
#         right = left + k_keep                     # ending index (exclusive)
        
#         eigenvalues  = eigenvalues[left:right]
#         eigenvectors = eigenvectors[:, left:right]
        
#         projections = tf.linalg.matvec(tf.transpose(eigenvectors), v0_mod)
        
#         # Square each projection
#         B = tf.square(projections)
        
#         mask = tf.cast((eigenvalues > -10) & (eigenvalues < 15), dtype=tf.float64)

#         # Apply the mask to zero out B where eigenvalue is negative
#         B = B * mask
        

#         hls = half_life_loss(eigenvalues, B, coeffs, g_A)
        
        
#         end = time.time()  # Start time
        
#         times.append(end-start)
#         hl_guess.append(hls)
        
        
#     return hl_guess, HLs_test, times


# def plot_half_lives_only_HL(test_set,params,n, coeffs, g_A, central_point, nucnam):
#     '''
#     Note that this function is for type 2 Alg in the paper
#     '''
#     hl_guess = []
#     times = []
    
#     HLs_test = data_table_only_HL(test_set,coeffs,g_A, nucnam)
    
#     D_mod, S1_mod, S2_mod = modified_DS_only_HL(params,n)
    
#     for idx in range(len(test_set)):
        
#         start = time.time()  # Start time
        
        
        
#         M_true = D_mod + (float(test_set[idx][0]) - float(central_point[0])) * S1_mod \
#                        + (float(test_set[idx][1]) - float(central_point[1])) * S2_mod
        
        

#         eigenvalues, eigenvectors = tf.linalg.eigh(M_true)


        
#         ''' Add half-lives to optimization as well'''
#         log_hls = eigenvalues[int(n/2)]#tf.reduce_sum(tf.square(eigenvalues))
        

        
#         end = time.time()  # Start time
        
#         times.append(end-start)
#         hl_guess.append(10**log_hls)
        
        
#     return hl_guess, HLs_test, times

def plot_half_lives(test_set, params, n, coeffs, g_A, central_point, nucnam, retain, *, reps=5, warmup=1):
    """
    Calculate half-lives for type 1 algorithm, with robust per-point timing.
    Returns: hl_guess (list[float]), HLs_test (as returned by data_table), times (list[float])
    """
    import time
    import numpy as np
    import tensorflow as tf

    def robust_time(fn, reps=20, warmup=1):
        # Warm-up (excluded)
        for _ in range(warmup):
            out = fn()
            try:
                _ = float(out.numpy())
            except Exception:
                _ = float(out)
        # Timed reps
        ts = []
        last_out = None
        for _ in range(reps):
            t0 = time.perf_counter()
            out = fn()
            try:
                last_out = float(out.numpy())
            except Exception:
                last_out = float(out)
            t1 = time.perf_counter()
            ts.append(t1 - t0)
        return float(np.median(ts)), last_out

    hl_guess = []
    times = []

    # Ground truth (unchanged)
    Lors_test, HLs_test = data_table(test_set, coeffs, g_A, nucnam)

    # Precompute constants once
    a0 = float(central_point[0])
    b0 = float(central_point[1])

    # Build model parts once (move inside the loop if you want to include build cost per point)
    D_mod, S1_mod, S2_mod, v0_mod, eta, x1, x2, x3 = modified_DS(params, n)

    for idx in range(len(test_set)):
        a = float(test_set[idx][0])
        b = float(test_set[idx][1])

        def eval_point():
            # Build matrix
            M_true = (D_mod
                      + (a - a0) * S1_mod
                      + (b - b0) * S2_mod)

            # Eigendecomposition
            eigenvalues, eigenvectors = tf.linalg.eigh(M_true)

            # Keep centered block (retain fraction)
            n_i = eigenvalues.shape[0]
            k_keep = int(round(retain * n_i))
            k_keep = max(1, min(k_keep, n_i))
            left  = (n_i - k_keep) // 2
            right = left + k_keep

            eigenvalues  = eigenvalues[left:right]
            eigenvectors = eigenvectors[:, left:right]

            # Projections and strengths
            projections = tf.linalg.matvec(tf.transpose(eigenvectors), v0_mod)
            B = tf.square(projections)

            # Mask eigenvalues (dtype-safe)
            mask = tf.cast((eigenvalues > -10) & (eigenvalues < 15), dtype=eigenvalues.dtype)
            B = B * mask

            # Half-life (may return tf.Tensor)
            return half_life_loss(eigenvalues, B, coeffs, g_A)

        med_time, hls_val = robust_time(eval_point, reps=reps, warmup=warmup)
        hl_guess.append(hls_val)   # numeric float
        times.append(med_time)

    return hl_guess, HLs_test, times


def plot_half_lives_only_HL(test_set, params, n, coeffs, g_A, central_point, nucnam, *, reps=5, warmup=1):
    """
    Type 2 algorithm (only HL) with robust per-point timing.
    Returns: hl_guess (list[float]), HLs_test, times (list[float])
    """
    import time
    import numpy as np
    import tensorflow as tf

    def robust_time(fn, reps=20, warmup=1):
        # Warm-up (excluded)
        for _ in range(warmup):
            out = fn()
            try:
                _ = float(out.numpy())
            except Exception:
                _ = float(out)
        # Timed reps
        ts = []
        last_out = None
        for _ in range(reps):
            t0 = time.perf_counter()
            out = fn()
            try:
                last_out = float(out.numpy())
            except Exception:
                last_out = float(out)
            t1 = time.perf_counter()
            ts.append(t1 - t0)
        return float(np.median(ts)), last_out

    hl_guess = []
    times = []

    # Ground truth (unchanged)
    HLs_test = data_table_only_HL(test_set, coeffs, g_A, nucnam)

    # Precompute constants once
    a0 = float(central_point[0])
    b0 = float(central_point[1])

    # Build model parts once (move inside the loop if you want to include build cost per point)
    D_mod, S1_mod, S2_mod = modified_DS_only_HL(params, n)

    for idx in range(len(test_set)):
        a = float(test_set[idx][0])
        b = float(test_set[idx][1])

        def eval_point():
            M_true = (D_mod
                      + (a - a0) * S1_mod
                      + (b - b0) * S2_mod)

            eigenvalues, eigenvectors = tf.linalg.eigh(M_true)

            # Original behavior: take the middle eigenvalue (as log(T1/2))
            mid_idx = int(n/2)
            mid_idx = max(0, min(mid_idx, int(eigenvalues.shape[0]) - 1))
            log_hls = eigenvalues[mid_idx]

            # Return 10**log_hls (half-life)
            # Ensure we return a Tensor/number suitable for robust_time materialization
            try:
                return tf.pow(tf.constant(10.0, dtype=log_hls.dtype), log_hls)
            except Exception:
                # Fallback if dtype mismatch (shouldn't happen)
                return 10.0 ** log_hls

        med_time, hls_val = robust_time(eval_point, reps=reps, warmup=warmup)
        hl_guess.append(hls_val)   # numeric float
        times.append(med_time)

    return hl_guess, HLs_test, times
