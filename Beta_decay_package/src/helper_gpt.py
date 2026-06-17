# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import os
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

try:
    from smlr.core import ansatz as core_ansatz
    from smlr.core import fitting as core_fitting
    from smlr.core import numerics as core_numerics
except ModuleNotFoundError:  # pragma: no cover - source-tree execution before install
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from smlr.core import ansatz as core_ansatz
    from smlr.core import fitting as core_fitting
    from smlr.core import numerics as core_numerics

emass = 0.511 # MeV
alpha_c = 1/137
hbarc = 197.33 # MeV fm
compton = hbarc/emass # fm
kappa = 6147 #s
del_np = 1.293 # MeV
del_nH = 0.782 # MeV

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.dirname(MODULE_DIR)
REPO_ROOT = os.path.dirname(PACKAGE_ROOT)


def beta_data_root(nucnam):
    return os.environ.get("SMLR_BETA_DATA_DIR", os.path.join(REPO_ROOT, "beta_decay_80Ni"))


def beta_data_dir(nucnam):
    root = beta_data_root(nucnam)
    lorm_dir = os.path.join(root, "total_lorm")
    if os.path.isdir(lorm_dir):
        return lorm_dir
    return root


def beta_excm_dir(nucnam):
    root = beta_data_root(nucnam)
    excm_dir = os.path.join(root, "total_excm")
    if os.path.isdir(excm_dir):
        return excm_dir
    return root


def encode_initial_guess(random_initial_guess, E, B, n, retain, num_components=2):
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

    D_full, v0_full, _ = core_numerics.centered_spectrum_initialization(
        E, B, n, retain, dtype=np.float64, step=2.0
    )

    # ---- write back into packed vector (your layout) ----
    num_upper = n * (n + 1) // 2
    idx = 0
    idx += 1                                   # eta (keep as-is)
    params[idx:idx + n] = v0_full; idx += n    # v0
    params[idx:idx + n] = D_full; idx += n     # D diagonal
    idx += int(num_components) * num_upper     # S1..Sk (leave)
    # width parameters unchanged

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
    return core_fitting.fit_strength_with_tf_lorentzian(
        omega, y, n, eta, grid_M=grid_M, min_spacing=min_spacing, l2=l2,
        np_dtype=np.float64, tf_dtype=tf.float64, gap_floor=1e-6,
    )

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
    return core_numerics.give_me_lorentzian(energy, poles, strength, width, dtype=tf.float64)



def nec_mat(n, num_components=2):
    D = np.diag(np.random.uniform(1, 10, n))
    S_list = []
    for _ in range(int(num_components)):
        A = np.random.uniform(1, 10, (n, n))
        S_list.append(np.abs(A + A.T) / 2)
    return (D, *S_list)


def dataset_entry_params(entry):
    if (
        isinstance(entry, (tuple, list))
        and len(entry) == 2
        and isinstance(entry[1], (str, os.PathLike))
    ):
        entry = entry[0]
    return tuple(float(v) for v in entry)


def dataset_entry_path(entry):
    if (
        isinstance(entry, (tuple, list))
        and len(entry) == 2
        and isinstance(entry[1], (str, os.PathLike))
    ):
        return os.fspath(entry[1])
    return None


def data_table(fmt_data, coeffs, g_A, nucnam=None, *, strength_window=None):
    '''
    Here split the dataset from "beta_decay_data" folder
    into: training set, validation set and test set
    
    For the optimization use only training set, and after you finish
    test it on validation set
    
    returns also number of QRPA poles n_QRPA
    '''
    Lors = []
    HLs = []
    
    for frmt in fmt_data:
        params = dataset_entry_params(frmt)
        alpha = params[0]
        beta = params[1] if len(params) > 1 else None

        # first open the file with the data
        strength_path = dataset_entry_path(frmt)
        if strength_path is None:
            if nucnam is None or beta is None:
                raise ValueError("Old beta-decay data needs nucnam and two parameters.")
            strength_path = os.path.join(beta_data_dir(nucnam), f"lorm_{nucnam}_{beta}_{alpha}.out")
        file = np.loadtxt(strength_path)
        
        # normalize the Lorentzians
        #norm = np.sum(file[:,1])
        #file[:,1] = file[:,1]/norm
        
        #print(alpha, beta, norm**2)

        if strength_window is None:
            if nucnam is not None:
                file = file[file[:,0]<del_nH]
            file = file[file[:,0]>0]
        else:
            lo, hi = strength_window
            if lo is not None:
                file = file[file[:, 0] >= float(lo)]
            if hi is not None:
                file = file[file[:, 0] <= float(hi)]

        Lors.append(file)  
        
        # now calculate half-lives the old way
        if nucnam is None:
            HLs.append(tf.constant(1.0, dtype=tf.float64))
        else:
            if strength_path is not None and os.path.basename(strength_path).startswith(f"lorm_{nucnam}_"):
                excm_name = os.path.basename(strength_path).replace(f"lorm_{nucnam}_", f"excm_{nucnam}_", 1)
                excm_path = os.path.join(beta_excm_dir(nucnam), excm_name)
            else:
                excm_path = os.path.join(beta_excm_dir(nucnam), f"excm_{nucnam}_{beta}_{alpha}.out")
            file = np.loadtxt(excm_path)
            file = file[file[:,0]<del_nH]
            file = file[file[:,0]>0]
            HLs.append(half_life_loss(file[:,0], file[:,1],coeffs, g_A))

    return Lors, HLs

def modified_DS(params, n):
    ''''
    Build PMM matrices:
    M = D + \alpha*S1 + \beta*S2
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

    D_mod = tf.linalg.diag(params[idx:idx+n])
    idx += n

    num_upper1 = n * (n + 1) // 2
    S1_mod = core_ansatz.sym_from_upper(params[idx:idx+num_upper1], n, dtype=tf.float64)
    idx += num_upper1

    num_upper2 = n * (n + 1) // 2
    S2_mod = core_ansatz.sym_from_upper(params[idx:idx+num_upper2], n, dtype=tf.float64)
    idx += num_upper2
    
    # Add new learned params x1 and x2 and x3
    x1 = tf.convert_to_tensor(params[idx])
    idx += 1
    x2 = tf.convert_to_tensor(params[idx])
    idx += 1
    x3 = tf.convert_to_tensor(params[idx])
    idx += 1

    return D_mod, S1_mod, S2_mod, v0_mod, eta, x1, x2, x3


def modified_DS_general(params, n, num_components=2):
    idx = 0
    eta = tf.convert_to_tensor(params[idx])
    idx += 1

    v0_mod = tf.convert_to_tensor(params[idx:idx+n])
    idx += n

    D_mod = tf.linalg.diag(params[idx:idx+n])
    idx += n

    num_upper = n * (n + 1) // 2
    S_list = []
    for _ in range(int(num_components)):
        S_list.append(core_ansatz.sym_from_upper(params[idx:idx+num_upper], n, dtype=tf.float64))
        idx += num_upper

    width_params = tf.convert_to_tensor(params[idx:idx + int(num_components) + 1], dtype=tf.float64)
    if int(width_params.shape[0]) != int(num_components) + 1:
        raise ValueError(
            f"Expected {int(num_components) + 1} width parameters for "
            f"{num_components} components, got {int(width_params.shape[0])}."
        )
    return D_mod, S_list, v0_mod, eta, width_params


def normalized_coordinates(point, central_point, coordinate_scales=None):
    point = dataset_entry_params(point)
    central_point = tuple(float(v) for v in central_point)
    if coordinate_scales is None:
        return tuple(float(value) - float(center) for value, center in zip(point, central_point))

    coordinate_scales = tuple(float(v) for v in coordinate_scales)
    if len(point) != len(coordinate_scales):
        raise ValueError(
            f"Point has {len(point)} components, but coordinate scales has {len(coordinate_scales)}."
        )
    return tuple(
        (float(value) - float(center)) / float(scale)
        for value, center, scale in zip(point, central_point, coordinate_scales)
    )


def linear_matrix(D_mod, S_list, point, central_point, coordinate_scales=None):
    q_point = normalized_coordinates(point, central_point, coordinate_scales)
    if len(q_point) != len(S_list):
        raise ValueError(f"Point has {len(q_point)} components, but model has {len(S_list)} S matrices.")

    M_true = D_mod
    for value, S_mod in zip(q_point, S_list):
        M_true = M_true + float(value) * S_mod
    return M_true


def affine_width(eta, width_params, point, central_point=None, coordinate_scales=None):
    if central_point is None:
        point = dataset_entry_params(point)
        width_linear = tf.constant(point, dtype=tf.float64)
    else:
        width_linear = tf.constant(
            normalized_coordinates(point, central_point, coordinate_scales),
            dtype=tf.float64,
        )
    raw_width = width_params[0] + tf.reduce_sum(width_params[1:] * width_linear)
    return tf.sqrt(tf.square(eta) + tf.square(raw_width))

# cost_function
# def cost_function(params, n, fmt_data, Lors_true, HLs_true,coeffs,g_A, weight, central_point, retain):
    
#     '''
#     params: tf.Variable
#     D_shape, S1_shape, S2_shape : int
#     alpha_values, beta_values: list
#     data_table: pd.DataFrame
    
#     calculates the cost function by subtracting two Lorentzians
    
#     '''
    
#     D_mod, S1_mod, S2_mod, v0_mod, eta, x1, x2, x3 = modified_DS(params, n)
    
    
#     total_cost = 0
    
#     count = 0
#     HLs_calc = []
#     for idx, alpha in enumerate(fmt_data):

#         M_true = D_mod + (float(alpha[0])-float(central_point[0])) * S1_mod \
#             + (float(alpha[1]) - float(central_point[1])) * S2_mod
        

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
        

#         #B = [tf.square(tf.tensordot(eigenvectors[:, i], v0_mod, axes=1)) for i in range(eigenvectors.shape[1])]
#         Lor_true = tf.constant(Lors_true[count][:,1], dtype=tf.float64)

#         #Generate the x values
#         x = tf.constant(Lors_true[count][:,0], dtype=tf.float64)
        
#         width = tf.sqrt(tf.square(eta) + tf.square(x1 + x2*float(alpha[0]) + x3*float(alpha[1])))
        

#         # Use tf.map_fn to apply the give_me_Lorentzian function over the x values
#         Lor = give_me_Lorentzian(x, eigenvalues, B, width)
        
        
        

#         total_cost += tf.reduce_sum((Lor - Lor_true) ** 2)
#         ''' Total cost modified to match previous definitions'''
        
        
#         ''' Add half-lives to optimization as well'''

#         hls = half_life_loss(eigenvalues, B, coeffs, g_A)
#         HLs_calc.append(hls)
        
#         total_cost += tf.constant(weight,dtype=tf.float64)*tf.reduce_sum((tf.math.log(hls) - tf.math.log(HLs_true[idx])) ** 2)

#         count+=1
            
#     return total_cost, Lor, Lor_true, x, HLs_calc, B, eigenvalues


'''
Changed definition of cost function for strength
'''
# helper: trapezoidal integral in TF (float64)
def tf_trapz(y, x):
    dx  = x[1:] - x[:-1]                       # (N-1,)
    avg = 0.5 * (y[:-1] + y[1:])               # (N-1,)
    return tf.reduce_sum(avg * dx)             # scalar (float64)

# cost_function
def cost_function(params, n, fmt_data, Lors_true, HLs_true, coeffs, g_A, weight, central_point, retain,
                  num_components=2, fixed_width=None, coordinate_scales=None):

    D_mod, S_list, v0_mod, eta, width_params = modified_DS_general(params, n, num_components)

    total_cost = tf.constant(0.0, dtype=tf.float64)
    HLs_calc   = []
    count      = 0

    EPS_DEN = tf.constant(1e-16, dtype=tf.float64)  # protect against /0

    for idx, alpha in enumerate(fmt_data):
        M_true = linear_matrix(D_mod, S_list, alpha, central_point, coordinate_scales)

        eigenvalues, eigenvectors = tf.linalg.eigh(M_true)

        # keep centered fraction of spectrum
        n_i    = eigenvalues.shape[0]
        k_keep = int(round(retain * n_i))
        k_keep = max(1, min(k_keep, n_i))
        left   = (n_i - k_keep) // 2
        right  = left + k_keep
        eigenvalues  = eigenvalues[left:right]
        eigenvectors = eigenvectors[:, left:right]

        # projections and strengths (B >= 0)
        projections = tf.linalg.matvec(tf.transpose(eigenvectors), v0_mod)
        B = tf.square(projections)

        # mask eigenvalues outside window
        mask = tf.cast((eigenvalues > 0) & (eigenvalues < 30), dtype=tf.float64)
        B = B * mask

        # true spectrum (E grid x, values S_true)
        Lor_true = tf.constant(Lors_true[count][:, 1], dtype=tf.float64)
        x        = tf.constant(Lors_true[count][:, 0], dtype=tf.float64)

        # width(E; alpha) and predicted spectrum
        if fixed_width is None:
            width = affine_width(eta, width_params, alpha, central_point, coordinate_scales)
        else:
            width = tf.constant(float(fixed_width), dtype=tf.float64)
        Lor   = give_me_Lorentzian(x, eigenvalues, B, width)

        # ---------- normalized L2(E) loss ----------
        diff = Lor - Lor_true
        numer = tf_trapz(tf.square(diff), x)          # ∫ (Ŝ - S)^2 dE
        denom = tf_trapz(tf.square(Lor_true), x)      # ∫ S^2 dE
        spec_loss = numer / (denom + EPS_DEN)
        total_cost += spec_loss
        # ------------------------------------------

        # ---------- half-life term ----------
        if float(weight) != 0.0:
            hls = half_life_loss(eigenvalues, B, coeffs, g_A)
            HLs_calc.append(hls)
            total_cost += tf.constant(weight, dtype=tf.float64) * \
                          tf.reduce_sum((tf.math.log(hls) - tf.math.log(HLs_true[idx])) ** 2)
        else:
            HLs_calc.append(tf.constant(1.0, dtype=tf.float64))
        # ------------------------------------

        count += 1

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
                       + (float(alpha[1]) - float(central_point[1])) * S2_mod \
                       + (float(alpha[2]) - float(central_point[0])) * S3_mod \
                       + (float(alpha[3]) - float(central_point[0])) * S4_mod
    
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
        file = np.loadtxt(os.path.join(beta_excm_dir(nucnam), f"excm_{nucnam}_{beta}_{alpha}.out"))
        file = file[file[:,0]<del_nH]
        file = file[file[:,0]>0]
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
    # initialize D, S1 and S2
    D_mod = tf.linalg.diag(params[:n])
    num_upper = n * (n + 1) // 2
    s1_start = n
    s2_start = s1_start + num_upper
    S1_mod = core_ansatz.sym_from_upper(params[s1_start:s2_start], n, dtype=tf.float64)
    S2_mod = core_ansatz.sym_from_upper(params[s2_start:s2_start + num_upper], n, dtype=tf.float64)
    
    return D_mod, S1_mod, S2_mod


# generalized_eigen for M_true(a)
def generalized_eigen(D, S1, S2, alpha):
    M_true = D + float(alpha[0]) * S1 + float(alpha[1]) * S2 + float(alpha[2]) * S3 + float(alpha[3]) * S4
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
            mask = tf.cast((eigenvalues > 0) & (eigenvalues < 30), dtype=eigenvalues.dtype)
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
