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

emass = 0.511 # MeV
alpha_c = 1/137
hbarc = 197.33 # MeV fm
compton = hbarc/emass # fm
kappa = 6147 #s
del_np = 1.293 # MeV
del_nH = 0.782 # MeV





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
    width = tf.constant(width, dtype=tf.float64)

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




def data_table(fmt_data, coeffs, g_A):
    '''
    
    Here split the dataset from "new_data" folder
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
        file = np.loadtxt('../new_data/lorm_Ni_74_'+beta+'_'+alpha+'.out')
        
        # normalize the Lorentzians
        #norm = np.sum(file[:,1])
        #file[:,1] = file[:,1]/norm
        
        #print(alpha, beta, norm**2)

        file = file[file[:,0]<del_nH]
        #file = file[file[:,0]<10]
        file = file[file[:,0]>-10]

        Lors.append(file)  
        
        # now calculate half-lives the old way
        file = np.loadtxt('../new_data/excm_Ni_74_'+beta+'_'+alpha+'.out')
        file = file[file[:,0]<del_nH]
        file = file[file[:,0]>-10]
        HLs.append(half_life_loss(file[:,0], file[:,1],coeffs, g_A))

     
    return Lors, HLs

'''
    data table is now constructed for alpha & beta parameters
'''



def modified_DS(params, n):
    '''

     added v0 in the mix
     
     added S1_shape & S2_shape 
     
     params: tf.Variable
     D_shape: int, shape of diagonal matrix
     S1_shape : int
     S2_shape : int
     
     given params, construct D, S1 and S2 matrices , 
     as well as the external field v0
    
    '''
    D_shape = (n,n)
    S1_shape = (n,n)
    S2_shape = (n,n)
    
    # initialize v0, D, S1 and S2
    v0_mod = tf.convert_to_tensor(params[:n])
    D_mod = tf.linalg.diag(params[n:D_shape[0]+n])
    S1_mod = tf.zeros(S1_shape, dtype=tf.float64)
    S2_mod = tf.zeros(S2_shape, dtype=tf.float64)
    
    # construct S1 and S2 matrices
    upper_tri_indices1 = np.triu_indices(S1_shape[0])
    indices1 = tf.constant(list(zip(upper_tri_indices1[0], upper_tri_indices1[1])))
    S1_mod = tf.tensor_scatter_nd_update(S1_mod, indices1,\
            params[D_shape[0]+n:D_shape[0] + len(upper_tri_indices1[0])+n])
    
    S1_mod = S1_mod + tf.linalg.band_part(tf.transpose(S1_mod), -1, 0) - tf.linalg.diag(tf.linalg.diag_part(S1_mod))
    
    
    upper_tri_indices2 = np.triu_indices(S2_shape[0])
    indices2 = tf.constant(list(zip(upper_tri_indices2[0], upper_tri_indices2[1])))
    S2_mod = tf.tensor_scatter_nd_update(S2_mod, indices2 \
    , params[D_shape[0] + len(upper_tri_indices1[0])+n:D_shape[0] \
        + len(upper_tri_indices1[0])+n+len(upper_tri_indices2[0])])
    
    S2_mod = S2_mod + tf.linalg.band_part(tf.transpose(S2_mod), -1, 0) - tf.linalg.diag(tf.linalg.diag_part(S2_mod))
    
    
    return D_mod, S1_mod, S2_mod, v0_mod

# cost_function
def cost_function(params, n, fmt_data, Lors_true, HLs_true,coeffs,g_A, weight, width):
    
    '''
    params: tf.Variable
    D_shape, S1_shape, S2_shape : int
    alpha_values, beta_values: list
    data_table: pd.DataFrame
    
    calculates the cost function by subtracting two Lorentzians
    
    '''
    
    D_mod, S1_mod, S2_mod, v0_mod = modified_DS(params, n)
    
    
    total_cost = 0
    
    count = 0
    HLs_calc = []
    for idx, alpha in enumerate(fmt_data):

        M_true = D_mod + float(alpha[0]) * S1_mod + float(alpha[1]) * S2_mod
        

        eigenvalues, eigenvectors = tf.linalg.eigh(M_true)
        
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
        

        # Use tf.map_fn to apply the give_me_Lorentzian function over the x values
        Lor = give_me_Lorentzian(x, eigenvalues, B, width)
        
        
        

        total_cost += tf.reduce_sum((Lor - Lor_true) ** 2)
        
        ''' Add half-lives to optimization as well'''

        hls = half_life_loss(eigenvalues, B, coeffs, g_A)
        HLs_calc.append(hls)
        
        total_cost += tf.constant(weight,dtype=tf.float64)*tf.reduce_sum((tf.math.log(hls) - tf.math.log(HLs_true[idx])) ** 2)

        count+=1
            
    return total_cost, Lor, Lor_true, x, HLs_calc, eigenvalues, B



def cost_function_only_HL(params, n, fmt_data, HLs_true):
    
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
    
    count = 0
    for idx, alpha in enumerate(fmt_data):

        M_true = D_mod + float(alpha[0]) * S1_mod + float(alpha[1]) * S2_mod
        
        

        eigenvalues, eigenvectors = tf.linalg.eigh(M_true)


        
        ''' Add half-lives to optimization as well'''

        log_hls = eigenvalues[0]
        
        total_cost += tf.reduce_sum((log_hls - tf.math.log(HLs_true[idx])) ** 2)

        count+=1
        
        # save the half lives for CV check
        HLs_calc.append(np.e**log_hls)
            
    return total_cost, HLs_calc

def data_table_only_HL(fmt_data,coeffs, g_A):
    '''
    
    Here split the dataset from "new_data" folder
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
        file = np.loadtxt('../new_data/excm_Ni_74_'+beta+'_'+alpha+'.out')
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
    
 
def plot_half_lives(test_set,params,n, coeffs, g_A): 
    hl_guess = []
    times = []
    
    Lors_test, HLs_test = data_table(test_set, coeffs, g_A)
    
    for idx in range(len(test_set)):
        
        start = time.time()  # Start time
        
        opt_D, opt_S1, opt_S2, opt_v0 = modified_DS(params, n)
        
        
        
        opt_eigenvalues, opt_eigenvectors = generalized_eigen(opt_D.numpy(), opt_S1.numpy(), opt_S2.numpy(), test_set[idx])
        opt_dot_products = [np.square(np.dot(opt_eigenvectors[:, i], opt_v0.numpy())) for i in range(opt_eigenvectors.shape[1])]
        end = time.time()  # Start time
        
        times.append(end-start)
        hl_guess.append( half_life_loss(opt_eigenvalues, opt_dot_products, coeffs, g_A))
        
        
    return hl_guess, HLs_test, times


def plot_half_lives_only_HL(test_set,params,n, coeffs, g_A):
    '''
    Note that this function is for type 2 Alg in the paper
    '''
    hl_guess = []
    times = []
    
    HLs_test = data_table_only_HL(test_set,coeffs,g_A)
    
    for idx in range(len(test_set)):
        
        start = time.time()  # Start time
        
        opt_D, opt_S1, opt_S2 = modified_DS_only_HL(params,n)
        
        
        
        eigenvalues, eigenvectors = generalized_eigen(opt_D.numpy(), opt_S1.numpy(), opt_S2.numpy(), test_set[idx])

        
        ''' Add half-lives to optimization as well'''

        log_hls = eigenvalues[0]
        
        end = time.time()  # Start time
        
        times.append(end-start)
        hl_guess.append(np.e**log_hls)
        
        
    return hl_guess, HLs_test, times
