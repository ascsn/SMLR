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
    
    mask = tf.cast((poles > 1) & (poles < 30), dtype=tf.float64)

    # Apply the mask to zero out B where eigenvalue is negative
    strength = strength * mask
    
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




def data_table(fmt_data):
    '''
    
    this loads Lorentzian and alphaD files
    
    '''
    strength = []
    alphaD = [] 
    

    for frmt in fmt_data:
        
        alpha = frmt[0]
        beta = frmt[1]

        # first open the file with the data
        file_strength = np.loadtxt('../dipoles_data/total_strength/strength_'+beta+'_'+alpha+'.out')
        file_alphaD = np.loadtxt('../dipoles_data/total_alphaD/alphaD_'+beta+'_'+alpha+'.out')
        
        file_strength = file_strength[file_strength[:,0] > 1]


        strength.append(file_strength)  
        alphaD.append(file_alphaD)

     
    return strength, alphaD



'''
    data table is now constructed for alpha & beta parameters
'''

def modified_DS_simple(params, n):
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

@tf.function
def calculate_alphaD(eigenvalues, B):
    
    mask = tf.cast((eigenvalues > 1) & (eigenvalues < 30), dtype=tf.float64)

    # Apply the mask to zero out B where eigenvalue is negative
    B = B * mask
    
    fac = tf.constant(8.0*np.pi*(7.29735e-3)*hqc/9.0 , dtype = tf.float64)
    val = tf.reduce_sum(B / eigenvalues)*fac
    
    return val
    


# cost_function
def cost_function(params, n, fmt_data, strength_true, alphaD_true, weight, fold):
    
    '''
    params: tf.Variable
    D_shape, S1_shape, S2_shape : int
    alpha_values, beta_values: list
    
    calculates the cost function by subtracting two Lorentzians
    and also alphaD
    
    fold: folding width for training
    
    '''
    
    D_mod, S1_mod, S2_mod, v0_mod = modified_DS(params,  n)
    
    total_cost = 0
    
    count = 0
    alphaD_calc = []
    for idx, alpha in enumerate(fmt_data):

        M_true = D_mod + float(alpha[0]) * S1_mod + float(alpha[1]) * S2_mod

        eigenvalues, eigenvectors = tf.linalg.eigh(M_true)
        
        #Get indices where eigenvalues are positive
        # positive_mask = eigenvalues_orig > 0
        
        # # Use tf.boolean_mask to filter
        # eigenvalues = tf.boolean_mask(eigenvalues_orig, positive_mask)
        # eigenvectors = tf.boolean_mask(eigenvectors_orig, positive_mask, axis=1)
        

        #B = compute_B(eigenvectors, v0_mod)
        #B = [tf.square(tf.tensordot(eigenvectors[:, i], v0_mod, axes=1)) for i in range(eigenvectors.shape[1])]
        # Compute dot product of each eigenvector (columns) with v0_mod
        projections = tf.linalg.matvec(tf.transpose(eigenvectors), v0_mod)
        
        # Square each projection
        B = tf.square(projections)
        
        mask = tf.cast((eigenvalues > 1) & (eigenvalues < 30), dtype=tf.float64)

        # Apply the mask to zero out B where eigenvalue is negative
        B = B * mask
    

        # these are the values from QFAM
        omega = tf.constant(strength_true[count][:,0], dtype=tf.float64)
        Lor_true = tf.constant(strength_true[count][:,1], dtype=tf.float64)


        # Use tf.map_fn to apply the give_me_Lorentzian function over the x values
        Lor = give_me_Lorentzian(omega, eigenvalues, B, fold)

        total_cost += tf.reduce_sum((Lor - Lor_true) ** 2)
        

        val_true = alphaD_true[count][2]
        val = calculate_alphaD(eigenvalues, B)
        #
        
#         # this has to be modified for the QFAM strength
        # x_mm = tf.linspace(tf.reduce_min(omega) , tf.reduce_max(omega) , 10000)
        # mm1 = tf.divide(give_me_Lorentzian(x_mm, eigenvalues, B, 0.02),x_mm)
        
        # dx = x_mm[1:] - x_mm[:-1]
        
        # avg_y = (mm1[1:] + mm1[:-1]) / 2.0
        # val = tf.reduce_sum(dx * avg_y)*8.0*np.pi*(7.29735e-3)*hqc/9.0
        
        total_cost += (val_true - val)**2*weight
        
        alphaD_calc.append(val)
        
        

        count+=1
            
    return total_cost, Lor, Lor_true, omega, alphaD_calc



def cost_function_only_alphaD(params, n, fmt_data, alphaD_true):
    
    '''
    params: tf.Variable
    D_shape, S1_shape, S2_shape : int
    alpha_values, beta_values: list
    
    calculates the cost function by subtracting two Lorentzians
    and also alphaD
    
    fold: folding width for training
    
    '''
    
    D_mod, S1_mod, S2_mod = modified_DS_simple(params,  n)
    
    total_cost = 0
    
    count = 0
    alphaD_calc = []
    for idx, alpha in enumerate(fmt_data):

        M_true = D_mod + float(alpha[0]) * S1_mod + float(alpha[1]) * S2_mod

        eigenvalues, eigenvectors = tf.linalg.eigh(M_true)
        
        

        val_true = alphaD_true[count][2]
        val = eigenvalues[0]
        #
    
        
        total_cost += (val_true - val)**2
        
        alphaD_calc.append(val)
        
        

        count+=1
            
    return total_cost, alphaD_calc






       # generalized_eigen for M_true(a)
#@tf.function
def generalized_eigen(D, S1, S2, alpha):
    M_true = D + float(alpha[0]) * S1 + float(alpha[1]) * S2
    eigenvalues, eigenvectors = eigh(M_true)
    return eigenvalues, eigenvectors 




def plot_Lorentzian_for_idx(idx, test_set,n,params, fold):

    alpha = float(test_set[idx][0])
    beta = float(test_set[idx][1])
    
    
    Lors_test, alphaD_test = data_table(test_set)
    Lors_orig = Lors_test[idx]
    
    opt_D, opt_S1, opt_S2, opt_v0 = modified_DS(params, n)
    opt_eigenvalues, opt_eigenvectors = generalized_eigen(opt_D.numpy(), opt_S1.numpy(), opt_S2.numpy(), test_set[idx])
    
    projections = tf.linalg.matvec(tf.transpose(opt_eigenvectors), opt_v0)
    
    # Square each projection
    B = tf.square(projections)
    
    mask = tf.cast((opt_eigenvalues > 1) &  (opt_eigenvalues < 30), dtype=tf.float64)

    # Apply the mask to zero out B where eigenvalue is negative
    opt_dot_products = B #* mask
    
    

    
    
    fig, ax = plt.subplots()
    
    
    
    # plot the Lorentzian for the original data
    x = Lors_orig[:,0]
        
    opt_Lor = give_me_Lorentzian(x, opt_eigenvalues, opt_dot_products, fold)
    
    plt.plot(x, Lors_orig[:,1], 'b--',label='FAM QRPA calculation')    
    plt.plot(x, opt_Lor, 'r-',label='emulated Lorentzian')
    
    # make the stem plot as well for the check
    #plt.stem(opt_eigenvalues, opt_dot_products)
        
    
    ax.set_title(r'$b_{TV}$ = '+str(round(alpha,1))+r', $d_{TV} = $'+str(round(beta,1)), size = 18)
    ax.legend(frameon = False)
    
    
    plt.xlabel('$\omega$ (MeV)', size = 18)
    plt.ylabel('$S$ (e$^2$ fm$^2$/MeV)', size = 18)
    
    plt.annotate('${}^{180}$Yb', (0.7,0.7), xycoords='axes fraction', size = 18)

    plt.gca().tick_params(axis="y",direction="in", which = 'both', labelsize = 12)
    plt.gca().tick_params(axis="x",direction="in", which = 'both', labelsize = 12)
    
    plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(1))
    
    #plt.stem(opt_eigenvalues, B)
    
    plt.xlim(0,30)
    plt.ylim(0,10)
    
    #plt.savefig('isovector_dipole_strength_emulator.pdf', bbox_inches='tight')
    
    return

def data_Lorentzian_for_idx(idx, test_set,n,params, fold):

    alpha = float(test_set[idx][0])
    beta = float(test_set[idx][1])
    
    
    Lors_test, alphaD_test = data_table(test_set)
    Lors_orig = Lors_test[idx]
    
    opt_D, opt_S1, opt_S2, opt_v0 = modified_DS(params, n)
    opt_eigenvalues, opt_eigenvectors = generalized_eigen(opt_D.numpy(), opt_S1.numpy(), opt_S2.numpy(), test_set[idx])
    
    projections = tf.linalg.matvec(tf.transpose(opt_eigenvectors), opt_v0)
    
    # Square each projection
    B = tf.square(projections)
    
    mask = tf.cast((opt_eigenvalues > 1) &  (opt_eigenvalues < 30), dtype=tf.float64)

    # Apply the mask to zero out B where eigenvalue is negative
    opt_dot_products = B #* mask
    

   
    x = Lors_orig[:,0]
        
    opt_Lor = give_me_Lorentzian(x, opt_eigenvalues, opt_dot_products, fold)
    

    
    return x, Lors_orig[:,1], opt_Lor
    
 
def plot_alphaD(test_set,params,n): 
    alphaD_guess = []
    times = []
    
    Lors_test, alphaD_test = data_table(test_set)
    alphaD_test = np.vstack(alphaD_test)
    
    for idx in range(len(test_set)):
        
        
        start = time.time()  # Start time
        
        opt_D, opt_S1, opt_S2, opt_v0 = modified_DS(params, n)
        
        
        
        M_true = opt_D + float(test_set[idx][0]) * opt_S1 + float(test_set[idx][1]) * opt_S2

        opt_eigenvalues, opt_eigenvectors = tf.linalg.eigh(M_true)
        
        #opt_eigenvalues, opt_eigenvectors = generalized_eigen(opt_D.numpy(), opt_S1.numpy(), opt_S2.numpy(), test_set[idx])
        
        projections = tf.linalg.matvec(tf.transpose(opt_eigenvectors), opt_v0)
        
        # Square each projection
        B = tf.square(projections)
        
        #mask = tf.cast((opt_eigenvalues > 1) &  (opt_eigenvalues < 30), dtype=tf.float64)

        # Apply the mask to zero out B where eigenvalue is negative
        #B = B * mask
        
        end = time.time()  # Start time
        
        alphaD_guess.append(calculate_alphaD(opt_eigenvalues, B))
        
        
        
        times.append(end-start)
        
        
    return alphaD_guess, alphaD_test[:,2], times


def plot_alphaD_simple(test_set,params,n): 
    alphaD_guess = []
    times = []
    
    Lors_test, alphaD_test = data_table(test_set)
    alphaD_test = np.vstack(alphaD_test)
    
    for idx in range(len(test_set)):
        
        start = time.time()  # Start time
        
        opt_D, opt_S1, opt_S2 = modified_DS_simple(params, n)
        
        
        
        #M_true = opt_D + float(test_set[idx][0]) * opt_S1 + float(test_set[idx][1]) * opt_S2

        opt_eigenvalues, opt_eigenvectors = generalized_eigen(opt_D.numpy(), opt_S1.numpy(), opt_S2.numpy(), test_set[idx])#tf.linalg.eigh(M_true)
        
        end = time.time()  # Start time
        
        alphaD_guess.append(opt_eigenvalues[0])
        
        
        
        times.append(end-start)
        
        
    return alphaD_guess, alphaD_test[:,2], times



