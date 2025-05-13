#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 18:17:51 2025

@author: anteravlic
"""

import helper
import numpy as np
from numpy.polynomial.polynomial import Polynomial
import matplotlib.pyplot as plt


A = 74
Z = 28
g_A = 1.2


'''
Construction of phase space integrals
'''

poly = helper.fit_phase_space(0, Z, A, 15)
coeffs = Polynomial(poly).coef

alpha_values = np.linspace(0,1.5,16)
formatted_alpha_values = [f"{num:.3f}" for num in alpha_values]
beta_values = np.linspace(0,1.,11)
formatted_beta_values = [f"{num:.3f}" for num in beta_values]


hls_old = []
hls_new = []

for beta in formatted_beta_values:
    for alpha in formatted_alpha_values:


        data_hl = np.loadtxt('../new_data/half_life_Ni_74_'+beta+'_'+alpha+'.txt')
        data_excm = np.loadtxt('../new_data/excm_Ni_74_'+beta+'_'+alpha+'.out')
        eigenvalues = data_excm[:,0]
        B = data_excm[:,1]
        
        hls = helper.half_life_loss(eigenvalues, B,coeffs,g_A)
        
        hls_old.append(data_hl)
        hls_new.append(hls.numpy())


plt.plot(hls_old)
plt.plot(hls_new)

plt.yscale('log')