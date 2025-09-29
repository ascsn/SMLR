#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 09:24:50 2025

@author: anteravlic
"""

import numpy as np
import helper

n = 15
retain = 0.9

params = np.loadtxt(f'params_best_n{n}_retain{retain}.txt')


D_mod, S1_mod, S2_mod, v0_mod, eta, x1, x2, x3 = helper.modified_DS(params, n)