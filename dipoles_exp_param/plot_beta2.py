#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 27 16:28:14 2025

@author: anteravlic
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

data = np.loadtxt('../dipoles_data/params.txt', usecols=(1,2,3))
data = data[data[:,1] > 0.5]

x = data[:, 0]
y = data[:, 1]
z = data[:, 2]

# Create grid to interpolate onto
xi = np.linspace(x.min(), x.max(), 500)
yi = np.linspace(y.min(), y.max(), 500)
X, Y = np.meshgrid(xi, yi)

# Interpolate z values on the grid
Z = griddata((x, y), z, (X, Y), method='cubic')  # use 'linear' or 'nearest' if cubic fails




# Plot
plt.figure(figsize=(7, 6))
pcm = plt.pcolormesh(X, Y, Z, shading='auto', cmap='Spectral', vmin = 0.296, vmax = 0.31)
plt.colorbar(pcm, label='z value')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Interpolated density plot of z(x, y)')

# Contour lines
contours = plt.contour(X, Y, Z, levels=10, colors='k', linewidths=0.8)
plt.clabel(contours, inline=True, fontsize=8)


plt.tight_layout()
plt.show()
