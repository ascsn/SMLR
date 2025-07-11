#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 11:30:06 2025

@author: anteravlic
"""

import numpy as np
import matplotlib.pyplot as plt


# split the data into test set and so on

# # alpha_values = np.linspace(0.400, 0.850, 10) #dipoles_data
# alpha_values = np.linspace(0.200, 2.000, 15) #dipoles_data_all
# formatted_alpha_values = [f"{num:.4f}" for num in alpha_values]

# #beta_values = np.linspace(1.4, 2.0, 13) #dipoles_data
# beta_values = np.linspace(1.0, 4.5, 15) #dipoles_data_all
# formatted_beta_values = [f"{num:.4f}" for num in beta_values]

# for alpha in formatted_alpha_values:
#     for beta in formatted_beta_values:
        
#         # first open the file with the data
#         file_strength = np.loadtxt('../dipoles_data_all/total_strength/strength_'+beta+'_'+alpha+'.out')
#         file_alphaD = np.loadtxt('../dipoles_data_all/total_alphaD/alphaD_'+beta+'_'+alpha+'.out')
        
#         file_strength = file_strength[file_strength[:,0] > 1]
        
#         plt.plot(file_strength[:,0], file_strength[:,1], alpha = 0.1, color = 'black')


# plt.xlim(3,25)
# plt.ylim(0)

# plt.xlabel('$\omega$ (MeV)', size = 16)
# plt.ylabel('$S$ ($e^2$fm$^2$/MeV)', size = 16)
# plt.tight_layout()

# plt.annotate('${}^{180}$Yb', (0.7,0.7), xycoords='axes fraction', size = 18)

# plt.gca().tick_params(axis="y",direction="in", which = 'both', labelsize = 12)
# plt.gca().tick_params(axis="x",direction="in", which = 'both', labelsize = 12)

#plt.savefig('isovector_dipole_variation.pdf', bbox_inches='tight')







strength_dir = '../dipoles_data_all/total_strength/'
alphaD_dir = '../dipoles_data_all/total_alphaD/'

# Pattern for strength files: strength_beta_alpha.out
pattern = re.compile(r'strength_([0-9.]+)_([0-9.]+)\.out')
pattern_alphaD = re.compile(r'alphaD_([0-9.]+)_([0-9.]+)\.out')

all_pairs = []
for fname in os.listdir(strength_dir):
    match = pattern.match(fname)
    if match:
        beta_val = match.group(1)
        alpha_val = match.group(2)
        all_pairs.append((beta_val, alpha_val))
            
filtered = [
    (beta, alpha) for (beta, alpha) in all_pairs
    if 1.5 <= float(beta_val := beta) <= 4.0
    if 0.4 <= float(alpha_val := alpha) <= 1.8
]

for beta, alpha in filtered:
    fstr = os.path.join(strength_dir, f'strength_{beta}_{alpha}.out')
    data = np.loadtxt(fstr)
    mask = data[:,0] > 1.0
    plt.plot(data[mask, 0], data[mask, 1], alpha=0.1, color='black')
    #print(f"Plotted strength for beta = {beta}, alpha={alpha}")



plt.xlim(3, 25)
plt.xlabel('$\omega$ (MeV)', size = 16)
plt.ylabel('$S$ ($e^2$fm$^2$/MeV)', size = 16)
plt.tight_layout()
plt.show()








# pick out plotting parameters
# n_dim = eigs_beta.shape[0]
# α0, β0 = 2.0, 3.0
# i0 = np.argmin(np.abs(α-α0))
# j0 = np.argmin(np.abs(β-β0))

fig, axs = plt.subplots(1, 3, figsize=(15, 4))

# (a) Spectrum + sticks + Lorentzian
# ax = axs[0]
# ax.plot(ω_true, L_true,    'k-', label="Original")
# ax.plot(ω_true, L_fit,    'r--', label="PMM-Lorentzian")
# ax.vlines(eigs_curr, ymin=0, ymax=max(L_true)*1.1,
#           color='g', alpha=0.7, lw=1)
# ax.scatter(eigs_curr, np.zeros_like(eigs_curr),
#            marker='o', color='g', zorder=5)
# ax.set_xlabel("E [MeV]")
# ax.set_ylabel("Strength")
# ax.set_title(f"PMM dimension n = {n_dim}\nα = {α0:.4f}, β = {β0:.4f}")
# ax.legend()





# (b) α–β grid colored by rel. error
ax = axs[1]
# make a meshgrid of points
A, B = np.meshgrid(α, β, indexing='xy')
sc = ax.scatter(A.flatten(), B.flatten(), c=rel_err.flatten(),
                s=50, cmap='viridis', norm=plt.LogNorm())
# draw training rectangle
α_train = (min(α), max(α))
β_train = (min(β), max(β))
rect = Rectangle((α_train[0], β_train[0]),
                 α_train[1]-α_train[0],
                 β_train[1]-β_train[0],
                 fill=False, lw=1.5, edgecolor='black')
ax.add_patch(rect)
# mark current point
ax.plot(α0, β0, 'x', ms=10, mew=2, color='white')
ax.set_xlabel("α")
ax.set_ylabel("β")
cbar = fig.colorbar(sc, ax=ax)
cbar.set_label("Relative error on α_D")
ax.set_title("Grid error map")






# (c) Eigenvalue evolution vs β
# ax = axs[2]
# for idx in range(n_dim):
#     ax.plot(β, eigs_beta[idx, :], lw=1)
# ax.set_xlabel("β")
# ax.set_ylabel("Eigenvalues")
# ax.set_title("Eigenvalue evolution")

# plt.tight_layout()
# plt.show()