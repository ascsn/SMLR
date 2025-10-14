#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import helper
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import tensorflow as tf
import os
import re
from matplotlib.colors import LogNorm
import matplotlib.patches as patches
from numpy.polynomial.polynomial import Polynomial
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from scipy.interpolate import CubicSpline  # <- for cubic-spline interpolation

# ------------------ user config ------------------
A = 80
Z = 28
g_A = 1.2
nucnam = 'Ni_80'

n      = 13
retain = 0.9

# sweep
mode = 'beta'          # 'alpha' or 'beta'
fixed_value = '2.000'  # string to match filenames

# animation output + speed controls
SAVE_AS     = 'gif'     # 'gif' or 'mp4'
FPS         = 3         # ↓ smaller FPS = slower playback (affects saved file)
HOLD_FRAMES = 4         # repeat each logical frame this many times
PAUSE_TAIL  = 20        # extra hold on final frame
DPI         = 150
# -------------------------------------------------

# -------- phase space / params / train set --------
poly   = helper.fit_phase_space(0, Z, A, 15)
coeffs = Polynomial(poly).coef

params = np.loadtxt(f'../figs/data_beta/params_best_n{n}_retain{retain}.txt')

train_set = []
with open("../figs/data_beta/train_set.txt", "r") as f:
    for line in f:
        tup = tuple(map(str, line.strip().split(",")))
        train_set.append(tup)

# -------- load grid points from strength filenames --------
strength_dir = f'../beta_decay_data_{nucnam}/'
pattern = re.compile(r'lorm_' + re.escape(nucnam) + r'_([0-9.]+)_([0-9.]+)\.out')

combined = []
for fname in os.listdir(strength_dir):
    m = pattern.match(fname)
    if m:
        beta_val  = m.group(1)
        alpha_val = m.group(2)
        combined.append((alpha_val, beta_val))

if not combined:
    raise RuntimeError(f"No files matching {pattern.pattern} in {strength_dir}")

x_all = [float(a) for (a, b) in combined]
y_all = [float(b) for (a, b) in combined]

# --- centroid of grid ---
combined_ar   = np.array(combined, dtype=float)
centroid      = combined_ar.mean(axis=0)
distances     = np.linalg.norm(combined_ar - centroid, axis=1)
central_index = np.argmin(distances)
central_point = tuple(combined[central_index])
print('Central data point in train set:', central_point)

# -------- true strengths & half-lives at grid points --------
Lors, HLs = helper.data_table(combined, coeffs, g_A, nucnam)  # Lors[i] is ndarray-like

opt_strength = []
HLs_opt_list = []

for idx in range(len(combined)):
    alpha = float(combined[idx][0])
    beta  = float(combined[idx][1])

    opt_D, opt_S1, opt_S2, opt_v0, fold, x1, x2, x3 = helper.modified_DS(params, n)

    M_true = opt_D + (alpha - float(central_point[0])) * opt_S1 \
                    + (beta  - float(central_point[1])) * opt_S2

    eigenvalues, eigenvectors = tf.linalg.eigh(M_true)

    # center-truncate spectrum
    n_i   = int(eigenvalues.shape[0])
    k_keep = max(1, min(int(round(retain * n_i)), n_i))
    left   = (n_i - k_keep) // 2
    right  = left + k_keep

    ev_trunc  = eigenvalues[left:right]
    vec_trunc = eigenvectors[:, left:right]

    projections = tf.linalg.matvec(tf.transpose(vec_trunc), opt_v0)
    B = tf.square(projections)

    # window mask
    mask = tf.cast((ev_trunc > -10) & (ev_trunc < 15), dtype=tf.float64)
    B = B * mask

    x = tf.constant(Lors[idx][:, 0], dtype=tf.float64)
    width = tf.sqrt(tf.square(fold) + tf.square(x1 + x2*alpha + x3*beta))
    Lor   = helper.give_me_Lorentzian(x, ev_trunc, B, width)

    hls = helper.half_life_loss(ev_trunc, B, coeffs, g_A)

    opt_strength.append(Lor)
    HLs_opt_list.append(hls)

# numeric arrays
HLs_np     = np.array([float(v.numpy()) for v in HLs])
HLs_opt_np = np.array([float(v.numpy()) for v in HLs_opt_list])

HLs_error = np.abs(HLs_opt_np - HLs_np) / np.maximum(HLs_np, 1e-12)
HLs_error = np.clip(HLs_error, 1e-12, None)

# ------------- build a sweep order -------------
if mode == 'alpha':
    sweep = sorted([(a, b) for (a, b) in combined if b == fixed_value], key=lambda p: float(p[0]))
    param_label = r"$\alpha$"
    param_min = min(float(a) for (a, b) in sweep)
    param_max = max(float(a) for (a, b) in sweep)
elif mode == 'beta':
    sweep = sorted([(a, b) for (a, b) in combined if a == fixed_value], key=lambda p: float(p[1]))
    param_label = r"$\beta$"
    param_min = min(float(b) for (a, b) in sweep)
    param_max = max(float(b) for (a, b) in sweep)
else:
    raise ValueError(f"Unknown mode {mode}")

if not sweep:
    raise RuntimeError(f"No sweep points found for mode={mode} and fixed_value={fixed_value}")

print(f"Sweeping mode: {mode}, fixed value: {fixed_value}, frames (unique): {len(sweep)}")

# ------------- figure / axes -------------
fig, (ax_strength, ax_grid, ax_eigs) = plt.subplots(
    3, 1, figsize=(8, 14), dpi=DPI, gridspec_kw={'height_ratios': [1, 1, 1]}
)

fig.suptitle(f"PMM dimension n = {n}", fontsize=16, y=0.96)
fig.subplots_adjust(top=0.92, hspace=0.4)

# eigenvalue history (as numpy)
param_vals_for_eigs = []
eigs_full_hist  = []   # list of 1D arrays (full)
eigs_trunc_hist = []   # list of 1D arrays (truncated)

# ---- eigenvalues panel ----
ax_eigs.set_xlabel(param_label)
ax_eigs.set_ylabel("Eigenvalues")
ax_eigs.set_xlim(param_min, param_max)
ax_eigs.set_ylim(-20, 20)
ax_eigs.grid(True)
ax_eigs.set_title("Eigenvalue evolution")

# ---- strength panel ----
strength_xlim = (-10, 0.782)
strength_ylim = (0, 20)
ax_strength.set_xlim(*strength_xlim)
ax_strength.set_ylim(*strength_ylim)
ax_strength.set_xlabel("E [MeV]")
ax_strength.set_ylabel("Strength")
ax_strength.legend()

# ---- parameter grid / errors ----
sc = ax_grid.scatter(x_all, y_all, c=HLs_error, cmap='Spectral', norm=LogNorm(), marker='s')
marker_x, = ax_grid.plot([], [], 'kx', markersize=10, markeredgewidth=2)
ax_grid.set_xlabel(r"$\alpha$")
ax_grid.set_ylabel(r"$\beta$")
ax_grid.set_xlim(min(x_all)-0.1, max(x_all)+0.1)
ax_grid.set_ylim(min(y_all)-0.1, max(y_all)+0.1)
cbar = fig.colorbar(sc, ax=ax_grid, label=r"Relative error on $T_{1/2}$")

# train-set rectangle
train_alpha = [float(a) for (a, b) in train_set]
train_beta  = [float(b) for (a, b) in train_set]
train_rect = patches.Rectangle(
    (min(train_alpha), min(train_beta)),
    max(train_alpha)-min(train_alpha),
    max(train_beta)-min(train_beta),
    linewidth=1.5, edgecolor='black', facecolor='none'
)
ax_grid.add_patch(train_rect)

# ------------- animation callbacks -------------
def init():
    return tuple()

def animate(i):
    alpha_str, beta_str = sweep[i]  # i comes from frame_indices below
    idx = combined.index((alpha_str, beta_str))
    print(f"Frame {i}: alpha={alpha_str}, beta={beta_str}")

    alpha = float(alpha_str)
    beta  = float(beta_str)

    opt_D, opt_S1, opt_S2, opt_v0, fold, x1, x2, x3 = helper.modified_DS(params, n)

    M_true = opt_D + (alpha - float(central_point[0])) * opt_S1 \
                    + (beta  - float(central_point[1])) * opt_S2

    eigenvalues, eigenvectors = tf.linalg.eigh(M_true)

    n_i    = int(eigenvalues.shape[0])
    k_keep = max(1, min(int(round(retain * n_i)), n_i))
    left   = (n_i - k_keep) // 2
    right  = left + k_keep

    ev_full  = eigenvalues.numpy()
    ev_trunc = eigenvalues[left:right].numpy()
    vec_trunc = eigenvectors[:, left:right]

    projections = tf.linalg.matvec(tf.transpose(vec_trunc), opt_v0)
    B = (tf.square(projections)).numpy()
    mask = ((ev_trunc > -10) & (ev_trunc < 15)).astype(float)
    B = B * mask

    x = Lors[idx][:, 0]      # NumPy
    orig = Lors[idx][:, 1]   # NumPy

    eta_new = np.sqrt(float(fold.numpy())**2 +
                      (float(x1.numpy()) + float(x2.numpy())*alpha + float(x3.numpy())*beta)**2)

    opt_Lor = helper.give_me_Lorentzian(
        tf.constant(x, dtype=tf.float64),
        tf.constant(ev_trunc, dtype=tf.float64),
        tf.constant(B, dtype=tf.float64),
        tf.constant(eta_new, dtype=tf.float64)
    ).numpy()

    # ---- Strength panel (with cubic-spline interpolation + legend FOM/EM1) ----
    ax_strength.cla()
    ax_strength.set_xlim(*strength_xlim)
    ax_strength.set_ylim(*strength_ylim)
    ax_strength.set_xlabel("E [MeV]")
    ax_strength.set_ylabel("Strength")
    ax_strength.set_title(fr"$\alpha = {alpha_str},\ \beta = {beta_str}$")

    # Ensure strictly increasing x for spline
    sort_idx   = np.argsort(x)
    x_sorted   = x[sort_idx]
    fom_sorted = orig[sort_idx]      # Full-order model
    em1_sorted = opt_Lor[sort_idx]   # Emulator 1

    # Build cubic splines (no extrapolation outside data range)
    fom_spline = CubicSpline(x_sorted, fom_sorted, extrapolate=False)
    em1_spline = CubicSpline(x_sorted, em1_sorted, extrapolate=False)

    # Fine grid for smooth lines
    x_fine = np.linspace(x_sorted[0], x_sorted[-1], 1200)
    y_fom  = fom_spline(x_fine)
    y_em1  = em1_spline(x_fine)

    # Plot smooth curves with requested legend labels
    ax_strength.plot(x_fine, y_fom,  'k-',  label='FOM')  # full-order model
    ax_strength.plot(x_fine, y_em1, 'r--', label='EM1')   # emulator 1

    # Keep the stem plot for eigenvalue weights
    ax_strength.stem(ev_trunc, B, basefmt=" ", markerfmt='go', linefmt='g-', label='Weights')
    ax_strength.legend()

    # ---- Grid marker ----
    marker_x.set_data([alpha], [beta])

    # ---- Eigenvalue history ----
    param_vals_for_eigs.append(alpha if mode == 'alpha' else beta)
    eigs_full_hist.append(ev_full.copy())
    eigs_trunc_hist.append(ev_trunc.copy())

    ax_eigs.cla()
    ax_eigs.set_xlabel(param_label)
    ax_eigs.set_ylabel("Eigenvalues")
    ax_eigs.set_xlim(param_min, param_max)
    ax_eigs.set_ylim(-20, 20)
    ax_eigs.grid(True)
    ax_eigs.set_title("Eigenvalue evolution")

    p = np.array(param_vals_for_eigs)
    full_mat  = np.vstack(eigs_full_hist)
    trunc_mat = np.vstack(eigs_trunc_hist)

    for k in range(full_mat.shape[1]):
        ax_eigs.plot(p, full_mat[:, k], '-', color='gray', linewidth=1, alpha=1.0)
    for k in range(trunc_mat.shape[1]):
        ax_eigs.plot(p, trunc_mat[:, k], 'o-', markersize=3)

    return tuple()

# -------- build repeated frame indices for slower playback --------
frame_indices = [i for i in range(len(sweep)) for _ in range(max(1, int(HOLD_FRAMES)))]
frame_indices += [len(sweep)-1] * max(0, int(PAUSE_TAIL))

ani = FuncAnimation(
    fig, animate, init_func=init,
    frames=frame_indices, interval=700, repeat=False
)

# -------- save --------
if SAVE_AS.lower() == 'gif':
    writer = PillowWriter(fps=max(1, int(FPS)))
    out = f"animate_strength_evolution_sweep_{mode}_fixed_{fixed_value}_PMM{n}.gif"
    ani.save(out, writer=writer, dpi=DPI)
    print(f"Saved GIF: {out}  (FPS={FPS}, HOLD_FRAMES={HOLD_FRAMES}, PAUSE_TAIL={PAUSE_TAIL})")
elif SAVE_AS.lower() == 'mp4':
    writer = FFMpegWriter(fps=max(1, int(FPS)))
    out = f"animate_strength_evolution_sweep_{mode}_fixed_{fixed_value}_PMM{n}.mp4"
    ani.save(out, writer=writer, dpi=DPI)
    print(f"Saved MP4: {out}  (FPS={FPS}, HOLD_FRAMES={HOLD_FRAMES}, PAUSE_TAIL={PAUSE_TAIL})")
else:
    raise ValueError("SAVE_AS must be 'gif' or 'mp4'")
