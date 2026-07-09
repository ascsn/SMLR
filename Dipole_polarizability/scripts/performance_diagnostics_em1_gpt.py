#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.patches as patches
from matplotlib.colors import LogNorm
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import helper_gpt as helper_gpt


"""
Evaluate Emulator 1 on the current helper_gpt/main_gpt workflow.

This script:
1. Loads the same dataset structure used in training.
2. Reconstructs predicted spectra and alpha_D values from best_params_global.txt.
3. Shows one detailed example spectrum and pole strengths.
4. Reports a global RMSE over all spectra.
5. Plots predicted-vs-true alpha_D.
6. Plots a parameter-space map of relative alpha_D error.
"""


# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
n = 13
retain = 0.6
detail_idx = 60

params = np.loadtxt("runs_em1/best_params_global.txt").astype(np.float32)

strength_dir = "data/nuclear/160Yb_2d/total_strength"
alphaD_dir = "data/nuclear/160Yb_2d/total_alphaD"
strength_regex = r"strength_(?P<p2>[0-9.]+)_(?P<p1>[0-9.]+)\.out"
alphaD_regex = None
filter_ranges = {"p1": [0.4, 1.8], "p2": [1.5, 4.0]}

config = helper_gpt.AnsatzConfig(
    n=n,
    n_params=2,
    ansatz="paper_dipole",
    width_model="affine",
    use_vector_terms=True,
)


# -----------------------------------------------------------------------------
# Load dataset
# -----------------------------------------------------------------------------
dataset = helper_gpt.load_dataset(
    strength_dir=strength_dir,
    alphaD_dir=alphaD_dir,
    strength_regex=strength_regex,
    alphaD_regex=alphaD_regex,
    filter_ranges=filter_ranges,
    central_point=None,
)

param_values_tf = tf.convert_to_tensor(dataset.param_values, dtype=tf.float32)
central_point_tf = tf.convert_to_tensor(dataset.central_point, dtype=tf.float32)
params_tf = tf.convert_to_tensor(params, dtype=tf.float32)

print("Loaded dataset")
print("  param names:", dataset.param_names)
print("  samples:", len(dataset.strengths))
print("  central point:", dataset.central_point.tolist())


# -----------------------------------------------------------------------------
# Build emulator outputs for all samples
# -----------------------------------------------------------------------------
M_batch, v_batch, eta_batch, _ = helper_gpt.build_model_matrices_and_vectors(
    params=params_tf,
    config=config,
    param_values=param_values_tf,
    central_point=central_point_tf,
)

eigenvalues, eigenvectors = tf.linalg.eigh(M_batch)

n_i = int(eigenvalues.shape[1])
k_keep = int(round(retain * n_i))
k_keep = max(1, min(k_keep, n_i))

left = (n_i - k_keep) // 2
right = left + k_keep

eigvals_kept = eigenvalues[:, left:right]
eigvecs_kept = eigenvectors[:, :, left:right]

proj = tf.matmul(tf.transpose(eigvecs_kept, perm=[0, 2, 1]), v_batch[:, :, None])
proj = tf.squeeze(proj, axis=-1)
B_batch = tf.square(proj)

omega = dataset.strengths[0][:, 0].astype(np.float32)
omega_tf = tf.convert_to_tensor(omega, dtype=tf.float32)

opt_strength = []
alphaD_opt = []
opt_eigs = []
opt_Bs = []

for i in range(len(dataset.strengths)):
    y_pred = helper_gpt.give_me_Lorentzian(
        energy=omega_tf,
        poles=eigvals_kept[i],
        strength=B_batch[i],
        width=eta_batch[i] / 2.0,
    ).numpy()

    opt_strength.append(y_pred)
    opt_eigs.append(eigvals_kept[i].numpy())
    opt_Bs.append(B_batch[i].numpy())
    alphaD_opt.append(
        helper_gpt.calculate_alphaD(
            eigvals_kept[i].numpy(),
            B_batch[i].numpy()
        )
    )

opt_strength = np.asarray(opt_strength)
alphaD_opt = np.asarray(alphaD_opt)
alphaD_true = np.asarray(dataset.alphaD_values)


# -----------------------------------------------------------------------------
# Detailed example spectrum
# -----------------------------------------------------------------------------
plt.figure(1)

x = dataset.strengths[detail_idx][:, 0]
y_true = dataset.strengths[detail_idx][:, 1]

plt.plot(x, opt_strength[detail_idx], label="pred")
plt.plot(x, y_true, ls="-", label="true")
plt.stem(opt_eigs[detail_idx], opt_Bs[detail_idx], basefmt=" ")

print("Detailed point:", dataset.param_values[detail_idx])
print("alpha_D true / pred:", alphaD_true[detail_idx], alphaD_opt[detail_idx])
print("eta:", float(eta_batch[detail_idx].numpy()))

plt.title(f"Emulator 1 detailed check (n={n}, retain={retain})")
plt.ylim(0, 8)
plt.legend()


# -----------------------------------------------------------------------------
# Global RMSE over strength curves
# -----------------------------------------------------------------------------
rmse = np.sqrt(np.sum([
    np.sum((dataset.strengths[i][:, 1] - opt_strength[i]) ** 2)
    for i in range(len(dataset.strengths))
]))
print("RMSE:", rmse)


# -----------------------------------------------------------------------------
# alpha_D predicted vs true
# -----------------------------------------------------------------------------
plt.figure(2, figsize=(6, 4))
plt.scatter(alphaD_true, alphaD_opt)

for i in range(len(alphaD_opt)):
    plt.text(alphaD_true[i], alphaD_opt[i], str(i), fontsize=9, ha="right", va="bottom")

xline = np.linspace(min(np.min(alphaD_true), np.min(alphaD_opt)),
                    max(np.max(alphaD_true), np.max(alphaD_opt)), 100)
plt.plot(xline, xline, color="black")
plt.xlabel(r"True $\alpha_D$")
plt.ylabel(r"Predicted $\alpha_D$")
plt.title("Emulator 1")


# -----------------------------------------------------------------------------
# Parameter-space relative error map
# -----------------------------------------------------------------------------
plt.figure(3)

x = dataset.param_values[:, 0]
y = dataset.param_values[:, 1]
rel_err = np.abs(alphaD_opt - alphaD_true) / np.maximum(np.abs(alphaD_true), 1e-8)

plt.scatter(x, y, c=rel_err, marker="s", cmap="Spectral", norm=LogNorm())
plt.colorbar(label=r"Relative error $\alpha_D$")

for idx, (xi, yi) in enumerate(zip(x, y)):
    plt.text(xi, yi, str(idx), ha="center", va="center", fontsize=8, color="black")

train_alpha = x
train_beta = y

alpha_min = np.min(train_alpha)
alpha_max = np.max(train_alpha)
beta_min = np.min(train_beta)
beta_max = np.max(train_beta)

train_rect = patches.Rectangle(
    (alpha_min, beta_min),
    alpha_max - alpha_min,
    beta_max - beta_min,
    linewidth=1.5,
    edgecolor="black",
    facecolor="none",
)
plt.gca().add_patch(train_rect)

plt.xlabel(dataset.param_names[0])
plt.ylabel(dataset.param_names[1])
plt.title(r"Parameter-space relative error in $\alpha_D$")

plt.show()
