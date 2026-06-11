#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LogNorm
import tensorflow as tf

import archive.Dipole_polarizability.src.helper_gpt as helper_gpt


"""
Evaluate Emulator 2 (alphaD-only) on the current helper_gpt/main_only_alphaD_gpt workflow.

This script:
1. Loads the same alphaD dataset structure used in training.
2. Reconstructs predicted alphaD values from best_params_global.txt.
3. Plots predicted-vs-true alphaD.
4. Plots a parameter-space map of relative alphaD error.
"""


# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
n = 10
params = np.loadtxt("runs_em2/best_params_global.txt").astype(np.float32)

strength_dir = "../dipoles_data_all/total_strength/"
alphaD_dir = "../dipoles_data_all/total_alphaD/"
strength_regex = r"strength_(?P<p2>[0-9.]+)_(?P<p1>[0-9.]+)\.out"
alphaD_regex = None
filter_ranges = None

config = helper_gpt.AlphaDOnlyConfig(
    n=n,
    n_params=2,
    ansatz="linear",
    alphaD_mode="poles_and_strengths",
)


# -----------------------------------------------------------------------------
# Load dataset
# -----------------------------------------------------------------------------
dataset = helper_gpt.load_generic_alphaD_dataset(
    strength_dir=strength_dir,
    alphaD_dir=alphaD_dir,
    strength_regex=strength_regex,
    alphaD_regex=alphaD_regex,
    filter_ranges=filter_ranges,
    central_point=None,
)

print("Loaded dataset")
print("  param names:", dataset.param_names)
print("  samples:", len(dataset.alphaD_values))
print("  central point:", dataset.central_point.tolist())



# -----------------------------------------------------------------------------
# Build alphaD predictions
# -----------------------------------------------------------------------------
pred_fn = helper_gpt.make_alphaD_only_loss_fn_generic(config)

alphaD_opt = pred_fn(
    tf.convert_to_tensor(params, dtype=tf.float32),
    tf.convert_to_tensor(dataset.param_values, dtype=tf.float32),
    tf.convert_to_tensor(dataset.central_point, dtype=tf.float32),
).numpy()

alphaD_true = np.asarray(dataset.alphaD_values)
rel_err = np.abs(alphaD_opt - alphaD_true) / np.maximum(np.abs(alphaD_true), 1e-8)

print("Mean relative alphaD error:", np.mean(rel_err))
print("Max relative alphaD error:", np.max(rel_err))


# -----------------------------------------------------------------------------
# alphaD predicted vs true
# -----------------------------------------------------------------------------
plt.figure(1, figsize=(6, 4))
plt.scatter(alphaD_true, alphaD_opt)

for i in range(len(alphaD_opt)):
    plt.text(alphaD_true[i], alphaD_opt[i], str(i), fontsize=9, ha="right", va="bottom")

xline = np.linspace(
    min(np.min(alphaD_true), np.min(alphaD_opt)),
    max(np.max(alphaD_true), np.max(alphaD_opt)),
    100,
)
plt.plot(xline, xline, color="black")
plt.xlabel(r"True $\alpha_D$")
plt.ylabel(r"Predicted $\alpha_D$")
plt.title(f"Emulator 2, n = {n}")


# -----------------------------------------------------------------------------
# Parameter-space relative error map
# -----------------------------------------------------------------------------
plt.figure(2)

x = dataset.param_values[:, 0]
y = dataset.param_values[:, 1]

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