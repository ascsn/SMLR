#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import tensorflow as tf
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import helper_gpt as helper_gpt


"""
Plot representative emulator-vs-true strength curves for a few selected samples.
This version matches the current helper_gpt.py API.
"""

# -----------------------------------------------------------------------------
# User settings
# -----------------------------------------------------------------------------
n = 13
retain = 0.6
params = np.loadtxt("runs_em1/best_params_global.txt").astype(np.float32)

# Hand-picked sample indices to visualize
idxs = [4, 1, 2]

# Dataset settings should match the training run
strength_dir = "data/nuclear/160Yb_2d/total_strength"
alphaD_dir = "data/nuclear/160Yb_2d/total_alphaD"
strength_regex = r"strength_(?P<p2>[0-9.]+)_(?P<p1>[0-9.]+)\.out"
alphaD_regex = None
filter_ranges = {"p1": [0.4, 1.8], "p2": [1.5, 4.0]}


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

config = helper_gpt.AnsatzConfig(
    n=n,
    n_params=int(dataset.param_values.shape[1]),
    ansatz="paper_dipole",
    width_model="affine",
    use_vector_terms=True,
)


# -----------------------------------------------------------------------------
# Small evaluation helper
# -----------------------------------------------------------------------------
def evaluate_strength_for_idx(idx):
    param_value = dataset.param_values[idx:idx + 1]
    M_batch, v_batch, eta_batch, _ = helper_gpt.build_model_matrices_and_vectors(
        params=tf.convert_to_tensor(params, dtype=tf.float32),
        config=config,
        param_values=tf.convert_to_tensor(param_value, dtype=tf.float32),
        central_point=tf.convert_to_tensor(dataset.central_point, dtype=tf.float32),
    )

    eigenvalues, eigenvectors = tf.linalg.eigh(M_batch)

    n_i = eigenvalues.shape[1]
    k_keep = int(round(retain * n_i))
    k_keep = max(1, min(k_keep, n_i))

    left = (n_i - k_keep) // 2
    right = left + k_keep

    eigvals_kept = eigenvalues[:, left:right]
    eigvecs_kept = eigenvectors[:, :, left:right]

    proj = tf.matmul(tf.transpose(eigvecs_kept, perm=[0, 2, 1]), v_batch[:, :, None])
    proj = tf.squeeze(proj, axis=-1)
    B_batch = tf.square(proj)

    omega = dataset.strengths[idx][:, 0].astype(np.float32)
    y_true = dataset.strengths[idx][:, 1].astype(np.float32)

    y_pred = helper_gpt.give_me_Lorentzian(
        energy=tf.convert_to_tensor(omega, dtype=tf.float32),
        poles=eigvals_kept[0],
        strength=B_batch[0],
        width=eta_batch[0] / 2.0,
    ).numpy()

    return omega, y_true, y_pred


# -----------------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------------
colors = ["red", "blue", "green"]

for color, idx in zip(colors, idxs):
    omega, y_true, y_pred = evaluate_strength_for_idx(idx)
    pvals = dataset.param_values[idx]

    label = ", ".join([f"{name}={value:.3f}" for name, value in zip(dataset.param_names, pvals)])

    plt.plot(omega, y_true, color=color, label=label)
    plt.plot(omega, y_pred, color=color, ls=":")

plt.xlabel(r"$\omega$ (MeV)", size=18)
plt.ylabel(r"$S$ (e$^2$ fm$^2$/MeV)", size=18)

plt.annotate(r"${}^{180}$Yb", (0.7, 0.5), xycoords="axes fraction", size=18)

plt.gca().tick_params(axis="y", direction="in", which="both", labelsize=12)
plt.gca().tick_params(axis="x", direction="in", which="both", labelsize=12)

plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(0.5))

plt.legend()
plt.xlim(5, 30)
plt.ylim(0)
plt.show()
