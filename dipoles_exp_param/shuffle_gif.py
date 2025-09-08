#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 11:49:34 2025

@author: anteravlic
"""

import os, re, random
import numpy as np
import matplotlib.pyplot as plt
import helper
import tensorflow as tf
import csv
import matplotlib.animation as animation

# -----------------------------
# 0) Config
# -----------------------------
strength_dir = '../dipoles_data_all/total_strength/'
pattern = re.compile(r'strength_([0-9.]+)_([0-9.]+)\.out')
SEED = 123
TEST_PICK = 50
N_TIERS = 10

# -----------------------------
# 1) Collect (alpha, beta) as STRINGS, but we’ll also keep float versions for geometry
# -----------------------------
combined_str = []  # list of (alpha_str, beta_str)
for fname in os.listdir(strength_dir):
    m = pattern.match(fname)
    if m:
        beta_s  = m.group(1)   # string
        alpha_s = m.group(2)   # string
        combined_str.append((alpha_s, beta_s))

if not combined_str:
    raise RuntimeError("No points found. Check 'strength_dir' and filename pattern.")

# Float views for geometry
combined_f = [(float(a), float(b)) for (a, b) in combined_str]

# Helper maps for easy conversion back to string tuples
def f2s_tuple(pt_f):
    a, b = pt_f
    # find exact match in original string list (avoid float->str drift):
    # since filenames provided exact strings, we reconstruct by formatting through the originals
    # Build a dict once for speed:
    return (format(a, ".12g"), format(b, ".12g"))

# Build a lookup from float to original string by nearest exact string in combined_str
# (we’ll use exact positions instead to avoid ambiguity)
float_to_str = { (float(a), float(b)):(a, b) for (a,b) in combined_str }

def as_str_tuple(pt_f):
    # Use exact lookup if possible; otherwise fall back to formatted strings
    return float_to_str.get((pt_f[0], pt_f[1]), (format(pt_f[0], ".12g"), format(pt_f[1], ".12g")))

# -----------------------------
# 2) Find center (nearest to box center) and “corner” samples
# -----------------------------
alpha_vals = [a for a, b in combined_f]
beta_vals  = [b for a, b in combined_f]
Amin, Amax = min(alpha_vals), max(alpha_vals)
Bmin, Bmax = min(beta_vals),  max(beta_vals)

cx, cy = (Amin + Amax) / 2.0, (Bmin + Bmax) / 2.0

def nearest_to(target, points):
    x0, y0 = target
    return min(points, key=lambda t: (t[0]-x0)**2 + (t[1]-y0)**2)

central_f = nearest_to((cx, cy), combined_f)
corners_target = [(Amin, Bmin), (Amin, Bmax), (Amax, Bmin), (Amax, Bmax)]
corner_fs = [nearest_to(c, combined_f) for c in corners_target]

# Deduplicate (center could coincide with a corner)
fixed_f = []
seen = set()
for p in [central_f] + corner_fs:
    if p not in seen:
        fixed_f.append(p)
        seen.add(p)

# -----------------------------
# 3) Build test set and nested training tiers
# -----------------------------
random.seed(SEED)

all_set_f = set(combined_f)
fixed_set_f = set(fixed_f)

# Candidates for test: exclude fixed so fixed remain in training
candidates_f = list(all_set_f - fixed_set_f)

# Pick test points
TEST_SIZE = min(TEST_PICK, len(candidates_f))
test_set_f = set(random.sample(candidates_f, TEST_SIZE))

# Remaining pool for training (besides fixed)
train_pool_f = list((all_set_f - test_set_f) - fixed_set_f)
random.shuffle(train_pool_f)

# Create 5 nested tiers:
# sizes grow roughly linearly from ~20% of pool up to 100% of pool
if len(train_pool_f) == 0:
    sizes = [0] * N_TIERS
else:
    base = int(np.ceil(len(train_pool_f) / N_TIERS))  # first tier ~ 1/N
    sizes = np.linspace(base, len(train_pool_f), N_TIERS, dtype=int)
    sizes = np.maximum.accumulate(sizes)  # strictly non-decreasing

train_tiers_f = []
for s in sizes:
    add = train_pool_f[:s]
    tier_pts = list(fixed_f) + add
    train_tiers_f.append(tier_pts)

# Final sets (float)
test_points_f = list(test_set_f)

# -----------------------------
# 4) Plot
# -----------------------------
plt.figure(figsize=(7.6, 6.4))

# All points (background)
all_a = [a for a, b in combined_f]
all_b = [b for a, b in combined_f]
plt.scatter(all_a, all_b, s=12, color="#d0d0d0", label="All points (background)", zorder=1)

# Test set
ta = [a for a, b in test_points_f]
tb = [b for a, b in test_points_f]
plt.scatter(ta, tb, s=28, marker="x", label=f"Test ({len(test_points_f)})", zorder=3)

# Training tiers (nested). To avoid clutter, show all with low alpha; last tier darker.
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
for idx, tier in enumerate(train_tiers_f):
    a = [p[0] for p in tier]
    b = [p[1] for p in tier]
    alpha_val = 0.25 if idx < len(train_tiers_f) - 1 else 0.85
    plt.scatter(a, b, s=16, alpha=alpha_val, color=colors[idx % len(colors)],
                label=f"Train tier {idx+1} ({len(tier)})", zorder=2)

# Fixed points (center + corners)
fa = [p[0] for p in fixed_f]
fb = [p[1] for p in fixed_f]
plt.scatter(fa, fb, s=72, facecolors='none', edgecolors='k', linewidths=1.6,
            label=f"Fixed (center+corners) ({len(fixed_f)})", zorder=4)

plt.xlabel("alpha")
plt.ylabel("beta")
plt.title("Train/Test split with nested training tiers")
plt.legend(loc="best", fontsize=8)
plt.tight_layout()
plt.show()

# -----------------------------
# 5) Convert outputs back to STRING tuples in original format
# -----------------------------
def to_str_list(points_f_list):
    return [(float_to_str.get((a, b), (format(a, ".12g"), format(b, ".12g"))))
            for (a, b) in points_f_list]

fixed_points_str = to_str_list(fixed_f)
test_points_str  = to_str_list(test_points_f)
train_tiers_str  = [to_str_list(tier) for tier in train_tiers_f]

central_point_str = float_to_str.get(central_f, (format(central_f[0], ".12g"), format(central_f[1], ".12g")))
corner_points_str = [float_to_str.get(p, (format(p[0], ".12g"), format(p[1], ".12g"))) for p in corner_fs]

split_info = {
    "fixed_points": fixed_points_str,     # list[tuple[str,str]]
    "test_points":  test_points_str,      # list[tuple[str,str]]
    "train_tiers":  train_tiers_str,      # list[list[tuple[str,str]]], nested
    "center":       central_point_str,    # tuple[str,str]
    "corners":      corner_points_str,    # list[tuple[str,str]]
    "seed":         SEED,
    "sizes":        [len(t) for t in train_tiers_f],
}

print("Total points:", len(combined_str))
print("Fixed points (center+corners):", len(fixed_points_str))
print("Test points:", len(test_points_str))
print("Training tier sizes:", split_info["sizes"])



# --- reuse your split_info ---
train_tiers_str = split_info["train_tiers"]
test_points_str = split_info["test_points"]
fixed_points_str = split_info["fixed_points"]

# convert to floats for plotting
def to_float(points):
    return [(float(a), float(b)) for (a,b) in points]

train_tiers_f = [to_float(tier) for tier in train_tiers_str]
test_points_f = to_float(test_points_str)
fixed_points_f = to_float(fixed_points_str)

all_points_f = [to_float(train_tiers_str[-1])]  # final tier covers all training pool
all_points_f = all_points_f[0] + test_points_f  # union for plotting limits

# --- build animation frames ---
fig, ax = plt.subplots(figsize=(6,5))

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def init():
    ax.clear()
    ax.set_xlabel("alpha")
    ax.set_ylabel("beta")
    ax.set_title("Train/Test split construction")
    a_all = [a for a,b in all_points_f]
    b_all = [b for a,b in all_points_f]
    ax.set_xlim(min(a_all)-0.5, max(a_all)+0.5)
    ax.set_ylim(min(b_all)-0.5, max(b_all)+0.5)

def update(frame):
    ax.clear()
    init()

    # plot background all points
    a_all = [a for a,b in all_points_f]
    b_all = [b for a,b in all_points_f]
    ax.scatter(a_all, b_all, s=10, color="#d0d0d0", label="All points")

    # fixed points
    fa, fb = zip(*fixed_points_f)
    ax.scatter(fa, fb, s=80, facecolors='none', edgecolors='k', linewidths=1.6,
               label="Fixed (center+corners)")

    # test points
    ta, tb = zip(*test_points_f)
    ax.scatter(ta, tb, s=30, marker="x", color="red", label="Test set")

    # training tier up to 'frame'
    for tier_idx in range(frame+1):
        tier_f = train_tiers_f[tier_idx]
        a, b = zip(*tier_f)
        ax.scatter(a, b, s=20, color=colors[tier_idx % len(colors)],
                   alpha=0.6, label=f"Train tier {tier_idx+1}")

    ax.legend(fontsize=7, loc="best")
    ax.set_title(f"Construction up to Tier {frame+1}")

# --- make animation ---
ani = animation.FuncAnimation(fig, update, frames=len(train_tiers_f),
                              init_func=init, blit=False, repeat=True)

ani.save("train_test_split.gif", writer="pillow", fps=1)
plt.close(fig)
print("Saved GIF -> train_test_split.gif")
