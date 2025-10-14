import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Training sizes (per tier)
n = [22, 39, 56, 73, 90, 107, 124, 141, 158, 175]
tiers = range(1, 11)
seeds = [123+i for i in range(100)]

def emulator_stats_and_curves(em_tag, color, label):
    """
    Load all seeds for an emulator, return mean and std across seeds.
    Also plot the individual seed curves in light gray.
    """
    per_seed_means = []
    for seed in seeds:
        seed_means = []
        for i in tiers:
            df = pd.read_csv(f"alphaD_eval_{em_tag}_{seed}/alphaD_relerr_test_tier{i}.csv")
            seed_means.append(df["rel_abs"].to_numpy().mean())
        per_seed_means.append(seed_means)
        # plot this seed in light gray
        #plt.plot(n, seed_means, '-', color="lightgray", linewidth=1, alpha=0.8)

    per_seed_means = np.asarray(per_seed_means)  # shape (4, 10)
    mean = per_seed_means.mean(axis=0)
    std  = per_seed_means.std(axis=0)
    
    # plot mean + band
    plt.plot(n, mean, '-o', color=color, label=f'{label} mean')
    plt.fill_between(n, mean - std, mean + std, color=color, alpha=0.3, label=f'{label} ±1σ')
    
    return mean, std

# ---- Plot ----
plt.figure(figsize=(6,6))
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(25))

# Em1 (blue)
em1_mean, em1_std = emulator_stats_and_curves("Em1", color="blue", label="Em1")

# Em2 (orange)
em2_mean, em2_std = emulator_stats_and_curves("Em2", color="orange", label="Em2")

plt.yscale('log')
plt.ylabel(r'Abs. relative error on $\alpha_D$')
plt.xlabel('Number of training points')
plt.legend(frameon=False)
plt.tight_layout()
plt.show()

def emulator_stats_and_curves(em_tag, color, label, sets):
    """
    Load all seeds for an emulator, return mean and std across seeds.
    Also plot the individual seed curves in light gray.
    """
    per_seed_means = []
    for seed in seeds:
        seed_means = []
        for i in tiers:
            df = pd.read_csv(f"alphaD_eval_{em_tag}_{seed}/alphaD_relerr_{sets}_tier{i}.csv")
            seed_means.append(df["rel_abs"].to_numpy().mean())
        per_seed_means.append(seed_means)
        # plot this seed in light gray
        #plt.plot(n, seed_means, '-', color="lightgray", linewidth=1, alpha=0.8)

    per_seed_means = np.asarray(per_seed_means)  # shape (4, 10)
    mean = per_seed_means.mean(axis=0)
    std  = per_seed_means.std(axis=0)
    
    # plot mean + band
    plt.plot(n, mean, '-o', color=color, label=f'{label} mean')
    plt.fill_between(n, mean - std, mean + std, color=color, alpha=0.3, label=f'{label} ±1σ')
    
    return mean, std

# ---- Plot ----
plt.figure(figsize=(6,6))
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(25))

# Em1 (blue)
em1_mean, em1_std = emulator_stats_and_curves("Em2", color="blue", label="Em2 (test)", sets="test")

# Em2 (orange)
em2_mean, em2_std = emulator_stats_and_curves("Em2", color="orange", label="Em2 (train)", sets="train")

plt.yscale('log')
plt.ylabel(r'Abs. relative error on $\alpha_D$')
plt.xlabel('Number of training points')
plt.legend(frameon=False)
plt.tight_layout()
plt.show()
