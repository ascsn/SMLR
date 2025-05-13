# QRPA Emulator

This repository contains the code accompanying the paper *L.Jin et al., "Emulating the Quasiparticle Random Phase Approximation"*.

The repository is organized into three main types of folders:

---

## `Beta_decay`

Contains scripts to run the emulator for the **charge-exchange QRPA**. These scripts read input from the `beta_decay_data` folder containing strength functions and beta-decay half-lives.

**Main files:**

1. `main.py` — Main script to train the PMM emulator  
2. `helper.py` — Helper subroutines used in training and evaluation  
3. `check_results.py` — Plots strength functions and beta-decay half-lives in the test set (Fig. 5)  
4. `plot_strength.py` — Produces the "rainbow plot" (Fig. 4)  
5. `CAT_plot.py` — Plots the CAT (Fig. 6)

**Note:** Files with the `_only_Hl.py` suffix correspond to Algorithm 2 described in the paper.

---

## `dipoles`

Contains scripts to run the emulator for the **like-particle QRPA**. These scripts read input from the `dipole_data` folder containing strength functions and relevant observables.

**Main files:**

1. `main.py` — Main script to train the PMM emulator  
2. `helper.py` — Helper subroutines used in training and evaluation  
3. `check_results.py` — Plots strength functions and dipole polarizability \( \alpha_D \) in the test set (Fig. 2)  
4. `plot_strength.py` — Produces the "rainbow plot" and phase space visualization (Fig. 1)  
5. `CAT_plot.py` — Plots the CAT (Fig. 6)

**Note:** Files with the `_only_alphaD.py` suffix correspond to Algorithm 2 described in the paper.

## DATA files

Folders which contain the high-fidelity QRPA calculations are stored in `dipoles_data` and `beta_decay_data`. Please ask Ante for details.
