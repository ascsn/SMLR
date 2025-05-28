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


## How to use code for Emulator 2

Please switch to `dev_branch` and locate `dipoles` folder, the training script used for Emulator 2 is called `main_only_alphaD.py`. In the file up to line `118` is just reading the data in and then you have to select dimension of PMM which is number `n`. That's the only input parameter we have, other than selecting the initialization values in the line `random_initial_guess = np.random.uniform(0, 2, nec_num_param)`.  The line `optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.1)` sets the learning rate for the optimization, please keep in mind that 0.1 is already a large number. Number of iterations are defined in line 169 `num_iterations=40000`. The code selects those parameters which minimize the cross-validation cost. Finally, the parameters are stored in a text file `params_n.txt` (`n` is replaced with a number of params) and the training set is stored in `train_set.txt`. These two are used by the plotting scripts. The main plotting script is `plot_alphaD.py`, it produces a figure showing how the $\alpha_D$ changes as a function of parameters. We still have to work a bit on those plots ... 

All functions are defined in file `helper.py`, and please note that in the current version, Emulator 2 uses the second eigenvalue, line 379, `eigenvalue[1]`, which seems to work for some reason. So we have to investigate that more.


