import numpy as np
from scipy.linalg import eigh
import os


# ------------------------------------------------------------
#  NECESSARY MATRICES: D (diagonal), S1, S2 (symmetric)
# ------------------------------------------------------------

def matrices(n, rng=42):
    """
    Generate:
        D  : diagonal matrix with entries uniform[1,10]
        S1 : symmetric matrix with entries uniform[1,10]
        S2 : symmetric matrix with entries uniform[1,10]
    """
    if rng is None:
        rng = np.random.default_rng()

    # Diagonal part
    D_vals = rng.uniform(1, 10, size=n)
    D = np.diag(D_vals)

    # Symmetric matrices
    A1 = rng.uniform(1, 10, size=(n, n))
    A2 = rng.uniform(1, 10, size=(n, n))

    S1 = (A1 + A1.T) / 2
    S2 = (A2 + A2.T) / 2

    return D, S1, S2


# ------------------------------------------------------------
#  FILE OUTPUT HELPERS
# ------------------------------------------------------------

def save_matrix_txt(filename, alpha, beta, M0, M1, M2, H):
    """
    Save M0, M1, M2, and H(alpha,beta) into a single text file.
    Matrices are printed explicitly.
    """
    with open(filename, "w") as f:
        f.write(f"alpha {alpha}\n")
        f.write(f"beta  {beta}\n\n")

        f.write("M0 (diagonal):\n")
        f.write(np.array2string(M0, precision=6, separator=' ') + "\n\n")

        f.write(f"alpha * M1:\n")
        f.write(np.array2string(alpha * M1, precision=6, separator=' ') + "\n\n")

        f.write(f"beta * M2:\n")
        f.write(np.array2string(beta * M2, precision=6, separator=' ') + "\n\n")

        f.write("H(alpha, beta) = M0 + alpha*M1 + beta*M2:\n")
        f.write(np.array2string(H, precision=6, separator=' ') + "\n")


def save_response_txt(filename, alpha, beta, eigvals, projections):
    """
    Save eigenvalues and b^T v_k projections in a two-column vertical layout:
       Eigenvalues:        Projections (b^T v_k):
       λ1                  p1
       λ2                  p2
       ...
       λn                  pn
    """
    with open(filename, "w") as f:
        f.write(f"alpha {alpha}\n")
        f.write(f"beta  {beta}\n\n")

        f.write(f"{'Eigenvalues:':<20} {'Projections (b^T v_k):':<20}\n")

        for val, proj in zip(eigvals, projections):
            f.write(f"{val:<20.12f} {proj:<20.12f}\n")


# ------------------------------------------------------------
#  MAIN DATASET GENERATOR
# ------------------------------------------------------------

def compute_and_save_all(M0, M1, M2, grid, b, outdir="toy_data"):
    """
    For each (alpha, beta) compute:
        H = M0 + alpha*M1 + beta*M2
        eigenvalues, eigenvectors
        projections = b^T v_k
    Saves:
        response_alpha_beta.txt
        matrices_alpha_beta.txt
    """
    n = M0.shape[0]
    b = np.asarray(b).reshape(n)

    # Create directories
    resp_dir = os.path.join(outdir, "total_response")
    mats_dir = os.path.join(outdir, "total_matrices")

    os.makedirs(resp_dir, exist_ok=True)
    os.makedirs(mats_dir, exist_ok=True)

    for alpha, beta in grid:
        # Construct H (already symmetric)
        H = M0 + alpha * M1 + beta * M2

        # Eigen-decomposition
        eigvals, eigvecs = eigh(H)
        projections = b @ eigvecs

        # filenames
        a_str = f"{alpha:.4f}"#.replace('.', 'p')
        b_str = f"{beta:.4f}"#.replace('.', 'p')

        resp_file = os.path.join(resp_dir, f"response_{a_str}_{b_str}.out")
        mats_file = os.path.join(mats_dir, f"matrices_{a_str}_{b_str}.out")

        save_response_txt(resp_file, alpha, beta, eigvals, projections)
        save_matrix_txt(mats_file, alpha, beta, M0, M1, M2, H)


# ------------------------------------------------------------
#  EXAMPLE USAGE
# ------------------------------------------------------------

if __name__ == "__main__":
    n = 10
    rng = np.random.default_rng(0)

    # Generate matrices
    M0, M1, M2 = matrices(n, rng=rng)

    # Create α, β grid: 10 points from 1 to 5
    alpha_vals = np.linspace(1, 5, 10)
    beta_vals  = np.linspace(1, 5, 10)
    grid = [(alpha, beta) for alpha in alpha_vals for beta in beta_vals]

    # User-specified b:
    #b = np.ones(n)  # or arbitrary vector later
    #b = rng.uniform(1, 10, n)
    b = rng.normal(0, 1, n)
    b /= np.linalg.norm(b)

    compute_and_save_all(M0, M1, M2, grid, b, outdir="toy_data")

    print("Generation complete.")
