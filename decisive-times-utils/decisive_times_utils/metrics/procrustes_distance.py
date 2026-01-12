import h5py
import numpy as np
from pathlib import Path

GT_PATH  = Path(r"/metrics/datasets/ibl_ground_truth.h5")
SUB_PATH = Path(r"/metrics/datasets/ibl_submission.h5")

def load_spike_counts(h5_path, drop_silent=False, keep_idx=None):
    with h5py.File(h5_path, "r") as f:
        trial_key = sorted(f.keys())[0]
        M = f[trial_key]["signals"][:]      # (T, 1 + 2*N)
        counts_mat = M[:, 1::2]             # (T, N)

    if keep_idx is not None:
        counts_mat = counts_mat[:, keep_idx]
    elif drop_silent:
        keep = np.where(counts_mat.sum(axis=0) > 0)[0]
        counts_mat = counts_mat[:, keep]
        return counts_mat, keep

    return counts_mat


def pca_basis(spc_matrix, k=10):
    """
    spc_matrix: (time_bins, neurons)
    returns basis U of shape (neurons, k) with orthonormal columns
    """
    # center in time
    X = spc_matrix - spc_matrix.mean(axis=0, keepdims=True)  # (T, N)

    # SVD-based PCA: X = U_t @ diag(s) @ Vt
    U_t, s, Vt = np.linalg.svd(X, full_matrices=False)
    V = Vt.T  # (N, N) eigenvectors in neuron space

    k = min(k, V.shape[1])
    U = V[:, :k]  # first k PCs, columns are orthonormal
    return U

def procrustes_subspace_distance(U1, U2):
    # U1, U2: (N, k), columns orthonormal (PCA)
    M = U1.T @ U2            # (k, k)
    _, s, _ = np.linalg.svd(M)   # singular values s_i
    s = np.clip(s, -1.0, 1.0)    # numerical safety

    # similarity = mean cosine of principal angles
    similarity = np.mean(s)      # in [0, 1]

    # Procrustes-style distance in [0, 1]
    distance = 1.0 - similarity
    return distance, similarity

# --- load GT and SUB spike-count matrices ---
spc_GT  = load_spike_counts(GT_PATH)   # (T, N_gt)
spc_SUB = load_spike_counts(SUB_PATH)  # (T, N_sub)

print("GT shape:", spc_GT.shape)
print("SUB shape:", spc_SUB.shape)

# --- build PCA bases for each ---
k = 10  # or whatever dimensionality you want, <= min(N_gt, N_sub)
U_GT  = pca_basis(spc_GT,  k=k)  # (N_gt_useful, k)
U_SUB = pca_basis(spc_SUB, k=k)  # (N_sub_useful, k)


# --- Procrustes distance between GT and SUB PCA subspaces ---
dist, svals = procrustes_subspace_distance(U_GT, U_SUB)

print("Procrustes subspace distance:", dist)
print("Singular values (cos θ_i):", svals)
print(np.allclose(spc_GT, spc_SUB))
print(np.max(np.abs(spc_GT - spc_SUB)))
