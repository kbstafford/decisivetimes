import numpy as np
import h5py
from pathlib import Path

GT_PATH  = Path(r"/metrics/datasets/ibl_ground_truth.h5")
SUB_PATH = Path(r"/metrics/datasets/ibl_submission.h5")


def load_spike_counts(h5_path):
    """
    Load spike-count matrix (time_bins x neurons) from the first trial
    in the given HDF5 file. Assumes dataset 'signals' with pattern:
      col 0 = time
      cols 1,3,5,... = spike counts
    """
    with h5py.File(h5_path, "r") as f:
        trial_key = sorted(f.keys())[0]          # e.g. 'trial_0000'
        M = f[trial_key]["signals"][:]           # (T, 1 + 2*N)
        counts_mat = M[:, 1::2]                  # spike counts: (T, N)

    return counts_mat


def pca_basis(spc_matrix, n_components=10):
    """
    spc_matrix: (time_bins, neurons)
    returns U: (neurons, k) with orthonormal columns (top k PCs)
    """
    X = spc_matrix - spc_matrix.mean(axis=0, keepdims=True)  # center
    U_t, s, Vt = np.linalg.svd(X, full_matrices=False)       # X = U_t S Vt
    V = Vt.T                                                 # (neurons, neurons)

    k = min(n_components, V.shape[1])
    U = V[:, :k]
    return U


def pca_subspace_angles(U_A, U_B):
    """
    Compute principal angles between two subspaces spanned by columns
    of U_A and U_B (assumed orthonormal).
    Returns:
        theta: array of angles in radians (theta[0] = smallest)
        s:     singular values = cos(theta_i)
    """
    k = min(U_A.shape[1], U_B.shape[1])
    U1 = U_A[:, :k]
    U2 = U_B[:, :k]

    # overlap matrix
    M = U1.T @ U2               # (k, k)
    _, s, _ = np.linalg.svd(M)  # singular values
    s = np.clip(s, -1.0, 1.0)   # numerical safety

    theta = np.arccos(s)        # radians
    return theta, s


if __name__ == "__main__":
    # load GT and SUB spike-count matrices
    spc_GT  = load_spike_counts(GT_PATH)
    spc_SUB = load_spike_counts(SUB_PATH)

    print("GT shape:", spc_GT.shape)
    print("SUB shape:", spc_SUB.shape)

    # PCA bases
    k = 10  # subspace dimensionality to compare
    U_GT  = pca_basis(spc_GT,  n_components=k)
    U_SUB = pca_basis(spc_SUB, n_components=k)

    # subspace angles
    theta, s = pca_subspace_angles(U_GT, U_SUB)

    # summary
    theta_deg = np.degrees(theta)
    print("Singular values (cos θ_i):", s)
    print("Principal angles (deg):   ", theta_deg)
    print("First principal angle (deg):", theta_deg[0])


import matplotlib.pyplot as plt
import numpy as np

# assuming you already have:
# theta, s = pca_subspace_angles(U_GT, U_SUB)

idx = np.arange(1, len(theta) + 1)  # 1..k
theta_deg = np.degrees(theta)

plt.figure()
plt.plot(idx, theta_deg, marker='o')
plt.xlabel("Component index")
plt.ylabel("Principal angle (degrees)")
plt.title("GT vs SUB PCA subspace angles")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(idx, s, marker='o')
plt.xlabel("Component index")
plt.ylabel("cos(θ)")
plt.title("GT vs SUB subspace similarity (cosine)")
plt.grid(True)
plt.ylim(0, 1.05)
plt.tight_layout()
plt.show()

# take first 2 PCs from each basis
U1 = U_GT[:, :2]   # (neurons, 2)
U2 = U_SUB[:, :2]  # (neurons, 2)

plt.figure()
plt.scatter(U1[:, 0], U1[:, 1], alpha=0.5, label="GT PCs (loadings)")
plt.scatter(U2[:, 0], U2[:, 1], alpha=0.5, label="SUB PCs (loadings)")
plt.xlabel("PC1 loading")
plt.ylabel("PC2 loading")
plt.title("Neuron loadings on first two PCs\nGT vs SUB")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


from mpl_toolkits.mplot3d import Axes3D  # just importing enables 3D

# --- project GT and SUB data into their own top-3 PC spaces ---

# center in time
X_GT  = spc_GT  - spc_GT.mean(axis=0, keepdims=True)   # (T, N)
X_SUB = spc_SUB - spc_SUB.mean(axis=0, keepdims=True)  # (T, N)

# project onto first 3 PCs in neuron space
GT_proj  = X_GT  @ U_GT[:, :3]   # (T, 3)
SUB_proj = X_SUB @ U_SUB[:, :3]  # (T, 3)

# --- 3D plot of trajectories / point clouds ---

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# you can use a subset of time points if it's too dense
idx = slice(0, GT_proj.shape[0], 5)  # every 5th point, for clarity

ax.plot(GT_proj[idx, 0], GT_proj[idx, 1], GT_proj[idx, 2],
        alpha=0.7, label='GT', linewidth=1)
ax.plot(SUB_proj[idx, 0], SUB_proj[idx, 1], SUB_proj[idx, 2],
        alpha=0.7, label='SUB', linewidth=1)

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('GT vs SUB neural manifold (top 3 PCs)')
ax.legend()
plt.tight_layout()
plt.show()
