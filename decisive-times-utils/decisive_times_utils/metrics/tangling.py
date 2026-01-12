import numpy as np
import h5py
from pathlib import Path
import matplotlib.pyplot as plt

# ---------- Load your HDF5 ----------
GT_PATH  = Path(r"/metrics/datasets/InDomainXOR_balanced_400trials_100ms.hdf5")

with h5py.File(GT_PATH, "r") as f:
    trial_key  = sorted(f.keys())[0]
    M          = f[trial_key]["signals"][:]     # (T, 1 + 2*N)
    t          = M[:, 0]
    counts_mat = M[:, 1::2].astype(np.float32)  # spike counts (T, N)


SUB_PATH = Path(r"/metrics/datasets/InDomainXOR_SUB_balanced_400trials_100ms.hdf5")


def load_spike_counts_and_time(h5_path):
    """
    Loads spike-count matrix and time vector from the first trial of an HDF5 file.
    Assumes:
      trial_xxx / "signals" dataset of shape (T, 1 + 2*N)
      col 0        = time (seconds)
      cols 1,3,5…  = spike counts
    Returns:
      spc_matrix: (T, N) spike counts
      t:          (T,)   time vector
      dt:         float  time step
    """
    with h5py.File(h5_path, "r") as f:
        trial_key = sorted(f.keys())[0]
        M = f[trial_key]["signals"][:]   # (T, 1 + 2N)
        t = M[:, 0]
        counts_mat = M[:, 1::2]         # (T, N)

    # optional: drop completely silent units
    keep = np.where(counts_mat.sum(axis=0) > 0)[0]
    spc_matrix = counts_mat[:, keep]

    dt = float(np.median(np.diff(t)))
    return spc_matrix, t, dt


def pca_to_3d(spc_matrix):
    """
    Center spike-count matrix and project into top-3 PCs via SVD.
    spc_matrix: (T, N)
    Returns:
      proj: (T, 3) trajectory in PC space
    """
    X = spc_matrix - spc_matrix.mean(axis=0, keepdims=True)
    # SVD: X = U_t S V^T
    U_t, s_vals, Vt = np.linalg.svd(X, full_matrices=False)
    V = Vt.T  # (N, N)
    PCs = V[:, :3]        # (N, 3)
    proj = X @ PCs        # (T, 3)
    return proj


def compute_tangling_local(proj, dt, step=10, r_min=1e-3, r_max=0.2, eps=1e-6):
    X = proj[::step]
    dX = np.gradient(X, dt * step, axis=0)
    T_sub = X.shape[0]
    Q = np.zeros(T_sub)

    for i in range(T_sub):
        diff_x = X[i] - X
        dist_x = np.linalg.norm(diff_x, axis=1)

        neighbors = (dist_x > r_min) & (dist_x < r_max)
        if not np.any(neighbors):
            Q[i] = 0.0
            continue

        diff_dx = dX[i] - dX
        dist_dx = np.linalg.norm(diff_dx, axis=1)

        ratio = dist_dx[neighbors] / (dist_x[neighbors] + eps)
        Q[i] = np.max(ratio)

    return Q, X



# --- approximate tangling ---
# dt already computed earlier from your time vector t
# proj has time step dt between rows

def plot_tangling_3d(h5_path, title, step=10):
    """
    Full pipeline for one dataset:
      - load spike counts
      - PCA to 3D
      - compute tangling
      - 3D scatter colored by Q(t)
    """
    spc_matrix, t, dt = load_spike_counts_and_time(h5_path)
    proj = pca_to_3d(spc_matrix)
    Q, X_sub = compute_tangling_local(proj, dt, step=step)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(
        X_sub[:, 0],
        X_sub[:, 1],
        X_sub[:, 2],
        c=Q,
        cmap='plasma',
        s=3
    )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(title)

    cb = fig.colorbar(sc, ax=ax, shrink=0.7)
    cb.set_label("Tangling Q(t)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # GT tangling visualization (3D)
    plot_tangling_3d(GT_PATH, "GT: Neural manifold with tangling", step=10)

    # SUB tangling visualization (3D) – separate figure
    plot_tangling_3d(SUB_PATH, "SUB: Neural manifold with tangling", step=10)


