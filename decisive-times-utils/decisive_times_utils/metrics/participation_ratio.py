import h5py
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

GT_PATH  = Path(r"/metrics/datasets/ibl_ground_truth.h5")
SUB_PATH = Path(r"/metrics/datasets/ibl_submission.h5")

# choose which HDF5 to read
with h5py.File(GT_PATH, "r") as f:
    # take the first trial; change to loop/concat if you want all trials
    trial_key = sorted(f.keys())[0]
    M = f[trial_key]["signals"][:]             # shape: (T, 1 + 2*N)
    t = M[:, 0]                                # bin centers (seconds)
    counts_mat = M[:, 1::2]                    # spike counts columns (T, N)
    N = counts_mat.shape[1]

# bin edges that match these centers
dt = float(np.median(np.diff(t)))
edges = np.arange(t[0] - 0.5*dt, t[-1] + 0.5*dt + 1e-12, dt)

# reconstruct spike times per unit from counts (uses bin centers)
# NOTE: this can be large if counts are big; OK for short windows.
spike_trains = [np.repeat(t, counts_mat[:, i].astype(int)) for i in range(N)]
unit_names   = [f"neuron{i+1}" for i in range(N)]

# build spike-count matrix [time_bins * neurons]
spc_matrix = np.zeros((edges.size - 1, len(spike_trains)), dtype=float)
for i, st in enumerate(spike_trains):
    counts, _ = np.histogram(st, bins=edges)
    spc_matrix[:, i] = counts

# covariance calculation
spc_matrix_c = spc_matrix - spc_matrix.mean(axis=0, keepdims=True)
cov_matrix = (spc_matrix_c.T @ spc_matrix) / (spc_matrix.shape[0] - 1)
cov_matrix = 0.5 * (cov_matrix + cov_matrix.T) # symmetrize
tr, tr2 = np.trace(cov_matrix), np.trace(cov_matrix @ cov_matrix)

# calculate participation ratio
pr = (tr * tr) / tr2 if tr2 > 0 else np.nan

# display participation ratio
print("Participation ratio:", pr)


# PCA via SVD on centered spike-count matrix
U_t, s_vals, Vt = np.linalg.svd(spc_matrix_c, full_matrices=False)
V = Vt.T  # (neurons, neurons) – eigenvectors in neuron space

# take top 3 PCs
k = 3
PCs = V[:, :k]                 # (neurons, 3)
proj = spc_matrix_c @ PCs      # (time_bins, 3)

# Optionally subsample time for clarity
step = 5   # plot every 5th time bin
proj_sub = proj[::step]
t_sub = t[::step]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 3D trajectory over time
ax.plot(proj_sub[:, 0],
        proj_sub[:, 1],
        proj_sub[:, 2],
        lw=1.0)

p = ax.scatter(proj_sub[:, 0], proj_sub[:, 1], proj_sub[:, 2],
                c=t_sub, cmap='viridis', s=4)
fig.colorbar(p, ax=ax, label='Time (s)')

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.set_title("Neural manifold (top 3 PCs) – GT")
plt.tight_layout()
plt.show()