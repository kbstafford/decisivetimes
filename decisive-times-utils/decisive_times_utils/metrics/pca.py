from pathlib import Path
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter1d


def load_counts_from_h5(h5_path: Path, trial_index: int = 0, load_all_trials: bool = False):
    with h5py.File(h5_path, "r") as f:
        trials = sorted(f.keys())
        if not trials:
            raise RuntimeError("No trials in HDF5.")

        if load_all_trials:
            print(f"Loading all {len(trials)} trials...")
            all_t = []
            all_S = []

            for i, key in enumerate(trials):
                M = f[key]["data"][:]
                t = M[:, 0].astype(float)
                S = M[:, 1:].astype(float)

                # Offset time for each trial so they don't overlap
                if i > 0:
                    t = t + all_t[-1][-1] + (t[1] - t[0])

                all_t.append(t)
                all_S.append(S)

            t = np.concatenate(all_t)
            S = np.concatenate(all_S, axis=0)
            print(f"Combined: {len(trials)} trials, {len(t)} time points, {S.shape[1]} channels")
        else:
            key = trials[trial_index]
            print(f"Loading trial: {key}")

            M = f[key]["data"][:]
            print(f"Data shape: {M.shape}")

            t = M[:, 0].astype(float)
            S = M[:, 1:].astype(float)

            print(f"Time points: {len(t)}, Neurons/channels: {S.shape[1]}")

    return t, S

def rebin_time_series(X: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1: return X
    T = (X.shape[0] // factor) * factor
    return X[:T].reshape(T // factor, factor, *X.shape[1:]).mean(axis=1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default=str(Path(__file__).resolve().parent.parent / "datasets" / "InDomainXOR_balanced_400trials_100ms.hdf5"))
    ap.add_argument("--trial", type=int, default=0)
    ap.add_argument("--target-bin", type=float, default=0.050, help="seconds; 0 keeps original")
    ap.add_argument("--smooth-sigma", type=float, default=5.0, help="gaussian sigma (bins) on rates")
    ap.add_argument("--k", type=int, default=10, help="# PCA components for visualization")
    ap.add_argument("--downsample", type=int, default=1, help="plot every Nth timepoint")
    args = ap.parse_args()

    # --- Load counts ---
    t, S = load_counts_from_h5(Path(args.gt), trial_index=args.trial)
    dt = float(np.median(np.diff(t)))
    if args.target_bin and args.target_bin > 0:
        factor = int(round(args.target_bin / dt))
        if factor < 1: factor = 1
        if factor > 1:
            S = rebin_time_series(S, factor)
            t = t[:S.shape[0]]
            dt *= factor

    # counts -> rates, smooth, z-score
    R = S / dt
    if args.smooth_sigma and args.smooth_sigma > 0:
        R = gaussian_filter1d(R, sigma=args.smooth_sigma, axis=0)
    X = (R - R.mean(0, keepdims=True)) / (R.std(0, keepdims=True) + 1e-8)

    # PCA (keep k for visuals, but compute full eigs for PR if you want)
    k = min(args.k, X.shape[1])
    pca = PCA(n_components=k, svd_solver="full")
    Z = pca.fit_transform(X)                       # (T, k)
    expl = pca.explained_variance_ratio_

    # Optional downsampling for plotting
    ds = max(1, args.downsample)
    t_plot = t[::ds]
    Zp = Z[::ds]

    # 1) Scree + cumulative
    plt.figure(figsize=(7,4))
    x = np.arange(1, k+1)
    plt.bar(x, expl, width=0.8)
    cum = np.cumsum(expl)
    plt.plot(x, cum, marker='o')
    plt.xlabel("PC")
    plt.ylabel("Variance explained")
    plt.title("Scree and cumulative variance")
    plt.tight_layout()

    # 2) 2D trajectory (PC1 vs PC2), colored by time
    plt.figure(figsize=(6,6))
    sc = plt.scatter(Zp[:,0], Zp[:,1], c=t_plot - t_plot[0], s=6)
    cb = plt.colorbar(sc); cb.set_label("time (s)")
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("PCA trajectory (2D)")
    plt.tight_layout()

    # 3) 3D trajectory (PC1–PC3)
    if Z.shape[1] >= 3:
        from mpl_toolkits.mplot3d import Axes3D  # noqa
        fig = plt.figure(figsize=(7,6))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(Zp[:,0], Zp[:,1], Zp[:,2], lw=1)
        p = Zp[::max(1,len(Zp)//100)]
        ax.scatter(p[:,0], p[:,1], p[:,2], s=8)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
        ax.set_title("PCA trajectory (3D)")
        fig.tight_layout()

    # 4) Spike-rate heatmap (time × neurons)
    plt.figure(figsize=(8,5))
    # z-limited for visibility (optional)
    vmax = np.percentile(R, 99)
    plt.imshow(R.T, aspect='auto', origin='lower',
               extent=[t[0], t[-1], 0, R.shape[1]], vmin=0, vmax=vmax)
    plt.xlabel("time (s)"); plt.ylabel("neuron idx"); plt.title("Spike rates (Hz)")
    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()
