# =============================================================================
# COMPLETE ANALYSIS: TIMESCALES ONLY (Neural) for 5 IBL subjects
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
from one.api import ONE

one = ONE(base_url="https://openalyx.internationalbrainlab.org")

print("=" * 70)
print("TIMESCALES ONLY: Neuron + Latent (PCA) timescale hierarchy")
print("=" * 70)

# =============================================================================
# CONFIG
# =============================================================================

SUBJECTS = ["FD_36", "FD_28", "FD_14", "FD_24", "FD_21"]

# Probe collections to try (many IBL sessions store spikes under these)
PROBE_COLLECTION_CANDIDATES = [
    "alf/probe00/pykilosort",
    "alf/probe00",
    "alf/probe01/pykilosort",
    "alf/probe01",
]

# Minimum data requirements
MIN_NEURONS = 50           # relaxable
MIN_SPIKES_TOTAL = 50_000  # relaxable

# Timescale computation settings
BIN_SIZE = 0.02            # seconds (20 ms bins); change to 0.01 if you want 10 ms
WIN_SEC = 10.0             # use first N seconds of session for speed; set None to use full session
MAX_LAG_BINS = 50          # fit autocorr from lag=1..MAX_LAG_BINS (=> 1..1s if 20ms bins)
N_PCS = 20                 # number of PCs to analyze for hierarchy

# =============================================================================
# Helper functions
# =============================================================================

def exp_decay(t, A, tau):
    return A * np.exp(-t / tau)

def autocorr_fft(x):
    """Autocorrelation R(lag) normalized so R(0)=1, using FFT."""
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    n = len(x)
    if n < 10:
        return None
    f = np.fft.rfft(x, n=2*n)
    ac = np.fft.irfft(f * np.conjugate(f))[:n]
    ac = ac / (ac[0] + 1e-12)
    return ac  # lags 0..n-1

def fit_tau_from_ac(ac, dt, max_lag_bins=50):
    """
    Fit ac[lag] ~ A exp(-lag*dt/tau) using positive ac values only.
    Returns tau in seconds.
    """
    if ac is None:
        return np.nan

    max_lag_bins = min(max_lag_bins, len(ac) - 1)
    if max_lag_bins < 5:
        return np.nan

    lags = np.arange(1, max_lag_bins + 1)
    y = ac[lags]

    # Use only positive region (common convention for neural timescales)
    mask = np.isfinite(y) & (y > 0)
    lags = lags[mask]
    y = y[mask]
    if len(y) < 5:
        return np.nan

    t = lags * dt

    # initial guesses
    A0 = float(y[0])
    tau0 = float(t[len(t)//2]) if len(t) else dt * 10

    try:
        popt, _ = curve_fit(exp_decay, t, y, p0=[A0, tau0], maxfev=5000)
        A_hat, tau_hat = popt
        if not np.isfinite(tau_hat) or tau_hat <= 0:
            return np.nan
        return float(tau_hat)
    except Exception:
        return np.nan

def bin_spike_train(spike_times, t0, t1, bin_size):
    """Return binned spike counts for spike_times within [t0, t1)."""
    n_bins = int(np.ceil((t1 - t0) / bin_size))
    if n_bins <= 5:
        return None
    # histogram expects bin edges
    edges = t0 + np.arange(n_bins + 1) * bin_size
    counts, _ = np.histogram(spike_times, bins=edges)
    return counts.astype(float)

def find_spike_collections(eid):
    """
    Return a list of collections that contain BOTH spikes.times and spikes.clusters.
    """
    try:
        datasets = one.list_datasets(eid)
    except Exception:
        return []

    # Find all spikes.times paths
    times_paths = [d for d in datasets if d.endswith("spikes.times.npy")]
    collections = []
    for p in times_paths:
        coll = p.replace("/spikes.times.npy", "")
        # check that spikes.clusters also exists in same collection
        clusters_path = coll + "/spikes.clusters.npy"
        if clusters_path in datasets:
            collections.append(coll)

    # De-duplicate while preserving order
    seen = set()
    out = []
    for c in collections:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


# =============================================================================
# STEP 0: Find sessions for your 5 subjects with spike data
# =============================================================================

print("\n" + "=" * 70)
print("STEP 0: Finding sessions with spikes for your 5 subjects")
print("=" * 70)

sessions = []

for subj in SUBJECTS:
    try:
        # Use Alyx search restricted to subject; scan a handful of sessions
        eids = one.search(subject=subj)  # all sessions for that mouse
        print(f"{subj}: found {len(eids)} sessions total")

        # Find the first usable session with spikes
        found = False
        for eid in eids:
            coll = find_spike_collections(eid)
            if coll is None:
                continue

            try:
                st = one.load_dataset(eid, "spikes.times", collection=coll)
                sc = one.load_dataset(eid, "spikes.clusters", collection=coll)
                n_neurons = len(np.unique(sc))
                n_spikes = len(st)

                if n_neurons >= MIN_NEURONS and n_spikes >= MIN_SPIKES_TOTAL:
                    sessions.append({
                        "subject": subj,
                        "eid": eid,
                        "collection": coll,
                        "n_neurons": n_neurons,
                        "n_spikes": n_spikes
                    })
                    print(f"✓ {subj}: {str(eid)[:8]}...  coll={coll}  neurons={n_neurons}  spikes={n_spikes:,}")
                    found = True
                    break
            except Exception:
                continue

        if not found:
            print(f"⚠️ {subj}: no usable session found (spikes missing or too small)")
    except Exception as e:
        print(f"⚠️ {subj}: search failed ({e})")

if not sessions:
    raise RuntimeError("No sessions found for your subjects that contain usable spikes data.")

print("\nSelected sessions:")
print(pd.DataFrame(sessions)[["subject", "eid", "collection", "n_neurons", "n_spikes"]])

# =============================================================================
# STEP 1: Load spikes + compute timescales (per subject)
# =============================================================================

print("\n" + "=" * 70)
print("STEP 1: Computing neuron + PCA timescales")
print("=" * 70)

neuron_rows = []
pc_rows = []

for s in sessions:
    subj = s["subject"]
    eid = s["eid"]
    coll = s["collection"]

    print(f"\n--- {subj} | {str(eid)[:8]}... | {coll} ---")
    spikes_times = one.load_dataset(eid, "spikes.times", collection=coll)
    spikes_clusters = one.load_dataset(eid, "spikes.clusters", collection=coll)

    # Limit time window for speed (optional)
    t0 = float(np.nanmin(spikes_times))
    t1 = float(np.nanmax(spikes_times))
    if WIN_SEC is not None:
        t1 = min(t1, t0 + WIN_SEC)

    unique_neurons = np.unique(spikes_clusters)
    print(f"Neurons: {len(unique_neurons)}, time window: [{t0:.3f}, {t1:.3f}] sec, bin={BIN_SIZE}s")

    # Build binned matrix X: (T bins, N neurons) using first N neurons for speed
    n_use = min(300, len(unique_neurons))  # keep consistent with teammate
    X = []

    kept_neurons = []
    for neuron_id in unique_neurons[:n_use]:
        neuron_spikes = spikes_times[spikes_clusters == neuron_id]
        neuron_spikes = neuron_spikes[(neuron_spikes >= t0) & (neuron_spikes < t1)]
        counts = bin_spike_train(neuron_spikes, t0, t1, BIN_SIZE)
        if counts is None:
            continue
        # drop extremely silent neurons (avoid degenerate autocorr)
        if counts.sum() < 5:
            continue
        X.append(counts)
        kept_neurons.append(int(neuron_id))

    if len(X) < 10:
        print(f"⚠️ {subj}: too few usable neurons after filtering ({len(X)})")
        continue

    X = np.stack(X, axis=1)  # (T, N)
    dt = BIN_SIZE

    # Z-score each neuron across time
    Xz = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)

    # ---- Single-neuron timescales ----
    taus = []
    for j in range(Xz.shape[1]):
        ac = autocorr_fft(Xz[:, j])
        tau = fit_tau_from_ac(ac, dt=dt, max_lag_bins=MAX_LAG_BINS)
        taus.append(tau)
        neuron_rows.append({
            "subject": subj,
            "eid": str(eid),
            "collection": coll,
            "neuron_id": kept_neurons[j],
            "tau_sec": tau
        })

    taus = np.array(taus, dtype=float)
    print(f"Neuron taus (sec): median={np.nanmedian(taus):.3f}, mean={np.nanmean(taus):.3f}")

    # ---- PCA latent timescales (timescale hierarchy) ----
    n_pcs = min(N_PCS, Xz.shape[1])
    pca = PCA(n_components=n_pcs)
    Z = pca.fit_transform(Xz)  # (T, n_pcs)

    for k in range(n_pcs):
        ac = autocorr_fft(Z[:, k])
        tau_k = fit_tau_from_ac(ac, dt=dt, max_lag_bins=MAX_LAG_BINS)
        pc_rows.append({
            "subject": subj,
            "eid": str(eid),
            "pc": k + 1,
            "tau_sec": tau_k,
            "var_explained": float(pca.explained_variance_ratio_[k])
        })

    print(f"PC taus (first 5): {[round(r['tau_sec'],3) for r in pc_rows if r['subject']==subj][:5]}")

neuron_df = pd.DataFrame(neuron_rows)
pc_df = pd.DataFrame(pc_rows)

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("Neuron median tau (sec) by subject:")
print(neuron_df.groupby("subject")["tau_sec"].median().sort_values(ascending=False))

print("\nPC1 tau (sec) by subject:")
print(pc_df[pc_df["pc"] == 1].set_index("subject")["tau_sec"].sort_values(ascending=False))

# =============================================================================
# STEP 2: Plots
# =============================================================================

print("\n" + "=" * 70)
print("STEP 2: Plotting timescales")
print("=" * 70)

# 1) Distribution of neuron taus per subject
plt.figure(figsize=(10, 4))
for subj in SUBJECTS:
    vals = neuron_df.loc[neuron_df["subject"] == subj, "tau_sec"].dropna().values
    if len(vals) == 0:
        continue
    plt.hist(vals, bins=30, alpha=0.4, label=f"{subj} (n={len(vals)})")
plt.xlabel("Neuron timescale τ (sec)")
plt.ylabel("Count")
plt.title("Single-neuron timescale distributions")
plt.legend()
plt.tight_layout()
plt.show()

# 2) Timescale hierarchy: τ vs PC index (mean ± SEM per subject)
plt.figure(figsize=(8, 4))
for subj in SUBJECTS:
    sub = pc_df[pc_df["subject"] == subj].sort_values("pc")
    if len(sub) == 0:
        continue
    plt.plot(sub["pc"], sub["tau_sec"], marker="o", linewidth=2, label=subj)
plt.xlabel("PC index")
plt.ylabel("PC timescale τ (sec)")
plt.title("Timescale hierarchy in PCA latent space")
plt.legend()
plt.tight_layout()
plt.show()

# 3) τ vs variance explained (per PC)
plt.figure(figsize=(8, 4))
plt.scatter(pc_df["var_explained"], pc_df["tau_sec"], alpha=0.6, s=20)
plt.xlabel("Variance explained (PC)")
plt.ylabel("PC timescale τ (sec)")
plt.title("Do high-variance PCs evolve more slowly?")
plt.tight_layout()
plt.show()
