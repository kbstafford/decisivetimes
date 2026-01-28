#!/usr/bin/env python3
"""
ibl_spikes_trial_binning.py

Single-file utility to:
  1) pick a random IBL session with spike time data (spikes.times),
  2) load spike times + clusters for a chosen probe,
  3) (option A) bin the entire session into (n_clusters, n_bins) using IBL's bincount2D,
  4) (option B) bin spikes per trial window into a list of (n_clusters, n_bins) arrays,
     and return a pandas DataFrame of trial metadata.

Requires:
  pip install ONE-api ibllib pandas numpy

Example usage:

  # Random session, per-trial binning aligned to stimOn_times:
  python ibl_spikes_trial_binning.py --mode per-trial --bin 0.02 --align stimOn_times --t-before 0.2 --t-after 0.6

  # Specific session:
  python ibl_spikes_trial_binning.py --mode per-trial --eid <EID> --probe probe00

  # Session-wide binning:
  python ibl_spikes_trial_binning.py --mode session --bin 0.02 --out session_binned_spikes.npz

Outputs:
  - per-trial: a .npz with counts_by_trial (object array of 2D int32 arrays), rel_bin_edges, cluster_ids,
              plus trials_df as CSV alongside.
  - session: a .npz with counts, bin_edges, cluster_ids.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from one.api import ONE

from brainbox.io.one import SpikeSortingLoader
from brainbox.processing import bincount2D


# -----------------------------
# Data containers
# -----------------------------

@dataclass
class BinnedSpikes:
    counts: np.ndarray        # (n_clusters, n_bins)
    bin_edges: np.ndarray     # (n_bins + 1,)
    cluster_ids: np.ndarray   # (n_clusters,)


@dataclass
class TrialBinnedSpikes:
    counts_by_trial: List[np.ndarray]   # list of (n_clusters, n_bins)
    trials_df: pd.DataFrame             # one row per (kept) trial
    cluster_ids: np.ndarray             # (n_clusters,)
    rel_bin_edges: np.ndarray           # (n_bins + 1,) edges relative to align time


# -----------------------------
# Session selection / loading
# -----------------------------

def pick_random_session_with_dataset(
    one: ONE,
    dataset: str = "spikes.times",
    project: Optional[str] = "brainwide",
    seed: Optional[int] = None,
) -> str:
    """
    Return a random experiment ID (eid) for a session that contains `dataset`.
    """
    rng = np.random.default_rng(seed)
    kwargs = {"datasets": dataset}
    if project:
        kwargs["project"] = project

    eids = one.search(**kwargs)
    if not eids:
        raise RuntimeError(
            f"No sessions found with dataset={dataset!r} (project={project!r}). "
            "Try --project '' (none) or a different dataset."
        )
    return str(rng.choice(eids))


def load_spike_times_for_probe(
    one: ONE,
    eid: str,
    probe: str = "probe00",
    spike_sorter: str = "pykilosort",
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load spike times and spike cluster ids for a given session + probe.

    Returns
    -------
    spike_times : (n_spikes,) float seconds
    spike_clusters : (n_spikes,) int cluster id per spike
    """
    sl = SpikeSortingLoader(one=one, eid=eid, pname=probe, spike_sorter=spike_sorter)
    spikes, _, _ = sl.load_spike_sorting(**kwargs)
    spike_times = np.asarray(spikes["times"], dtype=float)
    spike_clusters = np.asarray(spikes["clusters"], dtype=np.int64)
    return spike_times, spike_clusters


# -----------------------------
# Trials handling
# -----------------------------

def _trials_to_df(trials_obj) -> pd.DataFrame:
    """
    Convert an ALF trials object (AlfBunch-like) into a DataFrame (1 row per trial).
    Assumes:
      - all fields are 1-D except `intervals` and `intervals_bpod` (shape N x 2)
    """
    data = {}

    for k, v in trials_obj.items():
        a = np.asarray(v)

        # Split 2-column interval fields
        if a.ndim == 2 and a.shape[1] == 2:
            data[f"{k}_start"] = a[:, 0]
            data[f"{k}_end"] = a[:, 1]
        else:
            data[k] = a

    return pd.DataFrame(data)


def _add_derived_trial_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a few helpful derived columns used in IBL task analyses.

    Common IBL trials fields (often present):
      contrastLeft, contrastRight, choice, feedbackType, probabilityLeft,
      stimOn_times, goCue_times, response_times, feedback_times, etc.
    """
    df = df.copy()

    # Signed contrast (Right - Left)
    if "contrastLeft" in df.columns and "contrastRight" in df.columns:
        cl = df["contrastLeft"].to_numpy(dtype=float)
        cr = df["contrastRight"].to_numpy(dtype=float)
        df["signed_contrast"] = np.nan_to_num(cr, nan=0.0) - np.nan_to_num(cl, nan=0.0)
        df["stim_side"] = np.sign(df["signed_contrast"]).astype(int)  # +1 right, -1 left, 0 none/equal

    # Human-friendly choice label (IBL convention: -1 left, +1 right, 0 nogo)
    if "choice" in df.columns:
        choice = df["choice"].to_numpy()
        df["choice_str"] = np.where(
            choice == -1, "L",
            np.where(choice == 1, "R",
            np.where(choice == 0, "NoGo", "Other"))
        )

    # Correct/incorrect (often feedbackType: +1 correct, -1 incorrect)
    if "feedbackType" in df.columns:
        fb = df["feedbackType"].to_numpy()
        df["correct"] = (fb == 1)

    # Reaction time if possible
    if "goCue_times" in df.columns and "response_times" in df.columns:
        gc = df["goCue_times"].to_numpy(dtype=float)
        rt = df["response_times"].to_numpy(dtype=float) - gc
        df["rt"] = rt

    return df


# -----------------------------
# Binning (per-trial)
# -----------------------------

def get_trial_binned_spikes_and_trials_df(
    one: ONE,
    eid: str,
    probe: str = "probe00",
    align_event: str = "stimOn_times",
    t_before: float = 0.2,
    t_after: float = 0.6,
    bin_size_s: float = 0.02,
    good_units: bool = False,
    spike_sorter: str = "pykilosort",
) -> TrialBinnedSpikes:
    """
    Load IBL trials + spike times and return (n_clusters, n_bins) spike counts per trial.

    Returns
    -------
    TrialBinnedSpikes:
      counts_by_trial: list length n_trials_kept, each array (n_clusters, n_bins)
      trials_df: DataFrame with trial metadata + derived columns (n_trials_kept rows)
      cluster_ids: global cluster id list defining row order for all trial arrays
      rel_bin_edges: bin edges relative to align time (same for all trials)
    """
    if bin_size_s <= 0:
        raise ValueError("bin_size_s must be > 0.")
    if t_before < 0 or t_after <= 0:
        raise ValueError("t_before must be >= 0 and t_after must be > 0.")

    # ---- Load trials
    trials = one.load_object(eid, "trials")
    trials_df = _add_derived_trial_columns(_trials_to_df(trials))

    if align_event not in trials_df.columns:
        raise KeyError(
            f"align_event={align_event!r} not found in trials. "
            f"Available columns include: {list(trials_df.columns)}"
        )

    align_times = trials_df[align_event].to_numpy(dtype=float)

    # Keep only trials with finite align times
    keep = np.isfinite(align_times)
    if not np.any(keep):
        raise RuntimeError(f"No finite align times found in trials[{align_event!r}].")

    trials_df = trials_df.loc[keep].reset_index(drop=True)
    align_times = align_times[keep]

    # ---- Load spikes
    spike_times, spike_clusters = load_spike_times_for_probe(
        one=one,
        eid=eid,
        probe=probe,
        spike_sorter=spike_sorter,
        good_units=good_units,
    )

    # Global cluster list / row order
    cluster_ids = np.unique(spike_clusters)
    n_clusters = cluster_ids.size
    clu_to_row = {int(c): i for i, c in enumerate(cluster_ids)}

    # Fixed bins across trials (in relative time)
    n_bins = int(np.ceil((t_before + t_after) / bin_size_s))
    rel_bin_edges = (-t_before) + bin_size_s * np.arange(n_bins + 1, dtype=float)

    counts_by_trial: List[np.ndarray] = []

    for t_align in align_times:
        # Define window
        t0 = t_align - t_before
        t1 = t_align + t_after

        # Mask spikes explicitly (this is the key fix)
        mask = (
            np.isfinite(spike_times) &
            np.isfinite(spike_clusters) &
            (spike_times >= t0) &
            (spike_times < t1)        # strict < avoids right-edge bin issue
        )

        st = spike_times[mask]
        sc = spike_clusters[mask]

        # If no spikes in this window, just return zeros
        if st.size == 0:
            trial_counts = np.zeros((n_clusters, n_bins), dtype=np.int32)
            counts_by_trial.append(trial_counts)
            continue

        sub_counts, sub_edges, sub_clu = bincount2D(
            st,
            sc,
            xbin=bin_size_s,
            xlim=(t0, t1),
        )

        sub_counts = np.asarray(sub_counts, dtype=np.int32)
        sub_clu = np.asarray(sub_clu, dtype=np.int64)

        trial_counts = np.zeros((n_clusters, n_bins), dtype=np.int32)

        # Defensive: if numeric edge issues produce off-by-one bins
        nb = min(n_bins, sub_counts.shape[1])
        if sub_counts.size and nb > 0:
            for i, cid in enumerate(sub_clu):
                r = clu_to_row.get(int(cid))
                if r is not None:
                    trial_counts[r, :nb] = sub_counts[i, :nb]

        counts_by_trial.append(trial_counts)

    return TrialBinnedSpikes(
        counts_by_trial=counts_by_trial,
        trials_df=trials_df,
        cluster_ids=cluster_ids,
        rel_bin_edges=rel_bin_edges,
    )
