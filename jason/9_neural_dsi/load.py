from __future__ import annotations
from typing import Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from brainbox.io.one import SpikeSortingLoader
from one.api import ONE


# -----------------------------
# Session selection / loading
# -----------------------------


def pick_random_session_with_spikesorting(
    one: ONE,
    seed: Optional[int] = None,
    session_datasets: Tuple[str, ...] = ("spikes.times.npy", "spikes.clusters.npy"),
    **search_kwargs,
) -> str:
    """
    Return a random session eid that has at least one spikesorting dataset somewhere.

    Parameters
    ----------
    one
        ONE client instance.
    seed
        Random seed for selection.
    session_datasets
        Dataset names required for the session.
    **search_kwargs
        Passed to `one.search`.

    Returns
    -------
    str
        Session eid.

    Raises
    ------
    RuntimeError
        If no sessions are found.
    """
    rng = np.random.default_rng(seed)
    eids = one.search(datasets=list(session_datasets), **search_kwargs)
    if not eids:
        raise RuntimeError(f"No sessions found with datasets={session_datasets!r}.")
    return str(rng.choice(eids))


def list_spikesorting_pids_in_session(
    one: ONE,
    eid: str,
    datasets: Tuple[str, ...] = ("spikes.times.npy", "spikes.clusters.npy"),
    **search_kwargs,
) -> Tuple[str, ...]:
    """
    Return all probe insertion IDs (pids) in this session that contain the required datasets.

    Parameters
    ----------
    one
        ONE client instance.
    eid
        Session ID.
    datasets
        Dataset names required for the insertion.
    **search_kwargs
        Passed to `one.search_insertions`.

    Returns
    -------
    tuple of str
        Probe insertion IDs (pids).
    """
    # search_insertions returns probe UUIDs (pids) and supports dataset filtering :contentReference[oaicite:1]{index=1}
    pids = one.search_insertions(session=eid, datasets=list(datasets), **search_kwargs)
    return tuple(str(pid) for pid in pids)


def load_trials_df(eid: str, one: ONE = ONE()) -> pd.DataFrame:
    """
    Load the trials table for a session and return a flat DataFrame.

    Parameters
    ----------
    eid
        Session ID.
    one
        ONE client instance.

    Returns
    -------
    pandas.DataFrame
        Trials table with flattened intervals and reaction time.
    """
    trials = one.load_object(eid, "trials")

    def trials_to_df_simple(trials: Mapping[str, np.ndarray]) -> pd.DataFrame:
        """
        Convert ONE trials object into a flat dataframe with *_start/_end columns.

        Parameters
        ----------
        trials
            Mapping of trial fields to arrays.

        Returns
        -------
        pandas.DataFrame
            Flattened trials table.
        """
        data = {}
        for k, v in trials.items():
            a = np.asarray(v)
            if a.ndim == 2 and a.shape[1] == 2:
                data[f"{k}_start"] = a[:, 0]
                data[f"{k}_end"] = a[:, 1]
            else:
                data[k] = a
        return pd.DataFrame(data)

    trials_df = trials_to_df_simple(trials)

    # Reaction time (optional sanity check)
    trials_df["rt"] = trials_df["firstMovement_times"] - trials_df["stimOn_times"]
    return trials_df


# -----------------------------
# Neural data
# -----------------------------


def load_all_spikes_by_probe_for_session(
    one: ONE,
    eid: str,
    datasets: Tuple[str, ...] = ("spikes.times.npy", "spikes.clusters.npy"),
    **load_kwargs,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Load spikes for ALL probes in a session that have spikesorting.

    Parameters
    ----------
    one
        ONE client instance.
    eid
        Session ID.
    datasets
        Dataset names required for the insertion.
    **load_kwargs
        Passed to `SpikeSortingLoader.load_spike_sorting`.

    Returns
    -------
    spikes_by_probe : dict
        Keys are probe labels (e.g. 'probe00'), values are (spike_times, spike_clusters).

    Raises
    ------
    RuntimeError
        If no qualifying probe insertions are found.
    """
    pids = list_spikesorting_pids_in_session(one, eid, datasets=datasets)
    if not pids:
        raise RuntimeError(
            f"Session {eid} has no insertions with datasets={datasets!r}."
        )

    spikes_by_probe: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for pid in pids:
        # Convert pid -> (eid, pname) for labeling; supported by ONE / docs
        _, pname = one.pid2eid(pid)

        # Load by pid is the most direct and avoids collection/probe naming ambiguity
        sl = SpikeSortingLoader(pid=pid, one=one)
        spikes, _, _ = sl.load_spike_sorting(**load_kwargs)

        spike_times = np.asarray(spikes["times"], dtype=float)
        spike_clusters = np.asarray(spikes["clusters"], dtype=np.int64)
        spikes_by_probe[pname] = (spike_times, spike_clusters)

    return spikes_by_probe


def concatenate_spikes_by_probe(
    spikes_by_probe: Mapping[str, Tuple[np.ndarray, np.ndarray]],
    sort_by_time: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Concatenate spikes across probes into single (spike_times, spike_clusters),
    making cluster IDs globally unique via per-probe offsets.

    Parameters
    ----------
    spikes_by_probe
        Mapping of probe -> (spike_times, spike_clusters).
    sort_by_time
        If True, sort the concatenated spikes by time.

    Returns
    -------
    spike_times_all
        (N,) float array.
    spike_clusters_all
        (N,) int array.
    """
    times_all = []
    clusters_all = []
    offset = 0

    # deterministic order (optional, but helps reproducibility)
    for probe in sorted(spikes_by_probe.keys()):
        t, c = spikes_by_probe[probe]
        t = np.asarray(t, dtype=float)
        c = np.asarray(c, dtype=np.int64)

        if t.size == 0:
            continue

        # make clusters unique by shifting this probe's ids
        clusters_all.append(c + offset)
        times_all.append(t)

        # advance offset by number of cluster IDs used on this probe
        offset += int(c.max()) + 1

    if not times_all:
        return np.array([], dtype=float), np.array([], dtype=np.int64)

    spike_times = np.concatenate(times_all)
    spike_clusters = np.concatenate(clusters_all)

    if sort_by_time:
        order = np.argsort(spike_times, kind="mergesort")  # stable sort
        spike_times = spike_times[order]
        spike_clusters = spike_clusters[order]

    return spike_times, spike_clusters


def premotor_population_activity(
    counts: np.ndarray,
    bin_centers: np.ndarray,
    trials_df: pd.DataFrame,
    stim_key: str = "stimOn_times",
    move_key: str = "firstMovement_times",
) -> np.ndarray:
    """
    Compute pre-motor population activity vectors.

    Parameters
    ----------
    counts
        (n_neurons, n_bins) binned spike counts.
    bin_centers
        (n_bins,) bin center times in seconds.
    trials_df
        Trials table.
    stim_key
        Column name for stimulus onset times.
    move_key
        Column name for movement onset times.

    Returns
    -------
    X
        (n_trials, n_neurons) array. Mean spike count per neuron between stimulus and movement.
    """
    n_neurons = counts.shape[0]
    X_premotor = np.zeros((len(trials_df), n_neurons))

    for i, row in trials_df.iterrows():
        t0 = row[stim_key]
        t1 = row[move_key]

        if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
            continue

        # Find bin indices for this trial's window
        b0 = np.searchsorted(bin_centers, t0, side="left")
        b1 = np.searchsorted(bin_centers, t1, side="right")

        if b1 <= b0:
            continue

        # Mean firing rate per neuron
        duration = t1 - t0
        X_premotor[i] = counts[:, b0:b1].sum(axis=1) / duration

    return X_premotor
