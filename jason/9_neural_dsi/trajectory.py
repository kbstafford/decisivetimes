import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import Any, Dict, Optional, Sequence, Tuple

from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve


def trial_trajectory_projection(
    counts: np.ndarray,
    bin_centers: np.ndarray,
    stim_time: float,
    *,
    scaler: Any,
    w: np.ndarray,
    b: float = 0.0,
    bin_size: Optional[float] = None,
    t_before: float = 0.2,
    t_after: float = 1.0,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Project time-resolved population activity onto a decision axis in a window around stim_time.

    Parameters
    ----------
    counts : (n_neurons, n_bins) array
        Per-session binned spike counts (e.g. from bincount2D).
    bin_centers : (n_bins,) array
        Time of each bin center (seconds).
    stim_time : float
        Alignment time for the trial (seconds).
    scaler : fitted sklearn transformer
        Must implement .transform(X) where X is (n_time, n_neurons).
        Typically StandardScaler from the fitted Pipeline.
    w : (n_neurons,) array
        Decision axis weights in *scaled feature space* (e.g., clf.coef_.ravel()).
    b : float
        Intercept in decision space (e.g., clf.intercept_[0]).
    bin_size : float or None
        If provided, convert counts/bin -> rates by dividing by bin_size.
        If None, infer from bin_centers spacing when possible; otherwise leave as counts.
    t_before, t_after : float
        Window around stim_time.

    Returns
    -------
    t_rel : (T,) array of times relative to stim_time
    proj : (T,) array of projections (logit units)
    """
    if not np.isfinite(stim_time):
        return None, None

    # Infer bin_size if not provided
    if bin_size is None:
        if bin_centers is not None and len(bin_centers) >= 2:
            bin_size = float(np.median(np.diff(bin_centers)))
        else:
            bin_size = None  # fall back to counts

    t0, t1 = stim_time - t_before, stim_time + t_after
    i0 = np.searchsorted(bin_centers, t0, side="left")
    i1 = np.searchsorted(bin_centers, t1, side="right")
    if i1 <= i0:
        return None, None

    # (T, N)
    X_t = counts[:, i0:i1].T
    if bin_size is not None and bin_size > 0:
        X_t = X_t / bin_size  # counts -> rates

    Z_t = scaler.transform(X_t)
    proj = Z_t @ np.asarray(w).ravel() + float(b)
    t_rel = bin_centers[i0:i1] - stim_time
    return t_rel, proj


def mean_sem_trajectories(
    indices: Sequence[int],
    *,
    counts: np.ndarray,
    bin_centers: np.ndarray,
    stim_times: np.ndarray,
    scaler: Any,
    w: np.ndarray,
    b: float = 0.0,
    bin_size: Optional[float] = None,
    t_before: float = 0.2,
    t_after: float = 1.0,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Compute mean ± SEM trajectory over a set of trial indices.

    Parameters
    ----------
    indices
        Trial indices to include.
    counts
        (n_neurons, n_bins) binned spike counts.
    bin_centers
        (n_bins,) bin center times in seconds.
    stim_times
        Per-trial stimulus alignment times in seconds.
    scaler
        Fitted sklearn transformer implementing .transform(X).
    w
        Decision axis weights in scaled feature space.
    b
        Intercept in decision space.
    bin_size
        Bin size in seconds for rate conversion.
    t_before, t_after
        Window around stim_time.

    Returns
    -------
    t_ref, mean, sem  (or (None, None, None) if no valid trajectories)
    """
    trajs = []
    t_ref = None

    for i in indices:
        t_rel, proj = trial_trajectory_projection(
            counts,
            bin_centers,
            stim_times[i],
            scaler=scaler,
            w=w,
            b=b,
            bin_size=bin_size,
            t_before=t_before,
            t_after=t_after,
        )
        if t_rel is None:
            continue
        if t_ref is None:
            t_ref = t_rel
        if len(t_rel) != len(t_ref):
            # Skip trials that fall off the window near session edges.
            continue
        trajs.append(proj)

    if not trajs:
        return None, None, None

    M = np.vstack(trajs)  # (n_trials, T)
    mean = M.mean(axis=0)
    sem = M.std(axis=0, ddof=1) / np.sqrt(M.shape[0])
    return t_ref, mean, sem


def make_group_indices(
    mask: np.ndarray,
    correct: np.ndarray,
    choice: np.ndarray,
    max_trials: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Return dict of indices for {correct/incorrect} × {left/right} choices.

    Parameters
    ----------
    mask
        Boolean mask for trials to include.
    correct
        Boolean array indicating correct trials.
    choice
        Array of choices (-1 for left, 1 for right).
    max_trials
        Optional cap on trials per group.

    Returns
    -------
    dict
        Keys: "corr_L", "corr_R", "err_L", "err_R" with index arrays.
    """
    mask = np.asarray(mask, bool)
    correct = np.asarray(correct, bool)

    groups = {
        "corr_L": np.where(mask & correct & (choice == -1))[0],
        "corr_R": np.where(mask & correct & (choice == 1))[0],
        "err_L": np.where(mask & ~correct & (choice == -1))[0],
        "err_R": np.where(mask & ~correct & (choice == 1))[0],
    }
    if max_trials is not None:
        groups = {k: v[:max_trials] for k, v in groups.items()}
    return groups


def plot_correct_incorrect_panels(
    groups: Dict[str, np.ndarray],
    model: Any,
    mode: str = "spaghetti",
    *,
    counts: np.ndarray,
    bin_centers: np.ndarray,
    stim_times: np.ndarray,
    bin_size: Optional[float] = None,
    t_before: float = 0.2,
    t_after: float = 1.0,
    ylim: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """
    Plot two panels (Correct | Incorrect). `model` is a fitted Pipeline that
    contains "scaler" and "clf" steps (e.g. best_pipe from GridSearchCV.refit=True).

    Parameters
    ----------
    groups
        Dict mapping "corr_L","corr_R","err_L","err_R" to trial indices.
    model
        Fitted sklearn Pipeline with named_steps["scaler"] and ["clf"].
    mode
        "spaghetti" or "mean_sem".
    counts
        (n_neurons, n_bins) binned spike counts.
    bin_centers
        (n_bins,) bin center times in seconds.
    stim_times
        Per-trial stimulus alignment times in seconds.
    bin_size
        Bin size in seconds for rate conversion.
    t_before, t_after
        Window around stim_time.
    ylim
        Optional y-limits for spaghetti plot.

    Returns
    -------
    matplotlib.figure.Figure
        The resulting figure.
    """
    # extract context from model
    scaler = model.named_steps["scaler"]
    clf = model.named_steps["clf"]
    w = clf.coef_.ravel()
    b = float(clf.intercept_.ravel()[0])

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 4),
        sharex=True,
        sharey=True,
    )

    def _compute_spaghetti_ylim(
        all_indices: Sequence[int],
    ) -> Optional[Tuple[float, float]]:
        """Robust symmetric y-lims based on 1–99% quantiles."""
        proj_vals = []

        for i in all_indices:
            _, proj = trial_trajectory_projection(
                counts,
                bin_centers,
                stim_times[i],
                scaler=scaler,
                w=w,
                b=b,
                bin_size=bin_size,
                t_before=t_before,
                t_after=t_after,
            )
            if proj is not None:
                proj_vals.append(proj)

        if not proj_vals:
            return None

        y = np.concatenate(proj_vals)
        q_lo, q_hi = np.quantile(y, [0.01, 0.99])
        M = max(abs(q_lo), abs(q_hi))
        return (-M, M)

    if mode == "spaghetti" and ylim is None:
        all_indices = (
            groups["corr_L"].tolist()
            + groups["corr_R"].tolist()
            + groups["err_L"].tolist()
            + groups["err_R"].tolist()
        )
        ylim = _compute_spaghetti_ylim(all_indices)

    def _decorate(ax: Axes, title: str) -> None:
        """Apply consistent axis labeling and optional limits."""
        ax.axvline(0, color="k", lw=1)
        ax.axhline(0, color="k", lw=1)
        ax.set_title(title)
        ax.set_xlabel("Time from stim onset (s)")
        if ylim is not None:
            ax.set_ylim(ylim)

    def _alpha(n: int) -> float:
        """Heuristic to keep plots readable as trial count grows."""
        return float(np.clip(1.0 / (n**0.25), 0.05, 0.3))

    def _plot_spaghetti(ax: Axes, idx_L: Sequence[int], idx_R: Sequence[int]) -> None:
        """Plot all trial trajectories in a light 'spaghetti' style."""
        alpha = _alpha(len(idx_L) + len(idx_R))
        for i in idx_L:
            t_rel, proj = trial_trajectory_projection(
                counts,
                bin_centers,
                stim_times[i],
                scaler=scaler,
                w=w,
                b=b,
                bin_size=bin_size,
                t_before=t_before,
                t_after=t_after,
            )
            if t_rel is not None:
                ax.plot(t_rel, proj, color="tab:blue", lw=0.5, alpha=alpha)
        for i in idx_R:
            t_rel, proj = trial_trajectory_projection(
                counts,
                bin_centers,
                stim_times[i],
                scaler=scaler,
                w=w,
                b=b,
                bin_size=bin_size,
                t_before=t_before,
                t_after=t_after,
            )
            if t_rel is not None:
                ax.plot(t_rel, proj, color="tab:orange", lw=0.5, alpha=alpha)

    def _plot_mean_sem(ax: Axes, idx_L: Sequence[int], idx_R: Sequence[int]) -> None:
        """Plot mean ± SEM trajectories for left/right choices."""
        tL, mL, sL = mean_sem_trajectories(
            idx_L,
            counts=counts,
            bin_centers=bin_centers,
            stim_times=stim_times,
            scaler=scaler,
            w=w,
            b=b,
            bin_size=bin_size,
            t_before=t_before,
            t_after=t_after,
        )
        tR, mR, sR = mean_sem_trajectories(
            idx_R,
            counts=counts,
            bin_centers=bin_centers,
            stim_times=stim_times,
            scaler=scaler,
            w=w,
            b=b,
            bin_size=bin_size,
            t_before=t_before,
            t_after=t_after,
        )
        if tL is not None:
            ax.plot(tL, mL, color="tab:blue", lw=2)
            ax.fill_between(tL, mL - sL, mL + sL, color="tab:blue", alpha=0.2)
        if tR is not None:
            ax.plot(tR, mR, color="tab:orange", lw=2)
            ax.fill_between(tR, mR - sR, mR + sR, color="tab:orange", alpha=0.2)

    plot_fn = _plot_spaghetti if mode == "spaghetti" else _plot_mean_sem

    plot_fn(axes[0], groups["corr_L"], groups["corr_R"])
    _decorate(
        axes[0], "Correct trials" + ("" if mode == "spaghetti" else " (mean ± SEM)")
    )

    plot_fn(axes[1], groups["err_L"], groups["err_R"])
    _decorate(
        axes[1], "Incorrect trials" + ("" if mode == "spaghetti" else " (mean ± SEM)")
    )

    axes[0].set_ylabel("Decision-axis projection (logit)")

    axes[1].legend(
        handles=[
            plt.Line2D([], [], color="tab:blue", lw=2, label="Left choice"),
            plt.Line2D([], [], color="tab:orange", lw=2, label="Right choice"),
        ],
        loc="upper right",
        frameon=True,
    )
    plt.tight_layout()
    return fig


def plot_testset_roc(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    mask_corr: np.ndarray,
    mask_err: np.ndarray,
) -> Optional[plt.Figure]:
    """
    Plot ROC curves for overall/correct/incorrect subsets of the test set.

    Parameters
    ----------
    y_test
        True labels for the test set.
    y_pred
        Predicted class labels.
    y_prob
        Predicted probability for the positive class.
    mask_corr
        Mask for correct trials in the test set.
    mask_err
        Mask for incorrect trials in the test set.

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure handle, or None if ROC cannot be computed.
    """
    if np.unique(y_test).size < 2:
        return

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(5, 5))

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax.plot(fpr, tpr, label=f"Overall (AUC={auc:.3f}, acc={acc:.3f})")

    if mask_corr.sum() >= 10 and np.unique(y_test[mask_corr]).size >= 2:
        acc_c = accuracy_score(y_test[mask_corr], y_pred[mask_corr])
        fpr_c, tpr_c, _ = roc_curve(y_test[mask_corr], y_prob[mask_corr])
        auc_c = roc_auc_score(y_test[mask_corr], y_prob[mask_corr])
        ax.plot(
            fpr_c,
            tpr_c,
            label=f"Correct (AUC={auc_c:.3f}, acc={acc_c:.3f})",
        )

    if mask_err.sum() >= 10 and np.unique(y_test[mask_err]).size >= 2:
        acc_e = accuracy_score(y_test[mask_err], y_pred[mask_err])
        fpr_e, tpr_e, _ = roc_curve(y_test[mask_err], y_prob[mask_err])
        auc_e = roc_auc_score(y_test[mask_err], y_prob[mask_err])
        ax.plot(
            fpr_e,
            tpr_e,
            label=f"Incorrect (AUC={auc_e:.3f}, acc={acc_e:.3f})",
        )

    ax.plot([0, 1], [0, 1], ls="--", color="0.6", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Test set ROC")
    ax.set_aspect("equal", "box")
    ax.legend(loc="lower right")
    plt.tight_layout()
    return fig
