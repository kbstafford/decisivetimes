from __future__ import annotations
from typing import Union

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from brainbox.processing import bincount2D
from one.api import ONE

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


from load import (
    concatenate_spikes_by_probe,
    load_all_spikes_by_probe_for_session,
    load_trials_df,
    pick_random_session_with_spikesorting,
    premotor_population_activity,
)
from trajectory import (
    make_group_indices,
    plot_correct_incorrect_panels,
    plot_testset_roc,
)


def predict_left_right(
    eid: str,
    out_path: Union[str, Path],
    one: ONE = ONE(),
    bin_size: float = 0.05,
    C_grid: np.array = np.logspace(-2, 1, 5),
    l1_ratio_grid: np.array = np.linspace(0, 1, 5),
) -> None:
    """
    Fit a logistic model to predict left/right choice and save summary plots.

    Parameters
    ----------
    eid
        Session ID.
    out_path
        Output directory for plots.
    one
        ONE client instance.
    bin_size
        Bin size in seconds for spike counts.
    C_grid
        Grid of inverse regularization strengths for logistic regression.
    l1_ratio_grid
        Grid of l1 ratios for elastic-net logistic regression.

    Returns
    -------
    None

    Outputs:
      - ROC curve on the test set
      - Decision-axis trajectory plots (spaghetti and mean ± SEM)

    Usage example:
      python predict_left_right.py --seed 123 --out-path ./predict_left_right
      python predict_left_right.py --eid <eid> --bin-size 0.05
    """

    # ------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------

    out_path = Path(out_path)
    os.makedirs(out_path, exist_ok=True)

    warnings.filterwarnings("ignore")
    print("Predicting left/right decisions from eid:", eid)

    # ------------------------------------------------------------
    # Load spikes from ALL probes
    # ------------------------------------------------------------

    spikes_by_probe = load_all_spikes_by_probe_for_session(one, eid)
    spike_times, spike_clusters = concatenate_spikes_by_probe(spikes_by_probe)

    cluster_ids = np.unique(spike_clusters)
    n_neurons = cluster_ids.size

    print(f"Loaded {spike_times.size:,} spikes from {n_neurons:,} clusters")

    # ------------------------------------------------------------
    # Build pre-motor population vectors from binned data
    # ------------------------------------------------------------

    counts, bin_edges, _ = bincount2D(
        spike_times,
        spike_clusters,
        xbin=bin_size,
    )
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    trials_df = load_trials_df(eid, one=one)
    X_premotor = premotor_population_activity(counts, bin_centers, trials_df)

    # Labels: Left (-1) vs Right (+1)
    choice = trials_df["choice"].to_numpy()

    correct = (trials_df["feedbackType"] == 1).to_numpy()

    # Keep only L/R trials
    mask = np.isin(choice, [-1, 1])

    X = X_premotor[mask]
    y = (choice[mask] == 1).astype(int)  # 0 = Left, 1 = Right

    print("Pre-motor population matrix:", X.shape)

    # ------------------------------------------------------------
    # Fit logistic regression model to predict left/right
    # ------------------------------------------------------------

    # First, grid search for elastic net parameters
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    solver="saga",
                    max_iter=1000,  # TODO: configurable
                    tol=1e-3,
                ),
            ),
        ]
    )

    param_grid = {
        "clf__C": C_grid,
        "clf__l1_ratio": l1_ratio_grid,
    }

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

    gs = GridSearchCV(
        model,
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,  # parallelize
        refit=True,
        return_train_score=False,
    )

    gs.fit(X, y)

    cv_results = pd.DataFrame(gs.cv_results_)
    best = cv_results.loc[
        cv_results["mean_test_score"].idxmax(),
        ["param_clf__C", "param_clf__l1_ratio", "mean_test_score"],
    ]
    best_C = float(best["param_clf__C"])
    best_l1_ratio = float(best["param_clf__l1_ratio"])
    best_acc = float(best["mean_test_score"])

    print(
        f"Best LR model: acc={best_acc:.3f}, C={best_C:g}, l1_ratio={best_l1_ratio:.2f}"
    )

    # Fit final model
    X_train, X_test, y_train, y_test, correct_train, correct_test = train_test_split(
        X, y, correct[mask], test_size=0.1, random_state=0, stratify=y
    )

    model = gs.best_estimator_
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    def eval_subset(name: str, mask: np.ndarray) -> None:
        """Print accuracy/AUC summary for a subset of trials."""
        if mask.sum() < 10:
            print(f"{name}: too few trials")
            return

        acc = accuracy_score(y_test[mask], y_pred[mask])
        auc = roc_auc_score(y_test[mask], y_prob[mask])
        print(f"{name:>12s} | n={mask.sum():3d} | acc={acc:.3f} | auc={auc:.3f}")

    mask_corr = correct_test == 1
    mask_err = correct_test == 0

    eval_subset("All trials", correct_test > -1)
    eval_subset("Correct", mask_corr)
    eval_subset("Incorrect", mask_err)

    # ROC curves for the test set
    fig = plot_testset_roc(y_test, y_pred, y_prob, mask_corr, mask_err)
    if fig is not None:
        fig.savefig(out_path / f"{eid}_test_roc.png", dpi=300)

    # ------------------------------------------------------------
    # Visualize within-trial "decision-axis" trajectories
    # ------------------------------------------------------------

    groups_spaghetti = make_group_indices(mask, correct, choice, max_trials=1000)
    stim_times = trials_df["stimOn_times"].to_numpy()
    fig = plot_correct_incorrect_panels(
        groups_spaghetti,
        model,
        mode="spaghetti",
        counts=counts,
        bin_centers=bin_centers,
        stim_times=stim_times,
        bin_size=bin_size,
    )
    fig.savefig(out_path / f"{eid}_decision_spaghetti.png", dpi=300)

    groups_all = make_group_indices(mask, correct, choice, max_trials=None)
    fig = plot_correct_incorrect_panels(
        groups_all,
        model,
        mode="mean_sem",
        counts=counts,
        bin_centers=bin_centers,
        stim_times=stim_times,
        bin_size=bin_size,
    )
    fig.savefig(out_path / f"{eid}_decision_mean_sem.png", dpi=300)

    warnings.filterwarnings("default")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Predict left/right choice for a random session."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for session selection (default: None).",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=Path("./predict_left_right"),
        help="Output directory for plots (default: ./predict_left_right).",
    )
    parser.add_argument(
        "--eid",
        type=str,
        default=None,
        help="Session ID to use instead of random selection.",
    )
    parser.add_argument(
        "--bin-size",
        type=float,
        default=0.05,
        help="Bin size in seconds for spike counts (default: 0.05).",
    )
    args = parser.parse_args()
    seed = args.seed
    out_path = args.out_path
    bin_size = args.bin_size

    one = ONE()
    eid = args.eid
    if eid is None:
        eid = pick_random_session_with_spikesorting(one, seed=seed)

    predict_left_right(eid, out_path=out_path, one=one, bin_size=bin_size)
