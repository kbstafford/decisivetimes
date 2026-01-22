import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hssm
import arviz as az
import pytensor
import jax
import os
import torch

def prepare_trials(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["signed_contrast"] = d["contrastRight"].fillna(0) - d["contrastLeft"].fillna(0)

    d["response"] = d["choice"].astype(int)
    d = d[d["response"].isin([-1, 1])].copy()

    d["rt"] = (d["feedback_times"] - d["stimOn_times"]).astype(float)
    d = d[(d["rt"] > 0.1) & (d["rt"] < 10)].copy() # filters out outliers

    keep = ["rt", "response", "signed_contrast", "prev_choice", "prev_reward", "block_bias"]
    for c in ["subject", "eid"]:
        if c in d.columns:
            keep.append(c)

    # sort within session for history terms
    d = d.sort_values(["eid", "stimOn_times"]).copy()

    d["prev_choice"] = d.groupby("eid")["response"].shift(1).fillna(0.0)

    d["prev_reward"] = (d.groupby("eid")["feedbackType"].shift(1) == 1).astype(float).fillna(0.0)

    d["abs_contrast"] = d["signed_contrast"].abs()

    # map to bins that match your task
    def evidence_bin(x):
        if x == 0:
            return "0"
        elif x <= 0.0625:
            return "6.25"
        elif x <= 0.125:
            return "12.5"
        elif x <= 0.25:
            return "25"
        else:
            return "50_100"

    d["evidence_bin"] = d["abs_contrast"].apply(evidence_bin).astype("category")

    return d[["eid", "subject", "rt", "response", "signed_contrast",
              "abs_contrast", "evidence_bin", "prev_choice", "prev_reward"]]

if __name__ == "__main__":
    df = pd.read_parquet("../../data/unbiased_trials.parquet")
    print(df.columns)
    print(df.shape)

    hssm_df = prepare_trials(df)
    hssm_df["subject"] = hssm_df["subject"].astype("category")
    hssm_df["subject"] = hssm_df["subject"].cat.remove_unused_categories()
    print("n subjects:", hssm_df["subject"].nunique())

    counts = hssm_df.groupby("subject").size().sort_values()
    print(counts)
    print("min trials per subject:", counts.min())

    hssm_df["evidence_bin"] = hssm_df["evidence_bin"].astype("category")
    hssm_df["evidence_bin"] = hssm_df["evidence_bin"].cat.set_categories(
        ["50_100", "25", "12.5", "6.25", "0"],
        ordered=True
    )

    # Verify the columns are there
    print(hssm_df.columns)
    print(hssm_df.head())

    model = hssm.HSSM(
        data=hssm_df,
        model="ddm",
        include=[
            {
                "name": "v",
                "formula": "v ~ 1 + signed_contrast + prev_choice * evidence_bin + (1|subject)"

            }
        ]
    )

    print(model)

    idata = model.sample(
        sampler="blackjax",
        chains=4,
        cores=4,
        draws=1500,
        tune=1500,
        target_accept=1,
        idata_kwargs={"log_likelihood": False},
        progressbar=True,
    )

    az.plot_trace(idata, var_names=[
        "a", "t",
        "v_Intercept", "v_signed_contrast", "v_prev_choice",
        "v_1|subject_sigma"
    ])


    plt.tight_layout()
    plt.savefig("hddm1.png")
    plt.show()

    az.plot_posterior(idata, var_names=["v_signed_contrast"], hdi_prob=0.95)
    plt.title("Posterior for contrast effect on drift (v_signed_contrast)")
    plt.show()
    print(az.summary(idata, var_names=["v_signed_contrast"], hdi_prob=0.95))

    az.plot_forest(
        idata,
        var_names=["v_prev_choice", "v_prev_choice:evidence_bin"],
        combined=True,
        hdi_prob=0.95
    )
    plt.title("History effect by evidence bin (interaction terms)")
    plt.show()



