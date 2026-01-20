import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hssm
import arviz as az

df = pd.read_parquet("../../data/unbiased_trials.parquet")
print(df.columns)
print(df.shape)

def prepare_trials(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["signed_contrast"] = d["contrastRight"].fillna(0) - d["contrastLeft"].fillna(0)
    d["response"] = (d["choice"]).astype(int)
    d["rt"] = (d["feedback_times"] - d["stimOn_times"]).astype(float)
    d = d[(d["rt"] > 0.1) & (d["rt"] < 10)].copy() # filters out outliers

    keep = ["rt", "response", "signed_contrast", "prev_choice", "prev_reward", "block_bias"]
    for c in ["subject", "eid"]:
        if c in d.columns:
            keep.append(c)

    # sort within session for history terms
    d = d.sort_values(["eid", "stimOn_times"]).copy()
    d["prev_choice"] = d.groupby("eid")["response"].shift(1).fillna(0.5)
    d["prev_reward"] = (d.groupby("eid")["feedbackType"].shift(1) == 1).astype(float).fillna(0.0)

    return d[["eid", "subject", "rt", "response", "signed_contrast", "prev_choice", "prev_reward"]]

hssm_df = prepare_trials(df)

model = hssm.HSSM(
    data=hssm_df,
    model="ddm",
    include=[
        {
            "name": "v",
            "formula": "v ~ 1 + signed_contrast + prev_choice + (1|subject)"
        }
    ]
)
print(model)




