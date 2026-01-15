import pandas as pd
import numpy as np
from pyddm import Model, Fittable
from pyddm.models import Drift, NoiseConstant, BoundConstant, OverlayNonDecision, ICPointSourceCenter
from pyddm.functions import fit_adjust_model, display_model
from pyddm import Sample


df = pd.read_parquet("../../data/unbiased_trials.parquet")
print(df.head())
print(df.shape)


df = df.copy()
df['signed_contrast'] = df['contrastRight'].fillna(0) - df['contrastLeft'].fillna(0)
df['rt'] = df['response_times']
df['response'] = (df['choice'] == 1).astype(int)
df = df[(df['rt'] > 0.1) & (df['rt'] < 5)]

samp = Sample(
    rts=df["rt"].values,
    choices=df["response"].values,
    conditions={"signed_contrast": df["signed_contrast"].values},
)

# not finished (jan 15 2026)