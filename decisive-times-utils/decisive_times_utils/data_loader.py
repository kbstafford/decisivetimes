from one.api import ONE
import pandas as pd
from tqdm import tqdm
from pathlib import Path

one = ONE(base_url='https://openalyx.internationalbrainlab.org')

# Find 50/50 bias trials
# eid | subject | trial | choice | rt | correct | signed_contrast
def load_unbiased_trials(min_trials=5, max_sessions=50):
    eids = one.search(task='biasedChoiceWorld')
    eids = eids[:max_sessions]

    print(f"Processing {len(eids)} sessions")

    dfs = []
    total_trials = 0

    for eid in tqdm(eids, desc="Sessions"):
        try:
            trials = one.load_object(eid, 'trials')
            neutral = (trials.probabilityLeft == 0.5)
            correct = (trials.feedbackType == 1)
            mask = neutral & correct

            if not mask.any():
                continue

            df = pd.DataFrame({
                "eid": eid,
                "response_times": trials.response_times[mask],
                "choice": trials.choice[mask],
                "stimOn_times": trials.stimOn_times[mask],
                "contrastLeft": trials.contrastLeft[mask],
                "contrastRight": trials.contrastRight[mask],
                "feedback_times": trials.feedback_times[mask],
                "feedbackType": trials.feedbackType[mask],
            })

            if len(df) >= min_trials:
                dfs.append(df)

        except Exception as e:
            continue

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs)



ROOT = Path(__file__).resolve().parents[2]   # from decisive_times_utils/... up to decisivetimes/
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTFILE = DATA_DIR / "unbiased_trials.parquet"

df = load_unbiased_trials(min_trials=10, max_sessions=20)
df["eid"] = df["eid"].astype(str)
df.to_parquet(OUTFILE)