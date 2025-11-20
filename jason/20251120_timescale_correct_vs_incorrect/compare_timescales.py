import matplotlib.pyplot as plt
import numpy as np

from brainbox.io.one import SessionLoader, SpikeSortingLoader
from one.api import ONE
from scipy import stats
from tqdm import tqdm

from decisive_times_utils.metrics import compute_autocorr_timescale

one = ONE()
pid = '695476f6-4c14-4a2f-b658-948514629079' # example
eid, _ = one.pid2eid(pid)

# use brainbox helper loading functions
sl = SessionLoader(eid=eid, one=one)
sl.load_trials()

ssl = SpikeSortingLoader(one=one, pid=pid)
spikes, clusters, channels = ssl.load_spike_sorting()
clusters = ssl.merge_clusters(spikes, clusters, channels)

spike_times_per_cluster = [
    spikes.times[spikes.clusters == id] for id in np.sort(np.unique(spikes.clusters))
]

# for each cluster and trial, get autocorrelation
n_clusters = len(spike_times_per_cluster)
n_trials = sl.trials.shape[0]
timescales = np.zeros((n_clusters, n_trials)) + np.nan
for i, times in enumerate(tqdm(spike_times_per_cluster)): # this is pretty slow :(
    for j, trial in sl.trials.iterrows():
        t0 = trial.intervals_0 # start
        t1 = trial.intervals_1 # end
        result = compute_autocorr_timescale(times, t0, t1)
        timescales[i, j] = result.tau_int

correct_idx = np.where(sl.trials.feedbackType == 1)[0]
incorrect_idx = np.where(sl.trials.feedbackType == -1)[0]

mean_timescale_correct = np.nanmean(timescales[:, correct_idx], axis=1)
print("Mean correct timescale:", np.nanmean(mean_timescale_correct))
ci95_timescale_correct = 1.96 * stats.sem(timescales[:, correct_idx], axis=1, nan_policy="omit") # assume gaussian for CI

mean_timescale_incorrect = np.nanmean(timescales[:, incorrect_idx], axis=1)
print("Mean incorrect timescale:", np.nanmean(mean_timescale_incorrect))
ci95_timescale_incorrect = 1.96 * stats.sem(timescales[:, incorrect_idx], axis=1, nan_policy="omit")

# assume gaussian (not verified), use paired t-test
ttest = stats.ttest_rel(mean_timescale_correct, mean_timescale_incorrect, nan_policy="omit")

plt.figure(figsize=(5,4))
plt.axline((0, 0), slope=1, linestyle='--', color='k')
plt.scatter(mean_timescale_correct, mean_timescale_incorrect, 5)
plt.xlabel("Mean τ during correct trials")
plt.ylabel("Mean τ during incorrect trials")
plt.title(f"p={ttest.pvalue:.3e}, two-sided paired t-test")
plt.savefig("mean_timescale_correct_vs_incorrect.png")

plt.figure(figsize=(5,4))
plt.axline((0, 0), slope=1, linestyle='--', color='k')
plt.errorbar(mean_timescale_correct, mean_timescale_incorrect, xerr=ci95_timescale_correct, yerr=ci95_timescale_incorrect, fmt='none', capsize=5, alpha=0.4)
plt.xlabel("Mean τ during correct trials")
plt.ylabel("Mean τ during incorrect trials")
plt.title(f"p={ttest.pvalue:.3e}, two-sided paired t-test")
plt.savefig("mean_timescale_correct_vs_incorrect_errorbars.png")

plt.figure(figsize=(5,4))
plt.hist(mean_timescale_correct, bins=50, label="Correct trials", alpha=0.5)
plt.hist(mean_timescale_incorrect, bins=50, label="Incorrect trials", alpha=0.5)
plt.xlabel("Mean τ")
plt.ylabel("Number of neurons")
plt.legend()
plt.savefig("mean_timescale_correct_vs_incorrect_hist.png")
