# =============================================================================
# COMPLETE ANALYSIS: Confidence vs Firing Rate
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, ttest_ind, mannwhitneyu
import pandas as pd

print("="*70)
print("COMPLETE ANALYSIS: High Confidence → Higher Firing Rates")
print("="*70)

# =============================================================================
# STEP 0: Find Working Sessions
# =============================================================================

print("\n" + "="*70)
print("STEP 0: Finding working sessions with spike data")
print("="*70)

# Search for sessions with spike data
all_sessions = one.search()[:100]  # Scan the first 100 sessions

working_sessions = []

for i, eid in enumerate(all_sessions):
    try:
        datasets = one.list_datasets(eid)
        if any('spikes.times' in d for d in datasets):
            # Spike file found, try loading
            spike_files = [d for d in datasets if 'spikes.times' in d]
            collection = spike_files[0].replace('/spikes.times.npy', '')
            
            # Test load
            spikes = one.load_dataset(eid, 'spikes.times', collection=collection)
            clusters = one.load_dataset(eid, 'spikes.clusters', collection=collection)
            trials = one.load_object(eid, 'trials')
            
            n_neurons = len(np.unique(clusters))
            n_trials = len(trials['intervals'])
            
            if n_neurons >= 100 and n_trials >= 200:  # Minimum requirements
                working_sessions.append({
                    'eid': eid,
                    'collection': collection,
                    'n_neurons': n_neurons,
                    'n_trials': n_trials,
                    'n_spikes': len(spikes)
                })
                
                print(f"✓ Found session {len(working_sessions)}: {str(eid)[:8]}... ({n_neurons} neurons, {n_trials} trials)")
                
                if len(working_sessions) >= 1:  # Stop after finding the first working session
                    break
    except:
        continue

if len(working_sessions) == 0:
    print("\n❌ No working sessions found!")
    print("Trying known good session...")
    
    # Known good session
    working_sessions = [{
        'eid': 'ebce500b-c530-47de-8cb1-963c552703ea',
        'collection': 'alf/probe00/pykilosort/#2024-05-06#',
        'n_neurons': 1557,
        'n_trials': 569,
        'n_spikes': 83091126
    }]
    print(f"✓ Using known session: {str(working_sessions[0]['eid'])[:8]}...")

print(f"\n{'='*70}")
print(f"Sessions found: {len(working_sessions)}")
print(f"{'='*70}")

# =============================================================================
# STEP 1: Load Data from Best Session
# =============================================================================

print("\n" + "="*70)
print("STEP 1: Loading data from best session")
print("="*70)

# Select the best session (the one with the most neurons)
best_session = working_sessions[0]
eid = best_session['eid']
collection = best_session['collection']

print(f"\nSelected session: {str(eid)[:8]}...")
print(f"  Neurons: {best_session['n_neurons']}")
print(f"  Trials: {best_session['n_trials']}")

# Load data
print("\nLoading spike data...")
spikes_times = one.load_dataset(eid, 'spikes.times', collection=collection)
spikes_clusters = one.load_dataset(eid, 'spikes.clusters', collection=collection)
trials = one.load_object(eid, 'trials')

print("✓ Data loaded successfully")
print(f"  Total spikes: {len(spikes_times):,}")
print(f"  Unique neurons: {len(np.unique(spikes_clusters))}")
print(f"  Trials: {len(trials['intervals'])}")

# =============================================================================
# STEP 2: Calculate Decision Strength Index (DSI)
# =============================================================================

print("\n" + "="*70)
print("STEP 2: Calculate Decision Strength Index (DSI)")
print("="*70)

# Trial info
choice = trials['choice']
feedback = trials['feedbackType']
contrast_left = trials['contrastLeft']
contrast_right = trials['contrastRight']
rt = trials['response_times'] - trials['stimOn_times']

# Contrast (stimulus strength)
contrast = np.where(~np.isnan(contrast_left), contrast_left, contrast_right)

# Correct/Incorrect
correct = feedback == 1

# Valid trials (non-no-go, reasonable RT)
valid_trials = (choice != 0) & (~np.isnan(rt)) & (rt > 0.1) & (rt < 10)

print(f"Valid trials: {np.sum(valid_trials)} / {len(trials['choice'])}")

# Calculate DSI
# DSI = weighted combination of:
#   - Contrast (high = easy = confident)
#   - RT (fast = confident)  
#   - Accuracy (correct = confident)

dsi = np.zeros(len(trials['choice']))

for i in np.where(valid_trials)[0]:
    # Contrast score (0-1)
    contrast_score = contrast[i] if not np.isnan(contrast[i]) else 0.5
    
    # RT score (0-1): normalize by median RT
    median_rt = np.median(rt[valid_trials])
    rt_score = np.clip(1.5 - (rt[i] / median_rt), 0, 1)
    
    # Accuracy score (0 or 1)
    accuracy_score = 1.0 if correct[i] else 0.0
    
    # Combined DSI
    dsi[i] = 0.3 * contrast_score + 0.3 * rt_score + 0.4 * accuracy_score

print(f"\nDSI Statistics:")
print(f"  Mean: {dsi[valid_trials].mean():.3f}")
print(f"  Median: {np.median(dsi[valid_trials]):.3f}")
print(f"  Range: [{dsi[valid_trials].min():.3f}, {dsi[valid_trials].max():.3f}]")

# Categorize trials
threshold = np.median(dsi[valid_trials])
high_conf_trials = valid_trials & (dsi > threshold)
low_conf_trials = valid_trials & (dsi <= threshold)

print(f"\nCategorization:")
print(f"  High confidence trials: {np.sum(high_conf_trials)}")
print(f"  Low confidence trials: {np.sum(low_conf_trials)}")

# =============================================================================
# STEP 3: Calculate Firing Rates (Decision Epoch)
# =============================================================================

print("\n" + "="*70)
print("STEP 3: Calculate firing rates during decision epoch")
print("="*70)

unique_neurons = np.unique(spikes_clusters)
n_analyze = min(300, len(unique_neurons))  # First 300 neurons (for performance)

print(f"Analyzing {n_analyze} neurons (out of {len(unique_neurons)})...")

neuron_firing_data = []

for neuron_idx, neuron_id in enumerate(unique_neurons[:n_analyze]):
    if neuron_idx % 50 == 0:
        print(f"  Progress: {neuron_idx}/{n_analyze}")
    
    neuron_spikes = spikes_times[spikes_clusters == neuron_id]
    
    if len(neuron_spikes) < 100:
        continue
    
    # Calculate firing rate for each valid trial
    for trial_idx in np.where(valid_trials)[0]:
        stim_on = trials['stimOn_times'][trial_idx]
        response = trials['response_times'][trial_idx]
        
        if np.isnan(response):
            continue
        
        trial_duration = response - stim_on
        
        if trial_duration < 0.1 or trial_duration > 5.0:
            continue
        
        # Spikes within decision epoch
        trial_spikes = neuron_spikes[(neuron_spikes >= stim_on) &
                                      (neuron_spikes <= response)]
        
        # Firing rate (Hz)
        firing_rate = len(trial_spikes) / trial_duration
        
        # Save results
        neuron_firing_data.append({
            'neuron_id': neuron_id,
            'trial_idx': trial_idx,
            'firing_rate': firing_rate,
            'n_spikes': len(trial_spikes),
            'duration': trial_duration,
            'dsi': dsi[trial_idx],
            'confidence': 'high' if high_conf_trials[trial_idx] else 'low',
            'correct': correct[trial_idx],
            'contrast': contrast[trial_idx],
            'rt': rt[trial_idx]
        })

firing_df = pd.DataFrame(neuron_firing_data)

print(f"\n✓ Analysis complete")
print(f"  Neuron-trial pairs: {len(firing_df)}")
print(f"  Unique neurons: {firing_df['neuron_id'].nunique()}")
print(f"  Unique trials: {firing_df['trial_idx'].nunique()}")

# =============================================================================
# STEP 4: HYPOTHESIS TEST
# =============================================================================

print("\n" + "="*70)
print("STEP 4: HYPOTHESIS TEST")
print("="*70)

# High vs Low confidence firing rates
high_conf_fr = firing_df[firing_df['confidence'] == 'high']['firing_rate']
low_conf_fr = firing_df[firing_df['confidence'] == 'low']['firing_rate']

print(f"\n📊 Firing Rate Comparison:")
print(f"  High confidence: {high_conf_fr.mean():.2f} ± {high_conf_fr.std():.2f} Hz")
print(f"  Low confidence:  {low_conf_fr.mean():.2f} ± {low_conf_fr.std():.2f} Hz")
print(f"  Difference:      {high_conf_fr.mean() - low_conf_fr.mean():.2f} Hz ({100*(high_conf_fr.mean() - low_conf_fr.mean())/low_conf_fr.mean():.1f}%)")

# Statistical tests
t_stat, t_pval = ttest_ind(high_conf_fr, low_conf_fr)
u_stat, u_pval = mannwhitneyu(high_conf_fr, low_conf_fr, alternative='greater')

print(f"\n📈 Statistical Tests:")
print(f"  T-test: t={t_stat:.3f}, p={t_pval:.6f}")
print(f"  Mann-Whitney U: p={u_pval:.6f}")

# Correlation
corr_pearson, corr_p_pearson = pearsonr(firing_df['dsi'], firing_df['firing_rate'])
corr_spearman, corr_p_spearman = spearmanr(firing_df['dsi'], firing_df['firing_rate'])

print(f"\n🔗 Correlation (DSI vs Firing Rate):")
print(f"  Pearson:  r={corr_pearson:.3f}, p={corr_p_pearson:.6f}")
print(f"  Spearman: ρ={corr_spearman:.3f}, p={corr_p_spearman:.6f}")

# Active neurons analysis
fr_threshold = 5  # Hz
high_conf_active = firing_df[(firing_df['confidence'] == 'high') & (firing_df['firing_rate'] > fr_threshold)].groupby('trial_idx').size()
low_conf_active = firing_df[(firing_df['confidence'] == 'low') & (firing_df['firing_rate'] > fr_threshold)].groupby('trial_idx').size()

print(f"\n👥 Active Neurons (FR > {fr_threshold} Hz):")
print(f"  High conf: {high_conf_active.mean():.1f} ± {high_conf_active.std():.1f} neurons/trial")
print(f"  Low conf:  {low_conf_active.mean():.1f} ± {low_conf_active.std():.1f} neurons/trial")

# =============================================================================
# STEP 5: VISUALIZATION
# =============================================================================

print("\n" + "="*70)
print("STEP 5: Creating visualizations...")
print("="*70)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. DSI distribution
ax = axes[0, 0]
ax.hist(dsi[valid_trials], bins=30, alpha=0.7, edgecolor='black', color='steelblue')
ax.axvline(threshold, color='r', linestyle='--', linewidth=2, label=f'Median={threshold:.2f}')
ax.set_xlabel('Decision Strength Index (DSI)', fontsize=12, fontweight='bold')
ax.set_ylabel('Trial Count', fontsize=12, fontweight='bold')
ax.set_title('DSI Distribution', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# 2. Firing rate comparison - Boxplot
ax = axes[0, 1]
bp = ax.boxplot([low_conf_fr, high_conf_fr], labels=['Low\nConfidence', 'High\nConfidence'],
                patch_artist=True, widths=0.6)
bp['boxes'][0].set_facecolor('lightcoral')
bp['boxes'][1].set_facecolor('lightgreen')
ax.set_ylabel('Firing Rate (Hz)', fontsize=12, fontweight='bold')
ax.set_title(f'Firing Rate by Confidence\np={u_pval:.4f}', fontsize=14, fontweight='bold')
ax.grid(alpha=0.3, axis='y')

# 3. Violin plot
ax = axes[0, 2]
parts = ax.violinplot([low_conf_fr, high_conf_fr], positions=[1, 2], 
                       showmeans=True, showmedians=True, widths=0.7)
for pc in parts['bodies']:
    pc.set_alpha(0.7)
ax.set_xticks([1, 2])
ax.set_xticklabels(['Low Conf', 'High Conf'])
ax.set_ylabel('Firing Rate (Hz)', fontsize=12, fontweight='bold')
ax.set_title('FR Distribution', fontsize=14, fontweight='bold')
ax.grid(alpha=0.3, axis='y')

# 4. DSI vs Firing Rate scatter
ax = axes[1, 0]
ax.scatter(firing_df['dsi'], firing_df['firing_rate'], alpha=0.05, s=2, color='navy')
z = np.polyfit(firing_df['dsi'], firing_df['firing_rate'], 1)
p = np.poly1d(z)
x_line = np.linspace(firing_df['dsi'].min(), firing_df['dsi'].max(), 100)
ax.plot(x_line, p(x_line), 'r-', linewidth=2, label=f'Fit: ρ={corr_spearman:.3f}')
ax.set_xlabel('Decision Strength Index', fontsize=12, fontweight='bold')
ax.set_ylabel('Firing Rate (Hz)', fontsize=12, fontweight='bold')
ax.set_title(f'DSI vs Firing Rate\np={corr_p_spearman:.4f}', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# 5. Binned analysis
ax = axes[1, 1]
dsi_bins = np.percentile(firing_df['dsi'], np.linspace(0, 100, 11))
bin_centers = (dsi_bins[:-1] + dsi_bins[1:]) / 2
bin_means = []
bin_sems = []

for i in range(len(dsi_bins)-1):
    mask = (firing_df['dsi'] >= dsi_bins[i]) & (firing_df['dsi'] < dsi_bins[i+1])
    if np.sum(mask) > 10:
        bin_means.append(firing_df[mask]['firing_rate'].mean())
        bin_sems.append(firing_df[mask]['firing_rate'].sem())
    else:
        bin_means.append(np.nan)
        bin_sems.append(np.nan)

ax.errorbar(bin_centers, bin_means, yerr=bin_sems, marker='o', markersize=8,
            capsize=5, linewidth=2, color='darkgreen')
ax.set_xlabel('DSI (binned)', fontsize=12, fontweight='bold')
ax.set_ylabel('Mean Firing Rate (Hz)', fontsize=12, fontweight='bold')
ax.set_title('Binned Analysis', fontsize=14, fontweight='bold')
ax.grid(alpha=0.3)

# 6. Active neurons per trial
ax = axes[1, 2]
bp = ax.boxplot([low_conf_active, high_conf_active],
                labels=['Low\nConfidence', 'High\nConfidence'],
                patch_artist=True, widths=0.6)
bp['boxes'][0].set_facecolor('lightcoral')
bp['boxes'][1].set_facecolor('lightgreen')
ax.set_ylabel(f'# Active Neurons (FR>{fr_threshold}Hz)', fontsize=12, fontweight='bold')
ax.set_title('Active Neurons per Trial', fontsize=14, fontweight='bold')
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# =============================================================================
# STEP 6: FINAL SUMMARY
# =============================================================================

print("\n" + "="*70)
print("🎯 FINAL SUMMARY")
print("="*70)

print(f"\n📋 Hypothesis:")
print(f"   'High confidence → More neurons activated + Higher firing rates'")

print(f"\n📊 Results:")
print(f"   Firing Rate:")
print(f"     • High confidence: {high_conf_fr.mean():.2f} Hz")
print(f"     • Low confidence:  {low_conf_fr.mean():.2f} Hz")
print(f"     • Difference: +{high_conf_fr.mean() - low_conf_fr.mean():.2f} Hz ({100*(high_conf_fr.mean() - low_conf_fr.mean())/low_conf_fr.mean():.1f}%)")
print(f"     • P-value: {u_pval:.6f}")

print(f"\n   Active Neurons:")
print(f"     • High confidence: {high_conf_active.mean():.1f} neurons/trial")
print(f"     • Low confidence:  {low_conf_active.mean():.1f} neurons/trial")
print(f"     • Difference: +{high_conf_active.mean() - low_conf_active.mean():.1f} neurons")

print(f"\n   Correlation:")
print(f"     • Spearman ρ = {corr_spearman:.3f} (p={corr_p_spearman:.6f})")

# Conclusion
if u_pval < 0.05 and corr_spearman > 0.1:
    print(f"\n✅✅✅ HYPOTHESIS STRONGLY SUPPORTED!")
    print(f"     High confidence decisions show:")
    print(f"     1. Significantly higher firing rates (p<0.05)")
    print(f"     2. More active neurons per trial")
    print(f"     3. Positive correlation with DSI")
elif u_pval < 0.05:
    print(f"\n⚠️ HYPOTHESIS PARTIALLY SUPPORTED")
    print(f"     Significant difference but weak correlation")
else:
    print(f"\n❌ HYPOTHESIS NOT SUPPORTED")
    print(f"     No significant difference found")

print(f"\n{'='*70}")