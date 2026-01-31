import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from brainbox.io.one import SpikeSortingLoader

def analyze_decision_neuron_lda(eid, brain_region=None, pre_time=0.2, post_time=0, bin_size=0.05, acc_threshold=0.65):
    """
    Identifies decision neurons using LDA on movement-aligned, 0% contrast trials.

    Args:
        eid (str): Session UUID
        pre_time (float): Seconds before movement to include (Ramp period)
        post_time (float): Seconds after movement to include
        bin_size (float): Size of time bins for LDA features (e.g., 0.05 = 50ms)
        acc_threshold (float): Accuracy threshold to define 'decision-related' neurons
    """
    print(f"Processing session {eid}...")

    try:
        # --- 1. LOAD DATA ---
        sl = SpikeSortingLoader(eid=eid, one=one)
        spikes, clusters, channels = sl.load_spike_sorting()
        clusters = sl.merge_clusters(spikes, clusters, channels)

        # Load trials and movement times
        trials = one.load_object(eid, 'trials', collection='alf')
        if 'firstMovement_times' not in trials:
            print("  ⚠️ 'firstMovement_times' not found. Approximating with 'response_times'.")
            align_times = trials.response_times
        else:
            align_times = trials.firstMovement_times

        # --- 2. FILTER NEURONS ---
        clusters_df = pd.DataFrame(clusters)
        is_good = clusters_df['label'] == 1

        if brain_region:
            is_in_region = clusters_df['acronym'] == brain_region
            good_clusters = clusters_df[is_good & is_in_region]
            print(f"  -> Filtering for region '{brain_region}'. Found {len(good_clusters)} neurons.")
        else:
            good_clusters = clusters_df[is_good]
            print(f"  -> Using all regions. Found {len(good_clusters)} neurons.")

        if good_clusters.empty:
            print(f"  ❌ No good clusters found. Skipping.")
            return

        # --- 3. TRIAL SELECTION (Low Contrast < 50%) ---
        cLeft = np.nan_to_num(trials.contrastLeft, nan=0.0)
        cRight = np.nan_to_num(trials.contrastRight, nan=0.0)
        is_low_contrast = (cLeft < 0.5) & (cRight < 0.5)

        valid_mask = (
            ~np.isnan(trials.choice) & 
            ~np.isnan(align_times) & 
            is_low_contrast
        )

        choice = trials.choice[valid_mask]
        events = align_times[valid_mask]
        n_trials = len(choice)

        if n_trials < 10:
            print("  ❌ Not enough low contrast trials to run LDA. Skipping.")
            return

        # --- 4. PREPARE LDA FEATURES (Population Loop) ---
        bins = np.arange(-pre_time, post_time + bin_size, bin_size)
        n_bins = len(bins) - 1
        
        all_accuracies = []
        cluster_ids = []

        print(f"  -> Running LDA on {len(good_clusters)} neurons...")

        for cluster_id in good_clusters.index.values:
            n_spikes = spikes.times[spikes.clusters == cluster_id]
            X_neuron = np.zeros((n_trials, n_bins))

            for t_idx, t_ref in enumerate(events):
                b_edges = t_ref + bins
                idxs = np.searchsorted(n_spikes, b_edges)
                counts = np.diff(idxs)
                X_neuron[t_idx, :] = counts / bin_size 

            # LDA Classification with Cross-Validation
            lda = LinearDiscriminantAnalysis()
            cv_folds = min(5, n_trials // 2)
            if cv_folds < 2: 
                continue

            scores = cross_val_score(lda, X_neuron, choice, cv=cv_folds)
            all_accuracies.append(np.mean(scores))
            cluster_ids.append(cluster_id)

        # --- 5. REPORT & VISUALIZE POPULATION ---
        if not all_accuracies:
            print("  ❌ No neurons were successfully analyzed.")
            return

        all_accuracies = np.array(all_accuracies)

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10, 6))

        # Histogram of all neuron accuracies
        ax.hist(all_accuracies, bins=20, color='#3498db', edgecolor='white', alpha=0.8)

        # Add visual cues
        ax.axvline(0.5, color='#e74c3c', linestyle='--', lw=2, label='Theoretical Chance (50%)')
        ax.axvline(acc_threshold, color='#2ecc71', linestyle=':', lw=2, 
                   label=f'Decision Threshold (>{acc_threshold*100:.0f}%)')

        ax.set_title(f'Population Decision Informativeness\nSession: {eid}', fontsize=15)
        ax.set_xlabel('LDA Decoding Accuracy (Choice)', fontsize=12)
        ax.set_ylabel('Number of Neurons', fontsize=12)
        ax.set_xlim([0.3, 1.0]) 
        ax.legend()

        plt.tight_layout()
        plt.show()

        # Print Summary
        num_sig = np.sum(all_accuracies > acc_threshold)
        print(f"\n★ Analysis Complete ★")
        print(f"Total neurons analyzed: {len(all_accuracies)}")
        print(f"Neurons > {acc_threshold*100:.0f}% accuracy: {num_sig} ({num_sig/len(all_accuracies)*100:.1f}%)")
        print(f"Peak neuron accuracy: {np.max(all_accuracies)*100:.1f}%")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

# ==========================================
# RUN THE ANALYSIS
# ==========================================
target_eid = 'c557324b-b95d-414c-888f-6ee1329a2329' 
analyze_decision_neuron_lda(target_eid, acc_threshold=0.65)
