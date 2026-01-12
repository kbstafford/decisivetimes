"""
IBL Calcium Imaging Timescale-Confidence Analysis Pipeline
==========================================================

Complete pipeline for analyzing IBL widefield calcium imaging data using
à trous wavelet transform to link neural timescales to confidence-like behavior.

"""

import numpy as np
import matplotlib.pyplot as plt
import pywt
#import wfield
from scipy.signal import find_peaks
from scipy.stats import ttest_ind, mannwhitneyu
import pandas as pd
from pathlib import Path
import warnings
import re

warnings.filterwarnings('ignore')


# ============================================================================
# PART 1: DATA LOADING
# ============================================================================

class IBLCalciumLoader:
    """Load and preprocess IBL widefield calcium imaging data"""

    def __init__(self, one_api=None):
        """
        Initialize loader

        Parameters:
        -----------
        one_api : ONE object
            IBL ONE API instance (if None, uses mock data for demo)
        """
        self.one = one_api

    def load_session(self, eid):

        if self.one is None:
            print("No ONE API provided, generating mock data...")
            U, SVT, trials, framerate = self._generate_mock_data()
            return {'U': U, 'SVT': SVT, 'trials': trials, 'framerate': framerate}

        dsets = self.one.list_datasets(eid)  # list[str]

        # --------- 1) Load trials (minimal + robust) ----------
        trials = self.one.load_object(eid, 'trials', collection='alf', namespace='ibl')

        # --------- 2) Find widefield U + SVT datasets ----------
        # Avoid false positives like imaging.imagingLightSource.npy
        def pick_first(patterns, exclude_patterns=None):
            exclude_patterns = exclude_patterns or []
            for pat in patterns:
                for ds in dsets:
                    if any(re.search(ep, ds, flags=re.IGNORECASE) for ep in exclude_patterns):
                        continue
                    if re.search(pat, ds, flags=re.IGNORECASE):
                        return ds
            return None

        # Prefer exact canonical filenames (what you actually want)
        U_path = pick_first(
            patterns=[
                r'alf[/\\]widefield[/\\]widefieldU\.images\.npy$',
                r'widefieldU\.images\.npy$',
                r'alf[/\\]widefield[/\\].*widefieldU.*\.npy$',
                r'.*widefieldU.*images.*\.npy$'
            ],
            exclude_patterns=[
                r'imaging\.imagingLightSource\.npy$',
                r'imaging\.'  # exclude other imaging.* metadata
            ]
        )

        SVT_path = pick_first(
            patterns=[
                r'alf[/\\]widefield[/\\]widefieldSVT\.haemoCorrected\.npy$',
                r'widefieldSVT\.haemoCorrected\.npy$',
                r'alf[/\\]widefield[/\\].*widefieldSVT.*haem.*\.npy$',
                r'.*widefieldSVT.*haem.*\.npy$',
                r'.*widefieldSVT.*\.npy$'
            ]
        )

        if SVT_path is None:
            raise FileNotFoundError(
                "No widefield SVT dataset found for this eid. "
                "Print filtered datasets and pick a session that has widefield."
            )

        # --------- 3) Load only what you need ----------
        SVT = self.one.load_dataset(eid, SVT_path)

        # SVT is sometimes stored as (n_components, n_frames); transpose to (n_frames, n_components)
        if SVT.ndim == 2 and SVT.shape[0] < SVT.shape[1]:
            # In IBL widefield, K (~200) < T (~80k), so this is the transposed case
            SVT = SVT.T

        U = None
        if U_path is not None:
            U = self.one.load_dataset(eid, U_path)

        # --------- 4) Determine framerate (prefer actual timestamps if present) ----------
        ts_path = pick_first(
            patterns=[
                r'alf[/\\]widefield[/\\]widefield\.timestamps\.npy$',
                r'widefield\.timestamps\.npy$',
                r'widefield.*timestamps.*\.npy$',
                r'.*wf.*timestamps.*\.npy$'
            ]
        )

        if ts_path is not None:
            ts = self.one.load_dataset(eid, ts_path)
            dt = np.median(np.diff(ts))
            framerate = float(1.0 / dt)
        else:
            framerate = 60.0  # fallback

        return {
            'U': U,
            'SVT': SVT,
            'trials': trials,
            'framerate': framerate,
            'paths': {'U': U_path, 'SVT': SVT_path, 'ts': ts_path}
        }

    def _generate_mock_data(self):
        """Generate mock IBL-like data for testing"""
        # Mock spatial components (128x128 pixels, 200 components)
        U = np.random.randn(128, 128, 200) * 0.1

        # Mock temporal components (10000 frames, 200 components)
        # Simulate calcium-like dynamics
        n_frames = 10000
        n_components = 200
        SVT = np.zeros((n_frames, n_components))

        for i in range(n_components):
            # Add some slow drifts and transients
            drift = np.cumsum(np.random.randn(n_frames) * 0.01)
            transients = self._simulate_calcium_transients(n_frames, n_events=20)
            SVT[:, i] = drift + transients

        # Mock trials data
        n_trials = 200
        trials = {
            'contrastLeft': np.random.choice([0, 0.06, 0.12, 0.25, 0.5, 1.0], n_trials),
            'contrastRight': np.zeros(n_trials),
            'choice': np.random.choice([-1, 1], n_trials),
            'stimOn_times': np.linspace(10, 320, n_trials),
            'response_times': np.linspace(10.5, 320.5, n_trials),
            'feedback_times': np.linspace(11, 321, n_trials),
            'feedbackType': np.random.choice([-1, 1], n_trials),
        }

        framerate = 30  # Hz

        return U, SVT, trials, framerate

    def _simulate_calcium_transients(self, n_frames, n_events=20):
        """Simulate calcium transient events"""
        signal = np.zeros(n_frames)
        event_times = np.random.randint(100, n_frames - 100, n_events)

        for t in event_times:
            # Calcium transient: fast rise, slow decay
            duration = np.random.randint(20, 60)
            rise = np.linspace(0, 1, duration // 3)
            decay = np.exp(-np.linspace(0, 3, duration * 2 // 3))
            transient = np.concatenate([rise, decay])

            end_idx = min(t + len(transient), n_frames)
            signal[t:end_idx] += transient[:end_idx - t] * np.random.uniform(0.5, 2.0)

        return signal


# ============================================================================
# PART 2: À TROUS WAVELET TRANSFORM
# ============================================================================

class ATrousAnalyzer:
    """À trous wavelet transform for calcium imaging analysis"""

    def __init__(self, wavelet='bior3.1', level=5):
        """
        Initialize à trous analyzer

        Parameters:
        -----------
        wavelet : str
            Wavelet type (bior3.1 approximates cubic B-spline)
        level : int
            Number of decomposition levels
        """
        self.wavelet = wavelet
        self.level = level

    def denoise(self, signal, threshold_factor=1.5):
        """
        Denoise signal using à trous with hard thresholding

        Parameters:
        -----------
        signal : array
            Input signal
        threshold_factor : float
            Factor for threshold (default 3 for 3-sigma)

        Returns:
        --------
        denoised : array
            Denoised signal
        """
        # Pad to power of 2
        signal_padded, pad_len = self._pad_signal(signal)

        # SWT decomposition (à trous)
        coeffs = pywt.swt(signal_padded, self.wavelet, level=self.level)

        # Estimate noise from finest scale detail coefficients
        sigma_0 = np.median(np.abs(coeffs[0][1])) / 0.6745  # MAD estimator

        # Hard threshold at each level
        denoised_coeffs = []
        for i, (cA, cD) in enumerate(coeffs):
            sigma_j = sigma_0 * np.sqrt(i + 1)
            threshold = threshold_factor * sigma_j
            cD_thresh = pywt.threshold(cD, threshold, mode='hard')
            denoised_coeffs.append((cA, cD_thresh))

        # Reconstruct
        denoised = pywt.iswt(denoised_coeffs, self.wavelet)

        return denoised[:len(signal)]

    def decompose(self, signal):
        """
        Decompose signal into multiple scales

        Returns:
        --------
        coeffs : list of tuples
            (approximation, detail) at each level
        """
        signal_padded, _ = self._pad_signal(signal)
        coeffs = pywt.swt(signal_padded, self.wavelet, level=self.level)

        # Trim to original length
        coeffs_trimmed = []
        for cA, cD in coeffs:
            coeffs_trimmed.append((cA[:len(signal)], cD[:len(signal)]))

        return coeffs_trimmed

    def detect_events(self, signal, min_scale_presence=2):
        """
        Detect events that persist across multiple scales

        Parameters:
        -----------
        signal : array
            Input signal
        min_scale_presence : int
            Minimum number of scales event must appear in

        Returns:
        --------
        events : array
            Event indices
        event_properties : dict
            Properties of each event
        """
        # Denoise first
        denoised = self.denoise(signal)

        # Get multi-scale decomposition
        coeffs = self.decompose(denoised)

        # Detect peaks at each scale
        events_by_scale = []
        for i, (cA, cD) in enumerate(coeffs):
            threshold = np.std(cD) * 3
            peaks, properties = find_peaks(np.abs(cD),
                                           height=threshold,
                                           distance=5,
                                           width=1)
            events_by_scale.append(peaks)

        # Find events present across multiple scales
        if len(events_by_scale[0]) == 0:
            return np.array([]), {}

        confirmed_events = []
        for event in events_by_scale[0]:
            count = sum([1 for scale_events in events_by_scale[1:]
                         if any(abs(event - e) < 10 for e in scale_events)])
            if count >= min_scale_presence:
                confirmed_events.append(event)

        # Get event properties
        confirmed_events = np.array(confirmed_events)
        event_properties = self._compute_event_properties(denoised, confirmed_events)

        return confirmed_events, event_properties

    def _compute_event_properties(self, signal, event_indices):
        """Compute properties of detected events"""
        properties = {
            'peak_amplitudes': [],
            'durations': [],
            'rise_times': [],
            'decay_times': []
        }

        for idx in event_indices:
            # Find event boundaries
            threshold = signal[idx] * 0.5  # Half-max

            # Search backwards for start
            start = idx
            while start > 0 and signal[start] > threshold:
                start -= 1

            # Search forwards for end
            end = idx
            while end < len(signal) - 1 and signal[end] > threshold:
                end += 1

            properties['peak_amplitudes'].append(signal[idx])
            properties['durations'].append(end - start)
            properties['rise_times'].append(idx - start)
            properties['decay_times'].append(end - idx)

        return {k: np.array(v) for k, v in properties.items()}

    def compute_dominant_scale(self, signal):
        """
        Compute which scale has dominant power

        Returns:
        --------
        dominant_scale : int
            Scale index with maximum power
        scale_powers : array
            Power at each scale
        """
        coeffs = self.decompose(signal)
        scale_powers = np.array([np.sum(cD ** 2) for cA, cD in coeffs])
        dominant_scale = np.argmax(scale_powers)

        return dominant_scale, scale_powers

    def _pad_signal(self, signal):
        """Pad signal to power of 2"""
        pad_len = 2 ** self.level - (len(signal) % 2 ** self.level)
        if pad_len != 2 ** self.level:
            signal_padded = np.pad(signal, (0, pad_len), mode='reflect')
        else:
            signal_padded = signal
            pad_len = 0
        return signal_padded, pad_len


# ============================================================================
# PART 3: CONFIDENCE-TIMESCALE LINKING
# ============================================================================

class ConfidenceTimescaleAnalyzer:
    """Link neural timescales to confidence-like behavior"""

    def __init__(self, framerate=30):
        """
        Initialize analyzer

        Parameters:
        -----------
        framerate : float
            Imaging framerate in Hz
        """
        self.framerate = framerate
        self.atrous = ATrousAnalyzer()

    def extract_roi_timeseries(self, U, SVT, roi_coords=None):
        """
        Extract ROI time series from SVD components

        Parameters:
        -----------
        U : array
            Spatial components (height x width x n_components)
        SVT : array
            Temporal components (n_frames x n_components)
        roi_coords : tuple or None
            (x_center, y_center, radius) or None for mock ROI

        Returns:
        --------
        timeseries : array
            ROI-averaged time series
        """
        if roi_coords is None:
            # Use center region as ROI
            h, w = U.shape[:2]
            roi_coords = (w // 2, h // 2, 20)

        x, y, r = roi_coords

        # Create circular mask
        yy, xx = np.ogrid[:U.shape[0], :U.shape[1]]
        mask = (xx - x) ** 2 + (yy - y) ** 2 <= r ** 2

        # Weight spatial components by mask
        roi_weights = np.zeros(U.shape[2])
        for i in range(U.shape[2]):
            roi_weights[i] = np.sum(U[:, :, i] * mask)

        # Weighted sum of temporal components
        timeseries = SVT @ roi_weights

        return timeseries

    def classify_trials_by_confidence(self, trials, method='contrast'):
        """
        Robust confidence proxy that works across IBL task variants.
        Returns boolean masks (high_conf, low_conf) of same length as trials.
        """
        n = len(trials['intervals']) if 'intervals' in trials else len(next(iter(trials.values())))

        # Helper: safely get array
        def get(name):
            x = trials.get(name, None)
            if x is None:
                return None
            x = np.asarray(x)
            return x

        if method == 'contrast':
            cl = get('contrastLeft')
            cr = get('contrastRight')

            # Some tasks store NaNs; treat missing side as 0
            if cl is None and cr is None:
                # Fall back to accuracy if contrast not present
                method = 'accuracy'
            else:
                if cl is None: cl = np.zeros(n)
                if cr is None: cr = np.zeros(n)

                contrast = np.nanmax(np.c_[cl, cr], axis=1)

                # More forgiving bins (quantiles) so you ALWAYS get groups
                # High = top 30%, Low = bottom 30%
                lo = np.nanquantile(contrast, 0.30)
                hi = np.nanquantile(contrast, 0.70)

                high_conf = contrast >= hi
                low_conf = contrast <= lo

                # If still empty (e.g., all contrasts same), fall back to accuracy
                if high_conf.sum() == 0 or low_conf.sum() == 0:
                    method = 'accuracy'

        if method == 'accuracy':
            fb = get('feedbackType')
            if fb is None:
                # Absolute fallback: split trials by reaction time (fast=high conf)
                rt = get('response_times')
                st = get('stimOn_times') or get('stimOnTrigger_times') or get('goCueTrigger_times')
                if rt is None or st is None:
                    # Give up: return all False
                    return np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)

                rtime = rt - st
                lo = np.nanquantile(rtime, 0.30)
                hi = np.nanquantile(rtime, 0.70)
                high_conf = rtime <= lo  # faster
                low_conf = rtime >= hi  # slower
            else:
                fb = np.asarray(fb)
                high_conf = fb == 1
                low_conf = fb == -1

        return high_conf, low_conf

    def align_trials(self, timeseries, trials, framerate,
                     align_to='stimOn_times', window=(-0.5, 2.0)):
        """
        Align time series to trial events

        Parameters:
        -----------
        timeseries : array
            Continuous time series
        trials : dict
            Trial data with event times
        framerate : float
            Imaging framerate
        align_to : str
            Event to align to
        window : tuple
            (start, end) time window in seconds

        Returns:
        --------
        aligned_trials : list of arrays
            List of trial segments
        """
        event_times = trials[align_to]
        window_frames = (int(window[0] * framerate),
                         int(window[1] * framerate))

        aligned_trials = []
        for t in event_times:
            frame_idx = int(t * framerate)
            start_idx = frame_idx + window_frames[0]
            end_idx = frame_idx + window_frames[1]

            if start_idx >= 0 and end_idx < len(timeseries):
                aligned_trials.append(timeseries[start_idx:end_idx])

        return aligned_trials

    def compute_timescale_metrics(self, trial_segments):
        """
        Compute timescale metrics for trial segments

        Parameters:
        -----------
        trial_segments : list of arrays
            Trial-aligned segments

        Returns:
        --------
        metrics : dict
            Timescale metrics for each trial
        """
        metrics = {
            'event_durations': [],
            'rise_times': [],
            'decay_times': [],
            'dominant_scales': [],
            'peak_amplitudes': [],
            'total_power': []
        }

        for segment in trial_segments:
            segment = (segment - np.mean(segment)) / (np.std(segment) + 1e-8)

            # Detect events
            events, properties = self.atrous.detect_events(segment)

            if len(events) > 0:
                # Use first detected event
                metrics['event_durations'].append(
                    properties['durations'][0] / self.framerate
                )
                metrics['rise_times'].append(
                    properties['rise_times'][0] / self.framerate
                )
                metrics['decay_times'].append(
                    properties['decay_times'][0] / self.framerate
                )
                metrics['peak_amplitudes'].append(
                    properties['peak_amplitudes'][0]
                )
            else:
                # No event detected
                metrics['event_durations'].append(np.nan)
                metrics['rise_times'].append(np.nan)
                metrics['decay_times'].append(np.nan)
                metrics['peak_amplitudes'].append(np.nan)

            # Dominant scale
            dom_scale, powers = self.atrous.compute_dominant_scale(segment)
            metrics['dominant_scales'].append(dom_scale)
            metrics['total_power'].append(np.sum(powers))

        return {k: np.array(v) for k, v in metrics.items()}

    def compare_confidence_timescales(self, high_conf_metrics, low_conf_metrics):
        """
        Statistical comparison of timescales between confidence levels

        Returns:
        --------
        results : dict
            Statistical test results
        """
        results = {}

        for metric_name in high_conf_metrics.keys():
            high = high_conf_metrics[metric_name]
            low = low_conf_metrics[metric_name]

            # Remove NaNs
            high_clean = high[~np.isnan(high)]
            low_clean = low[~np.isnan(low)]

            if len(high_clean) > 0 and len(low_clean) > 0:
                # Mann-Whitney U test (non-parametric)
                stat, pval = mannwhitneyu(high_clean, low_clean,
                                          alternative='two-sided')

                results[metric_name] = {
                    'high_mean': np.mean(high_clean),
                    'high_std': np.std(high_clean),
                    'low_mean': np.mean(low_clean),
                    'low_std': np.std(low_clean),
                    'statistic': stat,
                    'p_value': pval,
                    'effect_size': (np.mean(high_clean) - np.mean(low_clean)) /
                                   np.sqrt((np.std(high_clean) ** 2 + np.std(low_clean) ** 2) / 2)
                }

        return results


# ============================================================================
# PART 4: VISUALIZATION
# ============================================================================

class TimescaleVisualizer:
    """Visualization tools for timescale-confidence analysis"""

    @staticmethod
    def plot_wavelet_decomposition(signal, atrous, framerate=30):
        """Plot multi-scale wavelet decomposition"""
        coeffs = atrous.decompose(signal)

        fig, axes = plt.subplots(len(coeffs) + 1, 1, figsize=(14, 10))

        time = np.arange(len(signal)) / framerate

        # Original signal
        axes[0].plot(time, signal, 'k', linewidth=1)
        axes[0].set_title('Original Signal', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Amplitude')
        axes[0].grid(alpha=0.3)

        # Each scale
        for i, (cA, cD) in enumerate(coeffs):
            timescale_ms = (2 ** (i + 1) / framerate) * 1000
            axes[i + 1].plot(time, cD, linewidth=1)
            axes[i + 1].set_title(f'Scale {i + 1} (Timescale ≈ {timescale_ms:.0f} ms)',
                                  fontsize=11)
            axes[i + 1].set_ylabel('Detail Coeff')
            axes[i + 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
            axes[i + 1].grid(alpha=0.3)

        axes[-1].set_xlabel('Time (s)', fontsize=11)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_confidence_comparison(high_metrics, low_metrics, stats):
        """Plot comparison of timescales between confidence levels"""
        metrics_to_plot = ['event_durations', 'rise_times', 'decay_times',
                           'dominant_scales']
        labels = ['Event Duration (s)', 'Rise Time (s)', 'Decay Time (s)',
                  'Dominant Scale']

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for i, (metric, label) in enumerate(zip(metrics_to_plot, labels)):
            # Skip if metric not available in either dict
            if metric not in high_metrics or metric not in low_metrics:
                axes[i].text(0.5, 0.5, "No data",
                             ha="center", va="center",
                             transform=axes[i].transAxes)
                axes[i].set_title(label, fontsize=10)
                axes[i].set_xticks([])
                continue

            # Convert to arrays and drop NaNs
            high_raw = np.asarray(high_metrics[metric])
            low_raw = np.asarray(low_metrics[metric])

            high = high_raw[~np.isnan(high_raw)]
            low = low_raw[~np.isnan(low_raw)]

            # Debug (optional)
            # print(f"[DEBUG] {metric}: high={high.size}, low={low.size}")

            datasets = []
            positions = []
            tick_labels = []

            if high.size > 0:
                datasets.append(high)
                positions.append(1)
                tick_labels.append('High\nConfidence')

            if low.size > 0:
                datasets.append(low)
                positions.append(2)
                tick_labels.append('Low\nConfidence')

            # If no valid data for this metric, just mark it and move on
            if len(datasets) == 0:
                axes[i].text(0.5, 0.5, "No valid data",
                             ha="center", va="center",
                             transform=axes[i].transAxes)
                axes[i].set_title(label, fontsize=10)
                axes[i].set_xticks([])
                continue

            # Violin plots
            parts = axes[i].violinplot([high, low], positions=[1, 2],
                                       showmeans=True, showmedians=True)

            # Color coding
            for pc in parts['bodies']:
                pc.set_facecolor('#3498db')
                pc.set_alpha(0.7)

            axes[i].set_xticks([1, 2])
            axes[i].set_xticklabels(['High\nConfidence', 'Low\nConfidence'])
            axes[i].set_ylabel(label, fontsize=11)
            axes[i].grid(axis='y', alpha=0.3)

            # Add statistics
            if metric in stats:
                pval = stats[metric]['p_value']
                effect = stats[metric]['effect_size']
                sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
                axes[i].set_title(f'{label}\np={pval:.4f} ({sig}), d={effect:.2f}',
                                  fontsize=10)

        plt.suptitle('Timescale Comparison: High vs Low Confidence Trials',
                     fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_trial_examples(high_trial, low_trial, framerate=30):
        """Plot example trials for high vs low confidence"""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        time_high = np.arange(len(high_trial)) / framerate
        time_low = np.arange(len(low_trial)) / framerate

        axes[0].plot(time_high, high_trial, 'b', linewidth=2)
        axes[0].set_title('High Confidence Trial Example',
                          fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Calcium Signal (a.u.)')
        axes[0].grid(alpha=0.3)
        axes[0].axvline(x=0, color='r', linestyle='--',
                        label='Stimulus Onset', linewidth=2)
        axes[0].legend()

        axes[1].plot(time_low, low_trial, 'orange', linewidth=2)
        axes[1].set_title('Low Confidence Trial Example',
                          fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Calcium Signal (a.u.)')
        axes[1].set_xlabel('Time (s)')
        axes[1].grid(alpha=0.3)
        axes[1].axvline(x=0, color='r', linestyle='--',
                        label='Stimulus Onset', linewidth=2)
        axes[1].legend()

        plt.tight_layout()
        return fig


# ============================================================================
# PART 5: MAIN PIPELINE
# ============================================================================

class IBLTimescalePipeline:
    """Complete pipeline for IBL calcium imaging timescale analysis"""

    def __init__(self, one_api=None):
        """
        Initialize pipeline

        Parameters:
        -----------
        one_api : ONE object or None
            IBL ONE API (None uses mock data)
        """
        self.loader = IBLCalciumLoader(one_api)
        self.analyzer = None  # Will be initialized with framerate
        self.visualizer = TimescaleVisualizer()

    def run(self, eid=None, roi_coords=None, save_dir='./results'):
        """
        Run complete analysis pipeline

        Parameters:
        -----------
        eid : str or None
            Session ID (None uses mock data)
        roi_coords : tuple or None
            ROI coordinates (x, y, radius)
        save_dir : str
            Directory to save results

        Returns:
        --------
        results : dict
            Complete analysis results
        """
        print("=" * 70)
        print("IBL CALCIUM IMAGING TIMESCALE-CONFIDENCE ANALYSIS PIPELINE")
        print("=" * 70)

        # Step 1: Load data
        print("\n[1/6] Loading data...")
        data = self.loader.load_session(eid)
        framerate = data['framerate']
        self.analyzer = ConfidenceTimescaleAnalyzer(framerate)
        print(f"✓ Loaded session with {data['SVT'].shape[0]} frames at {framerate} Hz")

        # Step 2: Extract ROI
        print("\n[2/6] Extracting ROI time series...")
        timeseries = self.analyzer.extract_roi_timeseries(
            data['U'], data['SVT'], roi_coords
        )
        print(f"✓ Extracted ROI time series (length: {len(timeseries)} frames)")

        # Step 3: Classify trials
        print("\n[3/6] Classifying trials by confidence...")
        high_conf, low_conf = self.analyzer.classify_trials_by_confidence(
            data['trials'], method='contrast'
        )
        print(f"✓ High confidence trials: {np.sum(high_conf)}")
        print(f"✓ Low confidence trials: {np.sum(low_conf)}")

        # Step 4: Align trials
        print("\n[4/6] Aligning trials to stimulus onset...")
        all_trials = self.analyzer.align_trials(
            timeseries, data['trials'], framerate
        )
        high_trials = [all_trials[i] for i in np.where(high_conf)[0]
                       if i < len(all_trials)]
        low_trials = [all_trials[i] for i in np.where(low_conf)[0]
                      if i < len(all_trials)]
        print(f"✓ Aligned {len(high_trials)} high confidence trials")
        print(f"✓ Aligned {len(low_trials)} low confidence trials")

        # Step 5: Compute timescale metrics
        print("\n[5/6] Computing timescale metrics...")
        high_metrics = self.analyzer.compute_timescale_metrics(high_trials)
        low_metrics = self.analyzer.compute_timescale_metrics(low_trials)
        print("✓ Computed metrics for all trials")

        # Step 6: Statistical comparison
        print("\n[6/6] Statistical comparison...")
        stats = self.analyzer.compare_confidence_timescales(
            high_metrics, low_metrics
        )

        print("\n" + "=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)
        for metric, result in stats.items():
            print(f"\n{metric}:")
            print(f"  High confidence: {result['high_mean']:.4f} ± {result['high_std']:.4f}")
            print(f"  Low confidence:  {result['low_mean']:.4f} ± {result['low_std']:.4f}")
            print(f"  p-value: {result['p_value']:.4f}")
            print(f"  Effect size (Cohen's d): {result['effect_size']:.3f}")

        # Visualization
        print("\n" + "=" * 70)
        print("GENERATING VISUALIZATIONS")
        print("=" * 70)

        # Create save directory
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        # Plot 1: Wavelet decomposition example
        print("\n[Viz 1/3] Wavelet decomposition...")
        fig1 = self.visualizer.plot_wavelet_decomposition(
            high_trials[0], self.analyzer.atrous, framerate
        )
        fig1.savefig(f'{save_dir}/wavelet_decomposition.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_dir}/wavelet_decomposition.png")

        # Plot 2: Confidence comparison
        print("\n[Viz 2/3] Confidence comparison...")
        fig2 = self.visualizer.plot_confidence_comparison(
            high_metrics, low_metrics, stats
        )
        fig2.savefig(f'{save_dir}/confidence_comparison.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_dir}/confidence_comparison.png")

        # Plot 3: Trial examples
        print("\n[Viz 3/3] Example trials...")
        fig3 = self.visualizer.plot_trial_examples(
            high_trials[0], low_trials[0], framerate
        )
        fig3.savefig(f'{save_dir}/trial_examples.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_dir}/trial_examples.png")

        print("\n" + "=" * 70)
        print("PIPELINE COMPLETE!")
        print("=" * 70)

        return {
            'high_metrics': high_metrics,
            'low_metrics': low_metrics,
            'statistics': stats,
            'data': data,
            'timeseries': timeseries
        }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of the pipeline

    For real IBL data:
        from one.api import ONE
        one = ONE()
        pipeline = IBLTimescalePipeline(one_api=one)
        results = pipeline.run(eid='your-session-id')

    For demo with mock data:
        pipeline = IBLTimescalePipeline()
        results = pipeline.run()
    """

    # Run with mock data for demonstration
   # pipeline = IBLTimescalePipeline(one_api=None)
   # results = pipeline.run(save_dir='./ibl_timescale_results')

    from one.api import ONE

    ONE.setup(base_url='https://openalyx.internationalbrainlab.org', silent=True)
    one = ONE(base_url='https://openalyx.internationalbrainlab.org',
              password='international',
              silent=True)

    print("offline:", one.offline)

    one = ONE()
    sessions = one.search(datasets='widefieldU.images.npy')
    print(f'{len(sessions)} sessions with widefield data found')

    pipeline = IBLTimescalePipeline(one_api=one)
    results = pipeline.run(eid=sessions[12])

    # Access results
    print("\n\nAccessing results:")
    print("- results['high_metrics']: metrics for high confidence trials")
    print("- results['low_metrics']: metrics for low confidence trials")
    print("- results['statistics']: statistical comparison")
    print("- results['data']: raw IBL data")
    print("- results['timeseries']: ROI time series")

    # Example: Access specific metrics
    print("\nExample - Event durations:")
    high_durations = results['high_metrics']['event_durations']
    low_durations = results['low_metrics']['event_durations']
    print(f"  High confidence mean: {np.nanmean(high_durations):.3f} s")
    print(f"  Low confidence mean: {np.nanmean(low_durations):.3f} s")