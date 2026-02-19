"""
CTG (Cross-Temporal Generalization) Stability Analysis Pipeline

This module implements an optimized pipeline for analyzing neural stability
across time using cross-temporal generalization approaches.

Optimizations included:
1. Delete ONE object after path lookup
2. Direct np.load with rglob, no SpikeSortingLoader
3. float32 spike times for memory efficiency
4. Bin only around events, not whole session
5. Controlled joblib parallelism
6. gc.collect() after each session
7. Revision-safe latest revision picker
"""

import argparse
import gc
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings

import numpy as np
import yaml

# IBL imports
try:
    from one.api import ONE
except ImportError:
    ONE = None
    warnings.warn("ONE API not available. Install with: pip install ONE-api")

try:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import StratifiedKFold
    from joblib import Parallel, delayed
except ImportError:
    warnings.warn("sklearn or joblib not available. Install with: pip install scikit-learn joblib")

# Public IBL OpenAlyx credentials (shared by entire IBL community)
OPENALYX_URL = 'https://openalyx.internationalbrainlab.org'
OPENALYX_PASSWORD = 'international'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ConfigLoader:
    """Load and manage configuration from multiple sources with priority."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize config loader.
        
        Priority order (highest to lowest):
        1. CLI arguments
        2. Environment variables
        3. config.yaml file
        4. config_default.yaml defaults
        """
        self.config_path = config_path
        self.config = self._load_base_config()
    
    def _load_base_config(self) -> Dict[str, Any]:
        """Load base configuration from YAML files."""
        # Start with default config
        default_path = Path(__file__).parent.parent.parent / 'config_default.yaml'
        config = {}
        
        if default_path.exists():
            with open(default_path, 'r') as f:
                config = yaml.safe_load(f) or {}
            logger.info(f"Loaded default config from {default_path}")
        
        # Override with user config if it exists
        if self.config_path and self.config_path.exists():
            with open(self.config_path, 'r') as f:
                user_config = yaml.safe_load(f) or {}
            config = self._merge_configs(config, user_config)
            logger.info(f"Loaded user config from {self.config_path}")
        else:
            # Try default location
            user_config_path = Path(__file__).parent.parent.parent / 'config.yaml'
            if user_config_path.exists():
                with open(user_config_path, 'r') as f:
                    user_config = yaml.safe_load(f) or {}
                config = self._merge_configs(config, user_config)
                logger.info(f"Loaded user config from {user_config_path}")
        
        return config
    
    def _merge_configs(self, base: Dict, override: Dict) -> Dict:
        """Recursively merge override config into base config."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        return result
    
    def apply_env_overrides(self) -> None:
        """Apply environment variable overrides."""
        # Check for cache directory override
        if 'CTG_CACHE_DIR' in os.environ:
            self.config['cache_dir'] = os.environ['CTG_CACHE_DIR']
            logger.info(f"Cache dir overridden by env var: {self.config['cache_dir']}")
    
    def apply_cli_overrides(self, args: argparse.Namespace) -> None:
        """Apply CLI argument overrides."""
        if hasattr(args, 'cache_dir') and args.cache_dir:
            self.config['cache_dir'] = args.cache_dir
            logger.info(f"Cache dir overridden by CLI: {self.config['cache_dir']}")
        
        if hasattr(args, 'eids_file') and args.eids_file:
            self.config['eids'] = self._load_eids_from_file(args.eids_file)
            logger.info(f"EIDs loaded from file: {args.eids_file}")
        
        if hasattr(args, 'output') and args.output:
            self.config['output_path'] = args.output
            logger.info(f"Output path overridden by CLI: {self.config['output_path']}")
    
    def _load_eids_from_file(self, filepath: str) -> List[str]:
        """Load EIDs from a text file (one per line)."""
        eids = []
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    eids.append(line)
        return eids
    
    def get_cache_dir(self) -> Path:
        """Get cache directory, auto-detecting Colab if needed."""
        cache_dir = self.config.get('cache_dir', '~/ibl_data')
        
        # Auto-detect Google Colab
        if self._is_colab() and cache_dir == '~/ibl_data':
            cache_dir = '/content/drive/MyDrive/IBL_Project/Cache'
            logger.info("Detected Google Colab, using Drive cache path")
        
        # Expand home directory
        cache_dir = Path(cache_dir).expanduser()
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        return cache_dir
    
    def _is_colab(self) -> bool:
        """Detect if running in Google Colab."""
        try:
            import google.colab
            return True
        except ImportError:
            return False
    
    def get_output_path(self) -> Path:
        """Get output path, defaulting to cache_dir/results if not specified."""
        if 'output_path' in self.config:
            output_path = Path(self.config['output_path']).expanduser()
        else:
            output_path = self.get_cache_dir() / 'results'
        
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path
    
    def get_eids(self) -> List[str]:
        """Get list of session EIDs to analyze."""
        return self.config.get('eids', [])
    
    def get_analysis_params(self) -> Dict[str, Any]:
        """Get analysis parameters with defaults."""
        defaults = {
            'bin_size': 0.05,
            'window_sec': [-0.2, 0.8],
            'decoder_window_bins': 3,
            'n_folds': 5,
            'n_iterations': 30,
            'n_subsample_neurons': 20,
            'n_subsample_trials': 300,
            'min_neurons': 10,
            'min_trials': 200,
            'n_jobs': 2,
        }
        
        analysis_config = self.config.get('analysis', {})
        params = defaults.copy()
        params.update(analysis_config)
        
        return params


# ============================================================================
# Optimization #7: Revision-safe latest revision picker
# ============================================================================

def _pick_latest_revision(revision_dirs: List[Path]) -> Optional[Path]:
    """
    Pick the latest revision directory from a list of revision paths.
    
    Args:
        revision_dirs: List of paths with revision format like #2024-05-06#
    
    Returns:
        Path to latest revision or None if list is empty
    """
    if not revision_dirs:
        return None
    
    # Sort by the revision date string (format ensures lexicographic = chronological)
    sorted_dirs = sorted(revision_dirs, key=lambda p: p.name, reverse=True)
    return sorted_dirs[0]


# ============================================================================
# Optimization #2: Direct np.load with rglob, no SpikeSortingLoader
# Optimization #3: float32 spike times
# ============================================================================

def load_session_spikes(cache_dir: Path, eid: str, probe: str = 'probe00') -> Optional[Dict[str, np.ndarray]]:
    """
    Load spike data directly using numpy without SpikeSortingLoader.
    
    Args:
        cache_dir: Root cache directory
        eid: Session EID
        probe: Probe name (default: 'probe00')
    
    Returns:
        Dictionary with 'times', 'clusters', 'depths' arrays or None if not found
    """
    logger.info(f"Loading spikes for session {eid}, probe {probe}")
    
    # Find session directory
    session_pattern = f"**/{eid}/**/alf/{probe}/**/spikes.times.npy"
    matches = list(cache_dir.rglob(session_pattern))
    
    if not matches:
        logger.warning(f"No spike data found for {eid}/{probe}")
        return None
    
    # Pick latest revision if multiple exist
    if len(matches) > 1:
        revision_dirs = [m.parent for m in matches]
        latest_dir = _pick_latest_revision(revision_dirs)
        spike_dir = latest_dir
    else:
        spike_dir = matches[0].parent
    
    try:
        # Load spike data (Optimization #3: use float32)
        times = np.load(spike_dir / 'spikes.times.npy').astype(np.float32)
        clusters = np.load(spike_dir / 'spikes.clusters.npy')
        
        # Depths are optional
        depths_file = spike_dir / 'spikes.depths.npy'
        depths = np.load(depths_file) if depths_file.exists() else None
        
        logger.info(f"Loaded {len(times)} spikes from {len(np.unique(clusters))} clusters")
        
        return {
            'times': times,
            'clusters': clusters,
            'depths': depths
        }
    
    except Exception as e:
        logger.error(f"Error loading spikes from {spike_dir}: {e}")
        return None


def load_trials_direct(cache_dir: Path, eid: str) -> Optional[Dict[str, np.ndarray]]:
    """
    Load trial data directly using numpy.
    
    Args:
        cache_dir: Root cache directory
        eid: Session EID
    
    Returns:
        Dictionary with trial data or None if not found
    """
    logger.info(f"Loading trials for session {eid}")
    
    # Find trials directory
    trials_pattern = f"**/{eid}/**/alf/trials.*.npy"
    matches = list(cache_dir.rglob(trials_pattern))
    
    if not matches:
        logger.warning(f"No trial data found for {eid}")
        return None
    
    # Get trials directory
    trials_dir = matches[0].parent
    
    try:
        trials = {}
        
        # Load common trial attributes
        for attr in ['stimOn_times', 'feedback_times', 'choice', 'feedbackType', 
                     'contrastLeft', 'contrastRight', 'goCue_times']:
            filepath = trials_dir / f'trials.{attr}.npy'
            if filepath.exists():
                trials[attr] = np.load(filepath)
        
        logger.info(f"Loaded {len(trials.get('stimOn_times', []))} trials")
        return trials
    
    except Exception as e:
        logger.error(f"Error loading trials from {trials_dir}: {e}")
        return None


# ============================================================================
# Optimization #4: Bin only around events, not whole session
# ============================================================================

def bin_spikes_around_events(
    spike_times: np.ndarray,
    spike_clusters: np.ndarray,
    event_times: np.ndarray,
    window: Tuple[float, float],
    bin_size: float
) -> np.ndarray:
    """
    Bin spikes around specific events (not whole session).
    
    Args:
        spike_times: Array of spike times (float32)
        spike_clusters: Array of cluster IDs for each spike
        event_times: Array of event times to align to
        window: (start, end) relative to event in seconds
        bin_size: Bin width in seconds
    
    Returns:
        Binned spike counts array of shape (n_trials, n_neurons, n_bins)
    """
    n_trials = len(event_times)
    n_neurons = len(np.unique(spike_clusters))
    n_bins = int((window[1] - window[0]) / bin_size)
    
    # Initialize output
    binned = np.zeros((n_trials, n_neurons, n_bins), dtype=np.float32)
    
    # Create neuron ID mapping
    neuron_ids = np.unique(spike_clusters)
    neuron_map = {nid: idx for idx, nid in enumerate(neuron_ids)}
    
    # Bin spikes for each trial
    for trial_idx, event_time in enumerate(event_times):
        trial_start = event_time + window[0]
        trial_end = event_time + window[1]
        
        # Find spikes in this trial window
        trial_mask = (spike_times >= trial_start) & (spike_times < trial_end)
        trial_spikes = spike_times[trial_mask]
        trial_clusters = spike_clusters[trial_mask]
        
        # Compute bin indices
        bin_indices = ((trial_spikes - trial_start) / bin_size).astype(int)
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        # Count spikes per neuron per bin
        for spike_bin, cluster_id in zip(bin_indices, trial_clusters):
            neuron_idx = neuron_map[cluster_id]
            binned[trial_idx, neuron_idx, spike_bin] += 1
    
    return binned


# ============================================================================
# Analysis functions
# ============================================================================

def compute_self_consistency_fast(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_folds: int = 5
) -> float:
    """
    Compute self-consistency (within-time decoding accuracy).
    
    Args:
        X_train: Training data (n_trials, n_features)
        y_train: Training labels (n_trials,)
        n_folds: Number of cross-validation folds
    
    Returns:
        Mean cross-validated accuracy
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    accuracies = []
    
    for train_idx, test_idx in skf.split(X_train, y_train):
        lda = LinearDiscriminantAnalysis()
        lda.fit(X_train[train_idx], y_train[train_idx])
        acc = lda.score(X_train[test_idx], y_train[test_idx])
        accuracies.append(acc)
    
    return np.mean(accuracies)


def analyze_stability(
    binned_spikes: np.ndarray,
    trial_labels: np.ndarray,
    decoder_window_bins: int,
    n_folds: int = 5
) -> Dict[str, np.ndarray]:
    """
    Analyze temporal stability using cross-temporal generalization.
    
    Args:
        binned_spikes: Binned spike data (n_trials, n_neurons, n_bins)
        trial_labels: Labels for each trial (e.g., choice)
        decoder_window_bins: Number of bins to use for decoder window
        n_folds: Number of CV folds
    
    Returns:
        Dictionary with 'ctg_matrix', 'self_consistency' arrays
    """
    n_trials, n_neurons, n_bins = binned_spikes.shape
    n_time_points = n_bins - decoder_window_bins + 1
    
    # Initialize CTG matrix
    ctg_matrix = np.zeros((n_time_points, n_time_points))
    self_consistency = np.zeros(n_time_points)
    
    # Compute for each time point
    for train_t in range(n_time_points):
        # Extract features for this time window
        train_window = slice(train_t, train_t + decoder_window_bins)
        X_train = binned_spikes[:, :, train_window].reshape(n_trials, -1)
        
        # Self-consistency
        self_consistency[train_t] = compute_self_consistency_fast(X_train, trial_labels, n_folds)
        
        # Cross-temporal generalization
        for test_t in range(n_time_points):
            test_window = slice(test_t, test_t + decoder_window_bins)
            X_test = binned_spikes[:, :, test_window].reshape(n_trials, -1)
            
            # Train decoder and test on different time
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            accuracies = []
            
            for train_idx, test_idx in skf.split(X_train, trial_labels):
                lda = LinearDiscriminantAnalysis()
                lda.fit(X_train[train_idx], trial_labels[train_idx])
                acc = lda.score(X_test[test_idx], trial_labels[test_idx])
                accuracies.append(acc)
            
            ctg_matrix[train_t, test_t] = np.mean(accuracies)
    
    return {
        'ctg_matrix': ctg_matrix,
        'self_consistency': self_consistency
    }


# ============================================================================
# Optimization #5: Controlled joblib parallelism
# ============================================================================

def process_subsample(
    binned_spikes: np.ndarray,
    trial_labels: np.ndarray,
    n_subsample_neurons: int,
    n_subsample_trials: int,
    decoder_window_bins: int,
    n_folds: int,
    iteration: int
) -> Dict[str, np.ndarray]:
    """
    Process a single subsample iteration.
    
    Args:
        binned_spikes: Binned spike data
        trial_labels: Trial labels
        n_subsample_neurons: Number of neurons to subsample
        n_subsample_trials: Number of trials to subsample
        decoder_window_bins: Decoder window size
        n_folds: Number of CV folds
        iteration: Iteration number for random seed
    
    Returns:
        Analysis results for this subsample
    """
    np.random.seed(iteration)
    
    n_trials, n_neurons, n_bins = binned_spikes.shape
    
    # Subsample neurons
    neuron_indices = np.random.choice(n_neurons, size=min(n_subsample_neurons, n_neurons), replace=False)
    
    # Subsample trials (stratified by label)
    unique_labels = np.unique(trial_labels)
    trial_indices = []
    trials_per_label = n_subsample_trials // len(unique_labels)
    
    for label in unique_labels:
        label_trials = np.where(trial_labels == label)[0]
        sampled = np.random.choice(label_trials, size=min(trials_per_label, len(label_trials)), replace=False)
        trial_indices.extend(sampled)
    
    trial_indices = np.array(trial_indices)
    
    # Extract subsample
    subsample = binned_spikes[trial_indices][:, neuron_indices, :]
    subsample_labels = trial_labels[trial_indices]
    
    # Analyze
    return analyze_stability(subsample, subsample_labels, decoder_window_bins, n_folds)


# ============================================================================
# Pipeline functions
# ============================================================================

def download_data(config: ConfigLoader) -> None:
    """
    Download IBL data for specified sessions (Cell 2 equivalent).
    
    Args:
        config: Configuration loader
    """
    if ONE is None:
        logger.error("ONE API not available. Cannot download data.")
        return
    
    logger.info("=" * 80)
    logger.info("DOWNLOADING IBL DATA")
    logger.info("=" * 80)
    
    cache_dir = config.get_cache_dir()
    eids = config.get_eids()
    
    # Initialize ONE API with public credentials
    one = ONE(base_url=OPENALYX_URL, password=OPENALYX_PASSWORD, cache_dir=str(cache_dir))
    
    for eid in eids:
        logger.info(f"Downloading session: {eid}")
        try:
            # Download spike sorting and trials data
            one.load_datasets(eid, ['spikes.times', 'spikes.clusters', 'spikes.depths'], download_only=True)
            one.load_datasets(eid, ['trials.stimOn_times', 'trials.choice', 'trials.feedbackType'], download_only=True)
        except Exception as e:
            logger.error(f"Error downloading {eid}: {e}")
    
    # Optimization #1: Delete ONE object after use
    del one
    gc.collect()
    
    logger.info("Download complete")


def populate_cache(config: ConfigLoader) -> None:
    """
    Populate cache by loading and validating all session data (Cell 3 equivalent).
    
    Args:
        config: Configuration loader
    """
    logger.info("=" * 80)
    logger.info("POPULATING CACHE")
    logger.info("=" * 80)
    
    cache_dir = config.get_cache_dir()
    eids = config.get_eids()
    params = config.get_analysis_params()
    
    for eid in eids:
        logger.info(f"Processing session: {eid}")
        
        # Load spikes
        spikes = load_session_spikes(cache_dir, eid)
        if spikes is None:
            logger.warning(f"Skipping {eid} - no spike data")
            continue
        
        # Load trials
        trials = load_trials_direct(cache_dir, eid)
        if trials is None:
            logger.warning(f"Skipping {eid} - no trial data")
            continue
        
        # Validate data
        n_neurons = len(np.unique(spikes['clusters']))
        n_trials = len(trials.get('stimOn_times', []))
        
        if n_neurons < params['min_neurons']:
            logger.warning(f"Skipping {eid} - only {n_neurons} neurons (min: {params['min_neurons']})")
            continue
        
        if n_trials < params['min_trials']:
            logger.warning(f"Skipping {eid} - only {n_trials} trials (min: {params['min_trials']})")
            continue
        
        logger.info(f"Session valid: {n_neurons} neurons, {n_trials} trials")
        
        # Optimization #6: Cleanup after each session
        del spikes, trials
        gc.collect()
    
    logger.info("Cache population complete")


def run_analysis(config: ConfigLoader) -> None:
    """
    Run CTG stability analysis (Cell 4 equivalent).
    
    Args:
        config: Configuration loader
    """
    logger.info("=" * 80)
    logger.info("RUNNING CTG STABILITY ANALYSIS")
    logger.info("=" * 80)
    
    cache_dir = config.get_cache_dir()
    output_path = config.get_output_path()
    eids = config.get_eids()
    params = config.get_analysis_params()
    
    results = {}
    
    for eid in eids:
        logger.info(f"Analyzing session: {eid}")
        
        # Load data
        spikes = load_session_spikes(cache_dir, eid)
        trials = load_trials_direct(cache_dir, eid)
        
        if spikes is None or trials is None:
            logger.warning(f"Skipping {eid} - data not available")
            continue
        
        # Validate
        n_neurons = len(np.unique(spikes['clusters']))
        n_trials = len(trials.get('stimOn_times', []))
        
        if n_neurons < params['min_neurons'] or n_trials < params['min_trials']:
            logger.warning(f"Skipping {eid} - insufficient data")
            continue
        
        # Bin spikes around stimulus onset
        event_times = trials['stimOn_times']
        binned = bin_spikes_around_events(
            spikes['times'],
            spikes['clusters'],
            event_times,
            window=tuple(params['window_sec']),
            bin_size=params['bin_size']
        )
        
        # Use choice as trial labels
        labels = trials.get('choice')
        if labels is None:
            logger.warning(f"Skipping {eid} - no choice labels")
            continue
        
        # Remove trials with invalid labels
        valid_mask = ~np.isnan(labels)
        binned = binned[valid_mask]
        labels = labels[valid_mask].astype(int)
        
        logger.info(f"Running {params['n_iterations']} subsample iterations with {params['n_jobs']} jobs")
        
        # Run subsampled analysis in parallel (Optimization #5)
        iteration_results = Parallel(n_jobs=params['n_jobs'])(
            delayed(process_subsample)(
                binned, labels,
                params['n_subsample_neurons'],
                params['n_subsample_trials'],
                params['decoder_window_bins'],
                params['n_folds'],
                i
            )
            for i in range(params['n_iterations'])
        )
        
        # Aggregate results
        ctg_matrices = np.stack([r['ctg_matrix'] for r in iteration_results])
        self_consistencies = np.stack([r['self_consistency'] for r in iteration_results])
        
        results[eid] = {
            'ctg_mean': np.mean(ctg_matrices, axis=0),
            'ctg_std': np.std(ctg_matrices, axis=0),
            'self_consistency_mean': np.mean(self_consistencies, axis=0),
            'self_consistency_std': np.std(self_consistencies, axis=0),
            'n_neurons': n_neurons,
            'n_trials': n_trials
        }
        
        # Save results
        output_file = output_path / f'{eid}_ctg_results.npz'
        np.savez(
            output_file,
            **results[eid]
        )
        logger.info(f"Results saved to {output_file}")
        
        # Optimization #6: Cleanup
        del spikes, trials, binned, iteration_results
        gc.collect()
    
    logger.info("=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info(f"Results saved to: {output_path}")
    logger.info("=" * 80)


# ============================================================================
# CLI and main entry point
# ============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI."""
    parser = argparse.ArgumentParser(
        description='CTG Stability Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default config
  python -m src.analysis.ctg_pipeline
  
  # Run with custom config file
  python -m src.analysis.ctg_pipeline --config my_config.yaml
  
  # Override cache directory
  python -m src.analysis.ctg_pipeline --cache-dir /path/to/cache
  
  # Use custom EIDs file
  python -m src.analysis.ctg_pipeline --eids-file my_eids.txt
  
  # Run only download step
  python -m src.analysis.ctg_pipeline --step download
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to config.yaml file (default: looks for config.yaml in penelope/)'
    )
    
    parser.add_argument(
        '--cache-dir',
        type=str,
        help='Cache directory for IBL data (overrides config)'
    )
    
    parser.add_argument(
        '--eids-file',
        type=str,
        help='Path to file with EIDs (one per line, overrides config)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output directory for results (default: cache_dir/results)'
    )
    
    parser.add_argument(
        '--step',
        choices=['download', 'cache', 'analysis', 'all'],
        default='all',
        help='Which pipeline step to run (default: all)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser


def main(args: Optional[List[str]] = None) -> int:
    """
    Main entry point for CTG pipeline.
    
    Args:
        args: Command line arguments (None = use sys.argv)
    
    Returns:
        Exit code (0 = success)
    """
    parser = create_parser()
    parsed_args = parser.parse_args(args)
    
    # Set logging level
    if parsed_args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    config_path = Path(parsed_args.config) if parsed_args.config else None
    config = ConfigLoader(config_path)
    
    # Apply environment and CLI overrides
    config.apply_env_overrides()
    config.apply_cli_overrides(parsed_args)
    
    # Log configuration
    logger.info(f"Cache directory: {config.get_cache_dir()}")
    logger.info(f"Output directory: {config.get_output_path()}")
    logger.info(f"Number of EIDs: {len(config.get_eids())}")
    
    # Run pipeline steps
    try:
        if parsed_args.step in ['download', 'all']:
            download_data(config)
        
        if parsed_args.step in ['cache', 'all']:
            populate_cache(config)
        
        if parsed_args.step in ['analysis', 'all']:
            run_analysis(config)
        
        logger.info("Pipeline completed successfully")
        return 0
    
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
