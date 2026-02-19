# Penelope - CTG Stability Analysis Pipeline

Cross-Temporal Generalization (CTG) analysis pipeline for neural stability assessment using IBL data.

## Quick Start

### 1. Setup Configuration

Copy the default configuration template:

```bash
cd penelope
cp config_default.yaml config.yaml
```

Edit `config.yaml` to customize:
- `cache_dir`: Where to store downloaded IBL data
- `eids`: List of session EIDs to analyze
- `analysis`: Analysis parameters (bin size, window, iterations, etc.)

**Note:** `config.yaml` is gitignored, so your local settings won't be committed.

### 2. Running the Pipeline

#### From Command Line

```bash
# Run full pipeline (download + cache + analysis)
python -m src.analysis.ctg_pipeline

# Run with custom config
python -m src.analysis.ctg_pipeline --config my_config.yaml

# Override cache directory
python -m src.analysis.ctg_pipeline --cache-dir /path/to/cache

# Use custom EIDs file
python -m src.analysis.ctg_pipeline --eids-file my_eids.txt

# Run only specific steps
python -m src.analysis.ctg_pipeline --step download
python -m src.analysis.ctg_pipeline --step cache
python -m src.analysis.ctg_pipeline --step analysis
```

#### From Jupyter Notebook

Open and run `notebooks/ctg_pipeline.ipynb`:

1. **Cell 1:** Install dependencies and mount drive (if on Colab)
2. **Cell 2:** Download IBL data
3. **Cell 3:** Populate cache and validate
4. **Cell 4:** Run CTG analysis
5. **Cell 5:** Visualize results

### 3. Environment Variables

You can override config settings via environment variables:

```bash
export CTG_CACHE_DIR=/path/to/cache
python -m src.analysis.ctg_pipeline
```

## Configuration

### Cache Directory

The pipeline auto-detects your environment:
- **Google Colab:** `/content/drive/MyDrive/IBL_Project/Cache` (default)
- **Local machine:** `~/ibl_data` (default)

Override with:
1. `config.yaml` → `cache_dir: /your/path`
2. Environment: `export CTG_CACHE_DIR=/your/path`
3. CLI: `--cache-dir /your/path`

### Session EIDs

Specify sessions to analyze in two ways:

**Option 1: In config.yaml**
```yaml
eids:
  - '5569f363-0934-464e-9a5b-77c8e67791a1'
  - '1191f865-b10a-45c8-9c48-24a980fd9402'
```

**Option 2: In a text file (one EID per line)**
```bash
python -m src.analysis.ctg_pipeline --eids-file my_sessions.txt
```

### Analysis Parameters

Configure in `config.yaml`:

```yaml
analysis:
  bin_size: 0.05              # Time bin size in seconds
  window_sec: [-0.2, 0.8]     # Analysis window around events
  decoder_window_bins: 3      # Number of bins for decoder window
  n_folds: 5                  # Cross-validation folds
  n_iterations: 30            # Subsample iterations
  n_subsample_neurons: 20     # Neurons per subsample
  n_subsample_trials: 300     # Trials per subsample
  min_neurons: 10             # Minimum neurons to include session
  min_trials: 200             # Minimum trials to include session
  n_jobs: 2                   # Parallel jobs
```

## Pipeline Architecture

### Optimizations

The pipeline includes several performance optimizations:

1. **Delete ONE object after path lookup** — Prevents memory leaks
2. **Direct np.load with rglob** — No SpikeSortingLoader overhead
3. **float32 spike times** — Reduces memory usage
4. **Bin only around events** — Not whole session
5. **Controlled joblib parallelism** — Efficient CPU usage
6. **gc.collect() after each session** — Memory cleanup
7. **Revision-safe picker** — Handles multiple data revisions

### Module Structure

```
penelope/
├── config_default.yaml          # Template configuration
├── config.yaml                  # User configuration (gitignored)
├── src/
│   └── analysis/
│       └── ctg_pipeline.py      # Main pipeline module
├── notebooks/
│   └── ctg_pipeline.ipynb       # Notebook interface
├── docs/
│   └── lme_model.md             # Analysis documentation
└── README.md                    # This file
```

### Pipeline Steps

1. **Download** (`download_data()`)
   - Connects to IBL OpenAlyx (public credentials)
   - Downloads spike sorting and trial data
   - Stores in cache directory

2. **Cache Population** (`populate_cache()`)
   - Loads and validates all sessions
   - Checks minimum neuron/trial requirements
   - Reports data quality

3. **Analysis** (`run_analysis()`)
   - Bins spikes around events
   - Runs cross-temporal generalization
   - Computes stability metrics
   - Saves results to output directory

## OpenAlyx Credentials

**Public IBL credentials (hardcoded):**
- URL: `https://openalyx.internationalbrainlab.org`
- Password: `international`

These are public credentials shared by the entire IBL community and documented in the IBL data access guide.

## Output

Results are saved to `{cache_dir}/results/` (or custom `--output` path):

```
results/
├── {eid}_ctg_results.npz
├── {eid}_ctg_results.npz
└── ...
```

Each `.npz` file contains:
- `ctg_mean`: Mean CTG matrix across iterations
- `ctg_std`: Standard deviation of CTG matrix
- `self_consistency_mean`: Mean self-consistency over time
- `self_consistency_std`: Std of self-consistency
- `n_neurons`: Number of neurons in session
- `n_trials`: Number of trials in session

## Dependencies

Required packages:
- `ONE-api` — IBL data access
- `scikit-learn` — Decoding analysis
- `joblib` — Parallel processing
- `pyyaml` — Configuration files
- `numpy` — Numerical computing
- `pandas` — Data manipulation (optional)
- `matplotlib` — Visualization (optional)
- `seaborn` — Enhanced plotting (optional)

Install with:
```bash
pip install ONE-api scikit-learn joblib pyyaml numpy pandas matplotlib seaborn
```

## Troubleshooting

### Import Errors

If you get import errors when running the notebook:
```python
import sys
sys.path.insert(0, '/path/to/penelope')
```

### ONE API Connection

If you can't connect to OpenAlyx:
- Check internet connection
- Verify URL: `https://openalyx.internationalbrainlab.org`
- Public password is: `international`

### Memory Issues

If analysis runs out of memory:
- Reduce `n_subsample_neurons` and `n_subsample_trials`
- Reduce `n_iterations`
- Reduce `n_jobs` (less parallelism)
- Analyze fewer sessions at once

## Contributing

When adding sessions to analyze:
1. Add EIDs to your local `config.yaml` (not committed)
2. Or create a custom `my_sessions.txt` file
3. Don't commit large data files or cache directories

## License

Part of the Decisive Times project.
