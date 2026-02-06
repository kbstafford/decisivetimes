"""
IBL Timescale Data Loader GUI
==============================

PyQt6 desktop application for interactively selecting and previewing
IBL Brainwide Map data before loading for timescale analysis.

Features:
- Browse subjects, sessions, and probes via ONE API
- Filter by brain region, lab, date range
- Preview spike rasters and firing rates
- Export selection to config or load directly

Requirements:
    pip install PyQt6 pyqtgraph numpy pandas ONE-api iblatlas

Usage:
    python ibl_data_loader_gui.py
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QTreeWidget, QTreeWidgetItem, QTableWidget, QTableWidgetItem,
    QGroupBox, QLabel, QLineEdit, QComboBox, QPushButton, QCheckBox,
    QSpinBox, QDoubleSpinBox, QProgressBar, QStatusBar, QTabWidget,
    QTextEdit, QFileDialog, QMessageBox, QHeaderView, QAbstractItemView,
    QListWidget, QListWidgetItem
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QColor, QAction

import pyqtgraph as pg

# IBL imports - wrapped for graceful failure
try:
    from one.api import ONE
    from iblatlas.atlas import AllenAtlas
    from iblatlas.regions import BrainRegions
    HAS_IBL = True
except ImportError:
    HAS_IBL = False
    print("Warning: IBL packages not installed. Install with:")
    print("  pip install ONE-api iblatlas")


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SessionInfo:
    """Container for session metadata."""
    eid: str
    subject: str
    date: str
    lab: str
    n_trials: int
    performance: float
    probes: List[str] = field(default_factory=list)
    pids: List[str] = field(default_factory=list)


@dataclass
class ProbeInfo:
    """Container for probe metadata."""
    pid: str
    eid: str
    probe_name: str
    n_units: int
    n_good_units: int
    regions: List[str] = field(default_factory=list)


@dataclass 
class UnitInfo:
    """Container for unit metadata."""
    cluster_id: int
    pid: str
    acronym: str
    firing_rate: float
    n_spikes: int
    label: int
    amp_median: float
    depth_um: float

@dataclass
class DatasetCatalog:
    eid: str
    datasets: List[str]
    by_collection: Dict[str, List[str]]

    @classmethod
    def from_one(cls, one: "ONE", eid: str) -> "DatasetCatalog":
        ds = one.list_datasets(eid)  # list of dataset paths (strings)
        by_col: Dict[str, List[str]] = {}
        for p in ds:
            # crude but works: collection is everything up to last '/'
            col = "/".join(p.split("/")[:-1])
            by_col.setdefault(col, []).append(p)
        return cls(eid=eid, datasets=ds, by_collection=by_col)

    def has_object(self, obj: str, collection: Optional[str] = None) -> bool:
        # obj like 'trials', 'spikes', 'clusters', 'wheel'
        if collection:
            prefix = f"{collection}/{obj}."
            return any(d.startswith(prefix) for d in self.datasets)
        # anywhere
        return any(f"/{obj}." in d or d.endswith(f"{obj}.pqt") for d in self.datasets)

    def collections_containing(self, obj: str) -> List[str]:
        cols = []
        for col, paths in self.by_collection.items():
            if any(f"/{obj}." in p for p in paths):
                cols.append(col)
        return cols

    def guess_probe_collection(self, probe_name: str) -> Optional[str]:
        # typical: 'alf/probe00'
        c = f"alf/{probe_name}"
        return c if c in self.by_collection else None

# =============================================================================
# Worker Threads
# =============================================================================

class ONEConnectionWorker(QThread):
    """Background thread for ONE API connection."""
    finished = pyqtSignal(bool, str, object)  # success, message, one_instance
    
    def __init__(self, base_url: str = "https://openalyx.internationalbrainlab.org", 
                 mode: str = "auto"):
        super().__init__()
        self.base_url = base_url
        self.mode = mode
        self.one = None
    
    def run(self):
        try:
            # Use mode='local' first to avoid interactive prompts
            # If that fails, we'll need user to authenticate separately
            self.one = ONE(
                base_url=self.base_url,
                mode=self.mode,
                silent=True,  # Suppress interactive prompts
            )
            # Test connection with a simple query
            _ = self.one.search(subject='test_fake_subject_12345', query_type='remote')
            self.finished.emit(True, "Connected to ONE API", self.one)
        except Exception as e:
            error_msg = str(e)
            if "authenticate" in error_msg.lower() or "password" in error_msg.lower():
                self.finished.emit(False, 
                    "Authentication required. Run 'one.setup()' in terminal first.", None)
            else:
                self.finished.emit(False, f"Connection failed: {error_msg}", None)


class SessionLoadWorker(QThread):
    """Background thread for loading session list."""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(list)
    error = pyqtSignal(str)
    
    def __init__(self, one: 'ONE', filters: Dict[str, Any]):
        super().__init__()
        self.one = one
        self.filters = filters

    def _probe_hits_region(self, eid: str, probe_name: str, region_acronym: str) -> bool:
        """
        Fast check: does this probe contain ANY channel whose atlas ID is a descendant of region_acronym?
        Loads only small channel location arrays (not spikes).
        """
        # Expand region → descendant IDs (including itself)
        target_id = int(self.br.acronym2id([region_acronym])[0])
        desc_ids = set(self.br.descendants([target_id]).tolist())

        # Preferred: brainLocationIds (tiny int array)
        try:
            ids = self.one.load_dataset(
                eid,
                "channels.brainLocationIds_ccf_2017.npy",
                collection=f"alf/{probe_name}"
            )
            ids = np.asarray(ids)
            ids = ids[~np.isnan(ids)].astype(int)
            return any(i in desc_ids for i in ids)
        except Exception:
            pass

        # Fallback: acronyms directly (also small if present)
        try:
            acr = self.one.load_dataset(
                eid,
                "channels.acronym.npy",
                collection=f"alf/{probe_name}"
            )
            acr = set(map(str, np.asarray(acr)))
            desc_acr = set(self.br.id2acronym(np.array(list(desc_ids), dtype=int)))
            return len(acr & desc_acr) > 0
        except Exception:
            return False


    def run(self):
        try:
            self.progress.emit(10, "Querying sessions...")

            query_params = {
                'task_protocol': 'ephysChoiceWorld',
                'project': 'brainwide',
            }
            if self.filters.get('lab'):
                query_params['lab'] = self.filters['lab']
            if self.filters.get('subject'):
                query_params['subject'] = self.filters['subject']

            region = self.filters.get('region')  # NEW
            region = region.strip() if isinstance(region, str) else None

            eids = self.one.search(**query_params)
            self.progress.emit(30, f"Found {len(eids)} sessions")

            sessions = []
            max_sessions = 100  # keep your cap

            for i, eid in enumerate(eids[:max_sessions]):
                try:
                    sess_info = self.one.get_details(eid)

                    insertions = self.one.alyx.rest('insertions', 'list', session=eid)
                    probes = [ins.get('name', 'unknown') for ins in insertions]
                    pids = [ins.get('id', None) for ins in insertions]

                    # --- NEW: region-based session filter (channels only) ---
                    if region:
                        # allow comma-separated (e.g., "CA3,CA1")
                        region_list = [r.strip() for r in region.split(",") if r.strip()]
                        has_region = False

                        for probe_name in probes:
                            if probe_name == "unknown":
                                continue
                            # match if probe hits ANY requested region
                            if any(self._probe_hits_region(eid, probe_name, r) for r in region_list):
                                has_region = True
                                break

                        if not has_region:
                            # skip session entirely
                            continue
                    # --------------------------------------------------------

                    # Trial count & perf (optional; this is heavier than channels)
                    try:
                        trials = self.one.load_object(eid, 'trials', collection='alf')
                        n_trials = len(trials.get('choice', []))
                        correct = trials.get('feedbackType', [])
                        performance = np.mean(correct == 1) if len(correct) > 0 else 0
                    except Exception:
                        n_trials = 0
                        performance = 0

                    sessions.append(SessionInfo(
                        eid=eid,
                        subject=sess_info.get('subject', 'unknown'),
                        date=str(sess_info.get('start_time', ''))[:10],
                        lab=sess_info.get('lab', 'unknown'),
                        n_trials=n_trials,
                        performance=performance,
                        probes=probes,
                        pids=pids,
                    ))
                except Exception:
                    continue

                self.progress.emit(
                    30 + int(60 * i / max(1, min(len(eids), max_sessions))),
                    f"Loading session {i+1}/{min(len(eids), max_sessions)}"
                )

            self.progress.emit(100, f"Loaded {len(sessions)} sessions")
            self.finished.emit(sessions)

        except Exception as e:
            self.error.emit(str(e))


class UnitsLoadWorker(QThread):
    """Background thread for loading units from a probe."""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(list, object)  # units, spike_times
    error = pyqtSignal(str)
    
    def __init__(self, one: 'ONE', pid: str, eid: str, probe_name: str = None):
        super().__init__()
        self.one = one
        self.pid = pid
        self.eid = eid
        self.probe_name = probe_name  # e.g., "probe00"
    
    def run(self):
        try:
            # Determine collection path
            collection = f'alf/{self.probe_name}' if self.probe_name else None
            
            self.progress.emit(10, f"Finding spike data in {collection or 'default'}...")

            # First, list available datasets to understand the structure
            try:
                datasets = self.one.list_datasets(self.eid)
                spike_datasets = [d for d in datasets if 'spike' in d.lower()]
                self.progress.emit(15, f"Found {len(spike_datasets)} spike-related datasets")
            except:
                spike_datasets = []

            self.progress.emit(20, "Loading spikes...")

            # Try multiple approaches to load spikes
            spikes = None
            load_errors = []

            # Approach 1: Standard with collection
            if spikes is None and collection:
                try:
                    spikes = self.one.load_object(self.eid, 'spikes', collection=collection)
                except Exception as e:
                    load_errors.append(f"collection={collection}: {e}")

            # Approach 2: Without collection
            if spikes is None:
                try:
                    spikes = self.one.load_object(self.eid, 'spikes')
                except Exception as e:
                    load_errors.append(f"no collection: {e}")

            # Approach 3: With _ibl_ namespace
            if spikes is None and collection:
                try:
                    spikes = self.one.load_object(self.eid, 'spikes',
                                                   collection=collection,
                                                   namespace='ibl')
                except Exception as e:
                    load_errors.append(f"namespace=ibl: {e}")

            # Approach 4: Load individual files directly
            if spikes is None:
                try:
                    self.progress.emit(25, "Trying direct file loading...")
                    spike_times_file = [d for d in datasets if 'spikes.times' in d and self.probe_name in d]
                    spike_clusters_file = [d for d in datasets if 'spikes.clusters' in d and self.probe_name in d]

                    if spike_times_file and spike_clusters_file:
                        spike_times = self.one.load_dataset(self.eid, spike_times_file[0])
                        spike_clusters = self.one.load_dataset(self.eid, spike_clusters_file[0])
                        spikes = {'times': spike_times, 'clusters': spike_clusters}

                        # Try to load amps too
                        spike_amps_file = [d for d in datasets if 'spikes.amps' in d and self.probe_name in d]
                        if spike_amps_file:
                            spikes['amps'] = self.one.load_dataset(self.eid, spike_amps_file[0])
                except Exception as e:
                    load_errors.append(f"direct loading: {e}")

            if spikes is None:
                error_details = "\n".join(load_errors)
                self.error.emit(f"Failed to load spikes: spikes\n\nTried approaches:\n{error_details}")
                return

            self.progress.emit(50, "Loading clusters...")

            # Load clusters with similar fallback approach
            clusters = None

            if collection:
                try:
                    clusters = self.one.load_object(self.eid, 'clusters', collection=collection)
                except:
                    pass

            if clusters is None:
                try:
                    clusters = self.one.load_object(self.eid, 'clusters')
                except:
                    pass

            if clusters is None:
                try:
                    # Load individual cluster files
                    cluster_files = [d for d in datasets if 'clusters.' in d and self.probe_name in d]
                    if cluster_files:
                        clusters = {}
                        for cf in cluster_files:
                            attr = cf.split('clusters.')[-1].split('.')[0]
                            try:
                                clusters[attr] = self.one.load_dataset(self.eid, cf)
                            except:
                                pass
                except:
                    clusters = {}

            if clusters is None:
                clusters = {}

            self.progress.emit(70, "Loading channels...")

            # Load channels
            channels = None

            if collection:
                try:
                    channels = self.one.load_object(self.eid, 'channels', collection=collection)
                except:
                    pass

            if channels is None:
                try:
                    channels = self.one.load_object(self.eid, 'channels')
                except:
                    pass

            if channels is None:
                try:
                    channel_files = [d for d in datasets if 'channels.' in d and self.probe_name in d]
                    if channel_files:
                        channels = {}
                        for cf in channel_files:
                            attr = cf.split('channels.')[-1].split('.')[0]
                            try:
                                channels[attr] = self.one.load_dataset(self.eid, cf)
                            except:
                                pass
                except:
                    channels = {}

            if channels is None:
                channels = {}

            self.progress.emit(85, "Processing units...")

            units = []
            spike_times = spikes.get('times', np.array([]))
            spike_clusters = spikes.get('clusters', np.array([]))
            spike_amps = spikes.get('amps', None)

            # Get unique cluster IDs from spikes
            unique_clusters = np.unique(spike_clusters)
            n_clusters = len(unique_clusters)

            # IBL cluster attributes (all indexed by cluster_id)
            cluster_amps = clusters.get('amps', None)           # Average amplitude (V) (double)
            cluster_depths = clusters.get('depths', None)       # Depth in µm (double)
            cluster_channels = clusters.get('channels', None)   # Peak channel index (int)
            cluster_metrics = clusters.get('metrics', None)     # QC metrics (DataFrame/csv)
            cluster_peak_to_trough = clusters.get('peakToTrough', None)  # Waveform duration (ms)
            cluster_uuids = clusters.get('uuids', None)  # Cluster identifier (int128)

            # Channel attributes for brain regions
            br = BrainRegions()
            ccf_ids = channels.get("brainLocationIds_ccf_2017", None)
            if ccf_ids is not None:
                channel_acronyms = br.id2acronym(ccf_ids).astype(str)
            else:
                channel_acronyms = None

            # If metrics is a DataFrame, index it
            metrics_df = None
            metrics_cols = []
            if cluster_metrics is not None:
                if isinstance(cluster_metrics, pd.DataFrame):
                    metrics_df = cluster_metrics
                    metrics_cols = metrics_df.columns.tolist()
                elif hasattr(cluster_metrics, 'to_frame'):
                    metrics_df = cluster_metrics.to_frame()
                    metrics_cols = metrics_df.columns.tolist()

            # Log what we found for debugging
            label_col = None
            for col in ['label', 'ks2_label', 'bitwise_label']:
                if col in metrics_cols:
                    label_col = col
                    break

            self.progress.emit(87, f"Found metrics cols: {len(metrics_cols)}, label col: {label_col}")

            for i, cluster_id in enumerate(unique_clusters):
                cluster_id = int(cluster_id)

                # Count spikes and compute firing rate
                spike_mask = spike_clusters == cluster_id
                n_spikes = int(np.sum(spike_mask))
                cluster_spike_times = spike_times[spike_mask]

                if len(cluster_spike_times) > 1:
                    duration = cluster_spike_times[-1] - cluster_spike_times[0]
                    firing_rate = n_spikes / duration if duration > 0 else 0
                else:
                    firing_rate = 0

                # Get depth (µm)
                depth = 0
                if cluster_depths is not None and cluster_id < len(cluster_depths):
                    depth = float(cluster_depths[cluster_id])

                # Get amplitude (convert V to µV)
                amp_median = 0
                if cluster_amps is not None and cluster_id < len(cluster_amps):
                    amp_median = float(cluster_amps[cluster_id]) * 1e6  # V to µV

                # Get peak channel
                peak_channel = 0
                if cluster_channels is not None and cluster_id < len(cluster_channels):
                    peak_channel = int(cluster_channels[cluster_id])

                # Get brain region from channel
                acronym = 'unknown'
                if channel_acronyms is not None and peak_channel < len(channel_acronyms):
                    acronym = str(channel_acronyms[peak_channel])

                # Get label and other metrics from metrics DataFrame
                # IBL label: 1.0 = all 3 QC pass, 0.666 = 2/3, 0.333 = 1/3, 0 = none
                label = 0
                label_raw = 0.0  # Keep raw value for potential filtering
                if metrics_df is not None:
                    try:
                        if 'cluster_id' in metrics_df.columns:
                            row = metrics_df[metrics_df['cluster_id'] == cluster_id]
                        else:
                            row = metrics_df.iloc[[cluster_id]] if cluster_id < len(metrics_df) else None

                        if row is not None and len(row) > 0:
                            row = row.iloc[0]

                            # Try different label column names used by IBL
                            # Priority: label > ks2_label > bitwise_label
                            if 'label' in row.index and pd.notna(row['label']):
                                val = row['label']
                                if isinstance(val, (int, float, np.integer, np.floating)):
                                    label_raw = float(val)
                                    # IBL: label == 1.0 means all 3 QC metrics pass
                                    # We consider >= 1.0 as "good" (label=1)
                                    label = 1 if label_raw >= 1.0 else 0
                                elif isinstance(val, str):
                                    label = 1 if val.lower() == 'good' else 0
                                    label_raw = 1.0 if label == 1 else 0.0
                            elif 'ks2_label' in row.index and pd.notna(row['ks2_label']):
                                # ks2_label is typically 'good' or 'mua' string
                                ks_label = str(row['ks2_label']).lower().strip()
                                label = 1 if ks_label == 'good' else 0
                                label_raw = 1.0 if label == 1 else 0.0
                            elif 'bitwise_label' in row.index and pd.notna(row['bitwise_label']):
                                # bitwise_label: 1 = good
                                label = 1 if int(row['bitwise_label']) == 1 else 0
                                label_raw = 1.0 if label == 1 else 0.0

                            # Override firing_rate if available in metrics
                            if 'firing_rate' in row.index and pd.notna(row['firing_rate']):
                                firing_rate = float(row['firing_rate'])
                    except Exception as e:
                        # If metrics lookup fails, keep default label=0
                        pass

                units.append(UnitInfo(
                    cluster_id=cluster_id,
                    pid=self.pid,
                    acronym=acronym,
                    firing_rate=float(firing_rate),
                    n_spikes=n_spikes,
                    label=label,
                    amp_median=float(amp_median),
                    depth_um=float(depth),
                ))

            self.progress.emit(100, f"Loaded {len(units)} units")
            self.finished.emit(units, {'times': spike_times, 'clusters': spike_clusters})

        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n\n{traceback.format_exc()}")


# =============================================================================
# Main Window
# =============================================================================

class IBLDataLoaderGUI(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.br = BrainRegions()  # load once
        self.one: Optional['ONE'] = None
        self.sessions: List[SessionInfo] = []
        self.current_units: List[UnitInfo] = []
        self.current_spikes: Optional[Dict] = None
        self.selected_units: List[int] = []

        self.init_ui()
        self.setup_connections()

        # Don't auto-connect - wait for user to click Connect
        self.status_bar.showMessage("Ready - Select mode and click 'Connect'")

    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("IBL Timescale Data Loader")
        self.setGeometry(100, 100, 1400, 900)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        # Main splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # Left panel: Selection
        left_panel = self.create_selection_panel()
        splitter.addWidget(left_panel)

        # Right panel: Preview
        right_panel = self.create_preview_panel()
        splitter.addWidget(right_panel)

        splitter.setSizes([500, 900])

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.hide()
        self.status_bar.addPermanentWidget(self.progress_bar)

        # Menu bar
        self.create_menu_bar()

    def show_available_regions_dialog(self):
        if not self.current_units:
            QMessageBox.information(self, "No units loaded", "Load a probe first to see available regions.")
            return

        regions = sorted({u.acronym for u in self.current_units if u.acronym and u.acronym != "unknown"})

        dlg = QDialog(self)
        dlg.setWindowTitle("Available Regions in This Probe")
        dlg.setMinimumSize(420, 500)

        v = QVBoxLayout(dlg)

        search = QLineEdit()
        search.setPlaceholderText("Search (e.g., CA, ENT, ORB)...")
        v.addWidget(search)

        lst = QListWidget()
        lst.addItems(regions)
        v.addWidget(lst)

        btn_row = QHBoxLayout()
        use_btn = QPushButton("Use selected")
        close_btn = QPushButton("Close")
        btn_row.addWidget(use_btn)
        btn_row.addWidget(close_btn)
        v.addLayout(btn_row)

        def apply_filter():
            items = lst.selectedItems()
            if items:
                chosen = [it.text() for it in items]
                self.region_edit.setText(", ".join(chosen))
            dlg.accept()

        def filter_list(text):
            text = text.strip().upper()
            lst.clear()
            if not text:
                lst.addItems(regions)
            else:
                lst.addItems([r for r in regions if text in r.upper()])

        use_btn.clicked.connect(apply_filter)
        close_btn.clicked.connect(dlg.reject)
        lst.itemDoubleClicked.connect(lambda _: apply_filter())
        search.textChanged.connect(filter_list)

        dlg.exec()

    def create_menu_bar(self):
        """Create the menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        export_action = QAction("Export Selection to Parquet", self)
        export_action.triggered.connect(self.export_to_parquet)
        file_menu.addAction(export_action)

        export_action = QAction("Export Selection to YAML", self)
        export_action.triggered.connect(self.export_to_yaml)
        file_menu.addAction(export_action)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Help menu
        help_menu = menubar.addMenu("Help")
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def create_selection_panel(self) -> QWidget:
        """Create the left selection panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Connection group
        conn_group = QGroupBox("ONE API Connection")
        conn_layout = QVBoxLayout(conn_group)

        url_row = QHBoxLayout()
        self.url_edit = QLineEdit("https://openalyx.internationalbrainlab.org")
        url_row.addWidget(QLabel("URL:"))
        url_row.addWidget(self.url_edit)
        conn_layout.addLayout(url_row)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["local", "auto", "remote"])
        self.mode_combo.setCurrentText("local")
        self.mode_combo.setToolTip(
            "local: Use cached data only (no login needed)\n"
            "auto: Use cache, fetch if needed\n"
            "remote: Always fetch from server"
        )
        mode_row.addWidget(self.mode_combo)

        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self.connect_to_one)
        mode_row.addWidget(self.connect_btn)

        self.conn_status = QLabel("⚪ Disconnected")
        mode_row.addWidget(self.conn_status)
        mode_row.addStretch()
        conn_layout.addLayout(mode_row)

        # Auth help label
        auth_label = QLabel(
            "<small>💡 If 'auto' hangs, run in terminal first: "
            "<code>python -c \"from one.api import ONE; ONE.setup()\"</code></small>"
        )
        auth_label.setTextFormat(Qt.TextFormat.RichText)
        auth_label.setWordWrap(True)
        conn_layout.addWidget(auth_label)

        layout.addWidget(conn_group)

        # Filters group
        filter_group = QGroupBox("Filters")
        filter_layout = QVBoxLayout(filter_group)

        # Lab filter
        lab_row = QHBoxLayout()
        lab_row.addWidget(QLabel("Lab:"))
        self.lab_combo = QComboBox()
        self.lab_combo.addItems(["All", "hoferlab", "churchlandlab", "cortexlab",
                                  "danlab", "maiabordalab", "wittenlab", "zaborlab"])
        lab_row.addWidget(self.lab_combo)
        filter_layout.addLayout(lab_row)

        # Subject filter
        subj_row = QHBoxLayout()
        subj_row.addWidget(QLabel("Subject:"))
        self.subject_edit = QLineEdit()
        self.subject_edit.setPlaceholderText("e.g., KS001 (leave empty for all)")
        subj_row.addWidget(self.subject_edit)
        filter_layout.addLayout(subj_row)

        # Region filter
        region_row = QHBoxLayout()
        region_row.addWidget(QLabel("Region:"))
        self.region_edit = QLineEdit()
        self.region_edit.setPlaceholderText("e.g., VISp, CA1 (comma-separated)")
        region_row.addWidget(self.region_edit)
        filter_layout.addLayout(region_row)

        # Good units only
        self.good_only_check = QCheckBox("Good units only (label=1)")
        self.good_only_check.setChecked(True)
        filter_layout.addWidget(self.good_only_check)

        # Load button
        self.load_sessions_btn = QPushButton("Load Sessions")
        self.load_sessions_btn.clicked.connect(self.load_sessions)
        self.load_sessions_btn.setEnabled(False)
        filter_layout.addWidget(self.load_sessions_btn)

        layout.addWidget(filter_group)

        # Sessions table
        sessions_group = QGroupBox("Sessions")
        sessions_layout = QVBoxLayout(sessions_group)

        self.sessions_table = QTableWidget()
        self.sessions_table.setColumnCount(6)
        self.sessions_table.setHorizontalHeaderLabels(
            ["Subject", "Date", "Lab", "Trials", "Perf", "Probes"]
        )
        self.sessions_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.sessions_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.sessions_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.sessions_table.itemSelectionChanged.connect(self.on_session_selected)
        sessions_layout.addWidget(self.sessions_table)

        layout.addWidget(sessions_group)

        # Probes list
        probes_group = QGroupBox("Probes")
        probes_layout = QVBoxLayout(probes_group)

        self.probes_list = QListWidget()
        self.probes_list.itemSelectionChanged.connect(self.on_probe_selected)
        probes_layout.addWidget(self.probes_list)

        layout.addWidget(probes_group)

        return panel

    def create_preview_panel(self) -> QWidget:
        """Create the right preview panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Tabs for different views
        tabs = QTabWidget()
        layout.addWidget(tabs)

        # Units table tab
        units_widget = QWidget()
        units_layout = QVBoxLayout(units_widget)

        self.units_table = QTableWidget()
        self.units_table.setColumnCount(7)
        self.units_table.setHorizontalHeaderLabels(
            ["ID", "Region", "FR (Hz)", "Spikes", "Label", "Amp (µV)", "Depth (µm)"]
        )
        self.units_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.units_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.units_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.units_table.itemSelectionChanged.connect(self.on_units_selected)
        units_layout.addWidget(self.units_table)

        # Unit count label
        self.unit_count_label = QLabel("No units loaded")
        units_layout.addWidget(self.unit_count_label)

        tabs.addTab(units_widget, "Units Table")

        # Raster plot tab
        raster_widget = QWidget()
        raster_layout = QVBoxLayout(raster_widget)

        self.raster_plot = pg.PlotWidget(title="Spike Raster")
        self.raster_plot.setLabel('left', 'Unit')
        self.raster_plot.setLabel('bottom', 'Time (s)')
        raster_layout.addWidget(self.raster_plot)

        # Raster controls
        raster_controls = QHBoxLayout()
        raster_controls.addWidget(QLabel("Time range (s):"))
        self.raster_start = QDoubleSpinBox()
        self.raster_start.setRange(0, 10000)
        self.raster_start.setValue(0)
        raster_controls.addWidget(self.raster_start)
        raster_controls.addWidget(QLabel("to"))
        self.raster_end = QDoubleSpinBox()
        self.raster_end.setRange(0, 10000)
        self.raster_end.setValue(10)
        raster_controls.addWidget(self.raster_end)

        self.update_raster_btn = QPushButton("Update Raster")
        self.update_raster_btn.clicked.connect(self.update_raster_plot)
        raster_controls.addWidget(self.update_raster_btn)
        raster_controls.addStretch()

        raster_layout.addLayout(raster_controls)
        tabs.addTab(raster_widget, "Spike Raster")

        # Firing rate histogram tab
        fr_widget = QWidget()
        fr_layout = QVBoxLayout(fr_widget)

        self.fr_plot = pg.PlotWidget(title="Firing Rate Distribution")
        self.fr_plot.setLabel('left', 'Count')
        self.fr_plot.setLabel('bottom', 'Firing Rate (Hz)')
        fr_layout.addWidget(self.fr_plot)

        tabs.addTab(fr_widget, "FR Distribution")

        # Region summary tab
        region_widget = QWidget()
        region_layout = QVBoxLayout(region_widget)

        self.region_table = QTableWidget()
        self.region_table.setColumnCount(4)
        self.region_table.setHorizontalHeaderLabels(["Region", "Total", "Good", "Mean FR"])
        self.region_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        region_layout.addWidget(self.region_table)

        tabs.addTab(region_widget, "Region Summary")

        # Bottom: Selection summary and export
        bottom_group = QGroupBox("Selection Summary")
        bottom_layout = QVBoxLayout(bottom_group)

        self.selection_summary = QTextEdit()
        self.selection_summary.setMaximumHeight(100)
        self.selection_summary.setReadOnly(True)
        bottom_layout.addWidget(self.selection_summary)

        export_row = QHBoxLayout()

        self.export_data_btn = QPushButton("Export Data to Parquet")
        self.export_data_btn.clicked.connect(self.export_to_parquet)
        self.export_data_btn.setToolTip("Save loaded units to a Parquet file")
        export_row.addWidget(self.export_data_btn)

        self.export_yaml_btn = QPushButton("Export Config (YAML)")
        self.export_yaml_btn.clicked.connect(self.export_to_yaml)
        self.export_yaml_btn.setToolTip("Save selection criteria to a YAML config file")
        export_row.addWidget(self.export_yaml_btn)

        self.copy_selection_btn = QPushButton("Copy Selection")
        self.copy_selection_btn.clicked.connect(self.copy_selection)
        export_row.addWidget(self.copy_selection_btn)

        export_row.addStretch()
        bottom_layout.addLayout(export_row)

        layout.addWidget(bottom_group)

        return panel

    def setup_connections(self):
        """Setup signal connections."""
        pass

    # =========================================================================
    # Actions
    # =========================================================================

    def connect_to_one(self):
        """Connect to ONE API."""
        if not HAS_IBL:
            QMessageBox.critical(self, "Error",
                "IBL packages not installed.\n\nInstall with:\n  pip install ONE-api iblatlas")
            return

        self.connect_btn.setEnabled(False)
        self.conn_status.setText("🟡 Connecting...")
        self.status_bar.showMessage("Connecting to ONE API...")
        QApplication.processEvents()  # Force UI update

        mode = self.mode_combo.currentText()

        try:
            self.one = ONE(
                base_url=self.url_edit.text(),
                mode=mode,
            )
            self.conn_status.setText("🟢 Connected")
            self.load_sessions_btn.setEnabled(True)
            self.status_bar.showMessage(f"Connected to ONE API (mode={mode})")

        except Exception as e:
            self.conn_status.setText("🔴 Failed")
            error_msg = str(e)
            self.status_bar.showMessage(f"Connection failed: {error_msg}")

            QMessageBox.warning(self, "Connection Failed",
                f"Could not connect to ONE API.\n\n"
                f"Error: {error_msg}\n\n"
                f"If authentication is needed, run in terminal:\n"
                f"  python -c \"from one.api import ONE; ONE.setup()\"\n\n"
                f"Or try 'local' mode if you have cached data.")

        finally:
            self.connect_btn.setEnabled(True)

    def load_sessions(self):
        """Load sessions from ONE API."""
        if self.one is None:
            return

        self.load_sessions_btn.setEnabled(False)
        self.progress_bar.show()
        self.progress_bar.setValue(0)

        filters = {
            'lab': self.lab_combo.currentText() if self.lab_combo.currentText() != "All" else None,
            'subject': self.subject_edit.text() or None,
            'region': self.region_edit.text().strip() or None,
        }

        self.session_worker = SessionLoadWorker(self.one, filters)
        self.session_worker.progress.connect(self.on_load_progress)
        self.session_worker.finished.connect(self.on_sessions_loaded)
        self.session_worker.error.connect(self.on_load_error)
        self.session_worker.start()

    def on_load_progress(self, value: int, message: str):
        """Update progress bar."""
        self.progress_bar.setValue(value)
        self.status_bar.showMessage(message)

    def on_sessions_loaded(self, sessions: List[SessionInfo]):
        """Populate sessions table."""
        self.sessions = sessions
        self.sessions_table.setRowCount(len(sessions))

        for i, sess in enumerate(sessions):
            self.sessions_table.setItem(i, 0, QTableWidgetItem(sess.subject))
            self.sessions_table.setItem(i, 1, QTableWidgetItem(sess.date))
            self.sessions_table.setItem(i, 2, QTableWidgetItem(sess.lab))
            self.sessions_table.setItem(i, 3, QTableWidgetItem(str(sess.n_trials)))
            self.sessions_table.setItem(i, 4, QTableWidgetItem(f"{sess.performance:.1%}"))
            self.sessions_table.setItem(i, 5, QTableWidgetItem(", ".join(sess.probes)))

        self.load_sessions_btn.setEnabled(True)
        self.progress_bar.hide()
        self.status_bar.showMessage(f"Loaded {len(sessions)} sessions")

    def on_load_error(self, error: str):
        """Handle load error."""
        self.load_sessions_btn.setEnabled(True)
        self.progress_bar.hide()
        self.status_bar.showMessage(f"Error: {error}")
        QMessageBox.warning(self, "Load Error", error)

    def on_session_selected(self):
        """Handle session selection."""
        rows = self.sessions_table.selectionModel().selectedRows()
        if not rows:
            return

        row = rows[0].row()
        session = self.sessions[row]

        # Populate probes list
        self.probes_list.clear()
        for probe, pid in zip(session.probes, session.pids):
            item = QListWidgetItem(f"{probe} ({pid[:8]}...)")
            item.setData(Qt.ItemDataRole.UserRole, (pid, session.eid, probe))  # Added probe name
            self.probes_list.addItem(item)

        self.update_selection_summary()

    def on_probe_selected(self):
        """Handle probe selection - load units."""
        items = self.probes_list.selectedItems()
        if not items or self.one is None:
            return

        data = items[0].data(Qt.ItemDataRole.UserRole)
        pid, eid = data[0], data[1]
        probe_name = data[2] if len(data) > 2 else None

        self.progress_bar.show()
        self.progress_bar.setValue(0)

        self.units_worker = UnitsLoadWorker(self.one, pid, eid, probe_name)
        self.units_worker.progress.connect(self.on_load_progress)
        self.units_worker.finished.connect(self.on_units_loaded)
        self.units_worker.error.connect(self.on_load_error)
        self.units_worker.start()

    def expand_regions(self, acronyms):
        """
        Expand parent regions (e.g. OFC, VIS, HPF) into all Allen CCF descendants.
        """
        acronyms = [a.strip().upper() for a in acronyms if a and str(a).strip()]
        ids = self.br.acronym2id(acronyms)
        child_ids = self.br.descendants(ids)
        return set(self.br.id2acronym(child_ids))

    def on_units_loaded(self, units: List[UnitInfo], spikes: Dict):
        """Populate units table."""
        self.current_units = units
        self.current_spikes = spikes

        # Apply region filter (with parent expansion)
        region_filter = [r.strip().upper() for r in self.region_edit.text().split(',') if r.strip()]

        if region_filter:
            expanded_regions = self.expand_regions(region_filter)
            units = [u for u in units if u.acronym.upper() in expanded_regions]

        # Apply good units filter
        if self.good_only_check.isChecked():
            display_units = [u for u in units if u.label == 1]
        else:
            display_units = units

        self.units_table.setRowCount(len(display_units))

        for i, unit in enumerate(display_units):
            self.units_table.setItem(i, 0, QTableWidgetItem(str(unit.cluster_id)))
            self.units_table.setItem(i, 1, QTableWidgetItem(unit.acronym))
            self.units_table.setItem(i, 2, QTableWidgetItem(f"{unit.firing_rate:.2f}"))
            self.units_table.setItem(i, 3, QTableWidgetItem(str(unit.n_spikes)))

            label_item = QTableWidgetItem(str(unit.label))
            if unit.label == 1:
                label_item.setBackground(QColor(200, 255, 200))
            self.units_table.setItem(i, 4, label_item)

            self.units_table.setItem(i, 5, QTableWidgetItem(f"{unit.amp_median:.1f}"))
            self.units_table.setItem(i, 6, QTableWidgetItem(f"{unit.depth_um:.0f}"))

        # Update count label
        n_good = sum(1 for u in self.current_units if u.label == 1)
        self.unit_count_label.setText(
            f"Total: {len(self.current_units)} | Good: {n_good} | Displayed: {len(display_units)}"
        )

        # Update plots
        self.update_fr_histogram()
        self.update_region_summary()
        self.update_raster_plot()

        self.progress_bar.hide()
        self.update_selection_summary()

    def on_units_selected(self):
        """Handle unit selection."""
        rows = self.units_table.selectionModel().selectedRows()
        self.selected_units = [int(self.units_table.item(r.row(), 0).text()) for r in rows]
        self.update_selection_summary()
        self.update_raster_plot()

    def update_raster_plot(self):
        """Update spike raster plot."""
        self.raster_plot.clear()

        if self.current_spikes is None:
            return

        # Get time range
        t_start = self.raster_start.value()
        t_end = self.raster_end.value()

        # Get units to plot
        if self.selected_units:
            units_to_plot = self.selected_units[:20]  # Limit to 20
        else:
            units_to_plot = [u.cluster_id for u in self.current_units[:20]]

        spike_times = self.current_spikes.get('times', np.array([]))
        spike_clusters = self.current_spikes.get('clusters', np.array([]))

        for i, cluster_id in enumerate(units_to_plot):
            mask = (spike_clusters == cluster_id) & (spike_times >= t_start) & (spike_times <= t_end)
            times = spike_times[mask]

            if len(times) > 0:
                # Downsample if too many spikes
                if len(times) > 1000:
                    times = times[::len(times)//1000]

                y = np.ones_like(times) * i
                self.raster_plot.plot(times, y, pen=None, symbol='|',
                                      symbolSize=5, symbolBrush='b')

        self.raster_plot.setYRange(-0.5, len(units_to_plot) - 0.5)

    def update_fr_histogram(self):
        """Update firing rate histogram."""
        self.fr_plot.clear()

        if not self.current_units:
            return

        frs = [u.firing_rate for u in self.current_units if u.firing_rate > 0]
        if not frs:
            return

        # Log-scale histogram
        log_frs = np.log10(np.array(frs) + 0.1)
        hist, bins = np.histogram(log_frs, bins=30)

        # Use bin centers for bar plot
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_width = bins[1] - bins[0]

        # Create bar graph
        bargraph = pg.BarGraphItem(x=bin_centers, height=hist, width=bin_width * 0.9,
                                   brush=(100, 100, 200, 150))
        self.fr_plot.addItem(bargraph)

        # Set labels (x-axis is in log10 units)
        self.fr_plot.setLabel('bottom', 'Firing Rate (log10 Hz)')
        self.fr_plot.setLabel('left', 'Count')

    def update_region_summary(self):
        """Update region summary table."""
        if not self.current_units:
            return

        # Group by region
        regions = {}
        for unit in self.current_units:
            if unit.acronym not in regions:
                regions[unit.acronym] = {'total': 0, 'good': 0, 'frs': []}
            regions[unit.acronym]['total'] += 1
            if unit.label >= 0.66:
                regions[unit.acronym]['good'] += 1
            regions[unit.acronym]['frs'].append(unit.firing_rate)

        self.region_table.setRowCount(len(regions))

        for i, (region, data) in enumerate(sorted(regions.items())):
            self.region_table.setItem(i, 0, QTableWidgetItem(region))
            self.region_table.setItem(i, 1, QTableWidgetItem(str(data['total'])))
            self.region_table.setItem(i, 2, QTableWidgetItem(str(data['good'])))
            mean_fr = np.mean(data['frs']) if data['frs'] else 0
            self.region_table.setItem(i, 3, QTableWidgetItem(f"{mean_fr:.2f}"))

    def update_selection_summary(self):
        """Update selection summary text."""
        lines = []

        # Session info
        rows = self.sessions_table.selectionModel().selectedRows()
        if rows:
            sess = self.sessions[rows[0].row()]
            eid_str = str(sess.eid)
            lines.append(f"Session: {sess.subject} / {sess.date} ({eid_str[:8]}...)")

        # Probe info
        items = self.probes_list.selectedItems()
        if items:
            data = items[0].data(Qt.ItemDataRole.UserRole)
            pid = data[0]
            lines.append(f"Probe: {items[0].text()}")

        # Units info
        if self.current_units:
            n_good = sum(1 for u in self.current_units if u.label == 1)
            lines.append(f"Units: {len(self.current_units)} total, {n_good} good")

        if self.selected_units:
            lines.append(f"Selected: {len(self.selected_units)} units")

        self.selection_summary.setText("\n".join(lines))

    def export_to_yaml(self):
        """Export current selection to YAML config."""
        # Get current selection
        rows = self.sessions_table.selectionModel().selectedRows()
        if not rows:
            QMessageBox.warning(self, "No Selection", "Please select a session first.")
            return

        sess = self.sessions[rows[0].row()]

        # Build YAML content
        yaml_content = f"""# IBL Timescale Config - Generated {datetime.now().isoformat()}
# Session: {sess.subject} / {sess.date}

one:
  base_url: "{self.url_edit.text()}"
  mode: "auto"

dataset:
  subjects:
    include: ["{sess.subject}"]
  sessions:
    include: ["{sess.eid}"]
  regions:
    include: [{', '.join(f'"{r}"' for r in self.region_edit.text().split(',') if r.strip()) or '# all regions'}]

unit_selection:
  label: {1 if self.good_only_check.isChecked() else 'null'}

preprocessing:
  binning:
    bin_size_ms: 10
  alignment:
    event: "stimOn_times"
    pre_event_ms: -500
    post_event_ms: 1500
"""

        # Save dialog
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Config", "ibl_timescale_config.yaml", "YAML Files (*.yaml *.yml)"
        )

        if path:
            with open(path, 'w') as f:
                f.write(yaml_content)
            self.status_bar.showMessage(f"Exported config to {path}")

    def copy_selection(self):
        """Copy selection info to clipboard."""
        text = self.selection_summary.toPlainText()
        QApplication.clipboard().setText(text)
        self.status_bar.showMessage("Selection copied to clipboard")

    def export_to_parquet(self):
        """Export loaded units data to Parquet file."""
        if not self.current_units:
            QMessageBox.warning(self, "No Data", "Please load units from a probe first.")
            return

        # Get session info
        rows = self.sessions_table.selectionModel().selectedRows()
        if not rows:
            QMessageBox.warning(self, "No Session", "Please select a session.")
            return

        sess = self.sessions[rows[0].row()]

        # Get probe info
        probe_items = self.probes_list.selectedItems()
        probe_data = probe_items[0].data(Qt.ItemDataRole.UserRole) if probe_items else (None, None, None)
        pid = probe_data[0] if probe_data else "unknown"
        probe_name = probe_data[2] if len(probe_data) > 2 else "probe00"

        # Default filename and location
        default_dir = Path("./data")
        default_dir.mkdir(exist_ok=True)
        default_filename = f"{sess.subject}_{sess.date}_{probe_name}_units.parquet"
        default_path = default_dir / default_filename

        # Save dialog
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Units to Parquet",
            str(default_path),
            "Parquet Files (*.parquet);;All Files (*)"
        )

        if not path:
            return

        try:
            # Build DataFrame from current units
            data = []
            for unit in self.current_units:
                data.append({
                    # Identifiers - convert UUIDs to strings
                    "subject": str(sess.subject),
                    "eid": str(sess.eid),
                    "pid": str(pid),
                    "cluster_id": int(unit.cluster_id),

                    # Brain region
                    "acronym": str(unit.acronym),

                    # Session info
                    "lab": str(sess.lab),
                    "session_date": str(sess.date),
                    "probe_name": str(probe_name) if probe_name else "unknown",

                    # Metrics
                    "label": int(unit.label),
                    "n_spikes": int(unit.n_spikes),
                    "firing_rate_hz": float(unit.firing_rate),
                    "amp_median_uV": float(unit.amp_median),
                    "depth_um": float(unit.depth_um),

                    # Placeholder for timescale (to be computed later)
                    "tau_ms": None,
                    "qc_status": "pending",

                    # Metadata
                    "processing_timestamp": datetime.now().isoformat(),
                })

            df = pd.DataFrame(data)

            # Ensure directory exists
            Path(path).parent.mkdir(parents=True, exist_ok=True)

            # Save to parquet
            df.to_parquet(path, index=False)

            n_good = sum(1 for u in self.current_units if u.label == 1)
            self.status_bar.showMessage(
                f"Exported {len(self.current_units)} units ({n_good} good) to {path}"
            )

            QMessageBox.information(self, "Export Successful",
                f"Saved {len(self.current_units)} units to:\n{path}\n\n"
                f"Good units: {n_good}\n"
                f"Regions: {len(set(u.acronym for u in self.current_units))}")

        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Error saving file:\n{str(e)}")
            self.status_bar.showMessage(f"Export failed: {e}")

    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(self, "About IBL Data Loader",
            "IBL Timescale Data Loader\n\n"
            "Interactive tool for selecting and previewing\n"
            "IBL Brainwide Map data for timescale analysis.\n\n"
            "Version 1.0")


# =============================================================================
# Main
# =============================================================================

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # Set dark theme (optional)
    # palette = app.palette()
    # palette.setColor(palette.ColorRole.Window, QColor(53, 53, 53))
    # app.setPalette(palette)

    window = IBLDataLoaderGUI()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()