"""
IBL Timescale Data Loader GUI
==============================

PyQt6 desktop application for interactively selecting and previewing
IBL Brainwide Map data before loading for timescale analysis.

Features:
- Browse subjects, sessions, and probes via ONE API
- Filter by brain region (IBL's 223 canonical regions), lab, date range
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
from typing import Optional, List, Dict, Any, Set, Tuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QTreeWidget, QTreeWidgetItem, QTableWidget, QTableWidgetItem,
    QGroupBox, QLabel, QLineEdit, QComboBox, QPushButton, QCheckBox,
    QSpinBox, QDoubleSpinBox, QProgressBar, QStatusBar, QTabWidget,
    QTextEdit, QFileDialog, QMessageBox, QHeaderView, QAbstractItemView,
    QListWidget, QListWidgetItem, QDialog, QDialogButtonBox, QGridLayout
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QColor, QAction

import pyqtgraph as pg

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
# IBL Canonical 223 Brain Regions (Beryl mapping)
# =============================================================================

IBL_CANONICAL_REGIONS = {
    "Isocortex - Visual": [
        "VISa", "VISam", "VISl", "VISp", "VISpl", "VISpm", "VISli", "VISrl", "VISpor",
    ],
    "Isocortex - Somatosensory": [
        "SSp-bfd", "SSp-ll", "SSp-m", "SSp-n", "SSp-tr", "SSp-ul", "SSp-un", "SSs",
    ],
    "Isocortex - Motor": [
        "MOp", "MOs",
    ],
    "Isocortex - Prefrontal": [
        "FRP", "PL", "ILA", "ORBl", "ORBm", "ORBvl", "ACAd", "ACAv",
    ],
    "Isocortex - Retrosplenial": [
        "RSPagl", "RSPd", "RSPv",
    ],
    "Isocortex - Auditory/Temporal": [
        "AUDd", "AUDp", "AUDpo", "AUDv", "TEa", "VISC", "GU", "ECT", "PERI",
    ],
    "Isocortex - Parietal/Association": [
        "PTLp", "AId", "AIp", "AIv",
    ],
    "Olfactory Areas": [
        "AOB", "AON", "TT", "DP", "PIR", "NLOT", "COA", "PAA", "TR",
    ],
    "Hippocampal Formation": [
        "CA1", "CA2", "CA3", "DG", "FC", "IG", "ENTl", "ENTm", "PAR", "POST", "PRE", "SUB", "ProS",
    ],
    "Cortical Subplate / Amygdala": [
        "CLA", "EP", "LA", "BLA", "BMA", "PA",
    ],
    "Striatum": [
        "CP", "ACB", "FS", "OT", "LSr", "LSc", "LSv", "SF", "SH",
    ],
    "Pallidum": [
        "GPe", "GPi", "SI", "MA", "NDB", "TRS", "BST", "BAC",
    ],
    "Thalamus - Sensory": [
        "VAL", "VM", "VPL", "VPLpc", "VPM", "VPMpc", "PoT", "SPF", "SPFm", "SPFp", "SPA",
        "PP", "MGd", "MGm", "MGv", "LGd", "LGv", "LP", "PO", "POL", "SGN",
    ],
    "Thalamus - Limbic/Association": [
        "AD", "AM", "AV", "IAD", "IAM", "LD", "MD", "MDc", "MDl", "MDm",
        "IMD", "SMT", "PR", "PVT", "PT", "RE", "RH", "CM", "PCN", "CL", "PF", "PIL",
    ],
    "Thalamus - Other": [
        "RT", "IGL", "IntG", "SubG", "EPI", "MH", "LH",
    ],
    "Hypothalamus": [
        "SO", "ASO", "PVH", "PVHd", "ARH", "ADP", "AVP", "AVPV", "DMH", "MEPO",
        "MPO", "OV", "PD", "PS", "PVa", "PVi", "PVpo", "SBPV", "SCH", "SFO", "VMPO",
        "AHN", "MBO", "MM", "SUM", "TM", "LM", "LPO", "PMd", "PMv", "VMH", "PH", "LHA", "STN", "ZI",
    ],
    "Midbrain - Basal Ganglia": [
        "SNr", "SNc", "VTA", "RR",
    ],
    "Midbrain - Superior/Inferior Colliculus": [
        "SCm", "SCs", "SCig", "SCiw", "SCsg", "SCzo",
        "ICc", "ICd", "ICe",
    ],
    "Midbrain - Other": [
        "MRN", "NB", "SAG", "PBG", "MEV", "APN", "NOT", "NPC", "OP", "PPT", "RPF",
        "CUN", "RN", "III", "IV", "VTN", "PAG", "PRT", "EW", "DR", "IF", "IPN", "RL",
    ],
    "Pons": [
        "NLL", "PSV", "PB", "KF", "SOC", "POR", "DTN", "LDT", "PCG", "PG", "PRNc", "PRNr",
        "SG", "SUT", "TRN", "V", "CS", "LC", "NI",
    ],
    "Medulla": [
        "AP", "CN", "DCN", "ECU", "CU", "GR", "NTB", "NTS", "DMX", "SPVI", "SPVC",
        "SPVO", "IO", "LRN", "MDRN", "PARN", "PRP", "VNC",
        "XII", "AMB", "RO", "RPA", "RM", "NR", "VI", "VII",
    ],
    "Cerebellum - Vermis": [
        "CENT", "CUL", "DEC", "FOTU", "PYR", "UVU", "NOD",
    ],
    "Cerebellum - Hemisphere": [
        "PRM", "COPY", "PFL", "FL", "SIM", "AN",
    ],
    "Cerebellum - Nuclei": [
        "IP", "FN", "DN",
    ],
}

def get_ibl_canonical_regions_with_names() -> List[Tuple[str, str, str]]:
    """Get canonical regions with full names and categories."""
    try:
        br = BrainRegions()
        results = []
        for category, acronyms in IBL_CANONICAL_REGIONS.items():
            for acronym in acronyms:
                try:
                    idx = br.acronym2index([acronym])[0]
                    if len(idx) > 0 and idx[0] >= 0:
                        name = br.name[idx[0]]
                    else:
                        name = acronym
                except Exception:
                    name = acronym
                results.append((acronym, name, category))
        return results
    except Exception:
        return [(a, a, cat) for cat, acrs in IBL_CANONICAL_REGIONS.items() for a in acrs]


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SessionInfo:
    eid: str
    subject: str
    date: str
    lab: str
    n_trials: int
    performance: float
    probes: List[str] = field(default_factory=list)
    pids: List[str] = field(default_factory=list)

@dataclass
class UnitInfo:
    cluster_id: int
    pid: str
    acronym: str
    firing_rate: float
    n_spikes: int
    label: int
    amp_median: float
    depth_um: float


# =============================================================================
# Region Browser Dialog
# =============================================================================

class RegionBrowserDialog(QDialog):
    """Dialog for browsing and selecting from IBL's canonical regions."""
    def __init__(self, parent=None, current_selection: str = ""):
        super().__init__(parent)
        self.setWindowTitle("IBL Canonical Brain Regions")
        self.setMinimumSize(700, 600)
        self.selected_regions: List[str] = []
        if current_selection:
            self.selected_regions = [r.strip() for r in current_selection.split(",") if r.strip()]
        self.init_ui()
        self.populate_regions()

    def init_ui(self):
        layout = QVBoxLayout(self)

        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("Search:"))
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Type to filter (e.g., CA, VIS, motor)...")
        self.search_edit.textChanged.connect(self.filter_regions)
        search_layout.addWidget(self.search_edit)
        layout.addLayout(search_layout)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        self.region_tree = QTreeWidget()
        self.region_tree.setHeaderLabels(["Region", "Full Name"])
        self.region_tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.region_tree.itemDoubleClicked.connect(self.add_selected_to_list)
        self.region_tree.setColumnWidth(0, 130)
        splitter.addWidget(self.region_tree)

        selection_widget = QWidget()
        selection_layout = QVBoxLayout(selection_widget)

        selection_layout.addWidget(QLabel("Selected Regions:"))
        self.selection_list = QListWidget()
        self.selection_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        selection_layout.addWidget(self.selection_list)

        for region in self.selected_regions:
            self.selection_list.addItem(region)

        btn_layout = QHBoxLayout()
        add_btn = QPushButton("Add →")
        add_btn.clicked.connect(self.add_selected_to_list)
        btn_layout.addWidget(add_btn)

        remove_btn = QPushButton("← Remove")
        remove_btn.clicked.connect(self.remove_selected)
        btn_layout.addWidget(remove_btn)

        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(self.clear_selection)
        btn_layout.addWidget(clear_btn)
        selection_layout.addLayout(btn_layout)

        quick_layout = QGridLayout()
        quick_regions = [
            ("Visual Cortex", ["VISp", "VISl", "VISa", "VISam", "VISpm"]),
            ("Motor", ["MOp", "MOs"]),
            ("Prefrontal", ["PL", "ILA", "ORBl", "ORBm", "ACAd", "ACAv"]),
            ("Hippocampus", ["CA1", "CA2", "CA3", "DG", "SUB"]),
        ]
        for i, (name, regions) in enumerate(quick_regions):
            btn = QPushButton(name)
            btn.setToolTip(", ".join(regions))
            btn.clicked.connect(lambda checked, r=regions: self.add_regions(r))
            quick_layout.addWidget(btn, i // 2, i % 2)

        selection_layout.addWidget(QLabel("Quick Select:"))
        selection_layout.addLayout(quick_layout)

        splitter.addWidget(selection_widget)
        splitter.setSizes([420, 280])
        layout.addWidget(splitter)

        self.count_label = QLabel("0 regions selected")
        layout.addWidget(self.count_label)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.update_count()

    def populate_regions(self):
        self.region_tree.clear()
        regions_data = get_ibl_canonical_regions_with_names()

        by_cat: Dict[str, List[Tuple[str, str]]] = {}
        for acronym, name, category in regions_data:
            by_cat.setdefault(category, []).append((acronym, name))

        for category, regions in by_cat.items():
            category_item = QTreeWidgetItem([category, f"({len(regions)} regions)"])
            category_item.setFlags(category_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            font = category_item.font(0)
            font.setBold(True)
            category_item.setFont(0, font)

            for acronym, name in regions:
                child = QTreeWidgetItem([acronym, name])
                child.setData(0, Qt.ItemDataRole.UserRole, acronym)
                category_item.addChild(child)

            self.region_tree.addTopLevelItem(category_item)

        self.region_tree.expandAll()

    def filter_regions(self, text: str):
        text = text.strip().lower()
        for i in range(self.region_tree.topLevelItemCount()):
            cat = self.region_tree.topLevelItem(i)
            visible = 0
            for j in range(cat.childCount()):
                item = cat.child(j)
                acronym = item.text(0).lower()
                name = item.text(1).lower()
                ok = (not text) or (text in acronym) or (text in name)
                item.setHidden(not ok)
                visible += int(ok)
            cat.setHidden(visible == 0)

    def add_selected_to_list(self):
        for item in self.region_tree.selectedItems():
            acronym = item.data(0, Qt.ItemDataRole.UserRole)
            if acronym and acronym not in self.selected_regions:
                self.selected_regions.append(acronym)
                self.selection_list.addItem(acronym)
        self.update_count()

    def add_regions(self, regions: List[str]):
        for r in regions:
            if r not in self.selected_regions:
                self.selected_regions.append(r)
                self.selection_list.addItem(r)
        self.update_count()

    def remove_selected(self):
        for item in self.selection_list.selectedItems():
            self.selected_regions.remove(item.text())
            self.selection_list.takeItem(self.selection_list.row(item))
        self.update_count()

    def clear_selection(self):
        self.selected_regions.clear()
        self.selection_list.clear()
        self.update_count()

    def update_count(self):
        n = len(self.selected_regions)
        self.count_label.setText(f"{n} region{'s' if n != 1 else ''} selected")

    def get_selection(self) -> str:
        return ", ".join(self.selected_regions)


# =============================================================================
# Worker Threads
# =============================================================================

class SessionLoadWorker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, one: 'ONE', filters: Dict[str, Any]):
        super().__init__()
        self.one = one
        self.filters = filters
        self.br = BrainRegions() if HAS_IBL else None

    def _expand_regions(self, acronyms: List[str], include_descendants: bool) -> Set[int]:
        if not self.br or not acronyms:
            return set()
        acronyms = [a.strip() for a in acronyms if a and str(a).strip()]
        if not acronyms:
            return set()

        try:
            ids = self.br.acronym2id(acronyms)
            ids = np.atleast_1d(ids)
            ids = ids[~np.isnan(ids)].astype(int)

            if not include_descendants:
                return set(ids.tolist())

            all_ids: Set[int] = set()
            for region_id in ids:
                desc = self.br.descendants(np.array([region_id]))
                all_ids.update(desc.astype(int).tolist())
            return all_ids
        except Exception as e:
            print(f"Warning: Region expansion failed: {e}")
            return set()

    def _probe_hits_region(self, eid: str, probe_name: str, target_ids: Set[int]) -> bool:
        if not target_ids:
            return True

        try:
            ids = self.one.load_dataset(
                eid,
                "channels.brainLocationIds_ccf_2017.npy",
                collection=f"alf/{probe_name}"
            )
            ids = np.asarray(ids)
            ids = ids[~np.isnan(ids)].astype(int)
            return any(i in target_ids for i in ids)
        except Exception:
            pass

        try:
            acr = self.one.load_dataset(
                eid,
                "channels.acronym.npy",
                collection=f"alf/{probe_name}"
            )
            acr = np.asarray(acr)
            acr = [str(a) for a in acr if a and str(a).strip()]
            if acr and self.br:
                channel_ids = self.br.acronym2id(acr)
                channel_ids = channel_ids[~np.isnan(channel_ids)].astype(int)
                return any(i in target_ids for i in channel_ids)
        except Exception:
            pass

        return False

    def run(self):
        try:
            self.progress.emit(10, "Querying sessions...")

            query_params = {
                "task_protocol": "ephysChoiceWorld",
                "project": "brainwide",
            }

            if self.filters.get("lab"):
                query_params["lab"] = self.filters["lab"]
            if self.filters.get("subject"):
                query_params["subject"] = self.filters["subject"]

            region_str = self.filters.get("region", "") or ""
            region_list = [r.strip() for r in region_str.split(",") if r.strip()]
            include_descendants = bool(self.filters.get("include_descendants", True))
            target_ids = self._expand_regions(region_list, include_descendants) if region_list else set()

            eids = self.one.search(**query_params)
            self.progress.emit(30, f"Found {len(eids)} sessions, filtering by region...")

            sessions: List[SessionInfo] = []
            max_sessions = 200
            checked = 0

            for eid in eids:
                if len(sessions) >= max_sessions:
                    break

                try:
                    sess_info = self.one.get_details(eid)
                    insertions = self.one.alyx.rest("insertions", "list", session=eid)
                    probes = [ins.get("name", "unknown") for ins in insertions]
                    pids = [ins.get("id", None) for ins in insertions]

                    if target_ids:
                        ok = False
                        for probe_name in probes:
                            if probe_name != "unknown" and self._probe_hits_region(eid, probe_name, target_ids):
                                ok = True
                                break
                        if not ok:
                            checked += 1
                            continue

                    try:
                        trials = self.one.load_object(eid, "trials", collection="alf")
                        n_trials = len(trials.get("choice", []))
                        correct = trials.get("feedbackType", [])
                        performance = float(np.mean(np.asarray(correct) == 1)) if len(correct) > 0 else 0.0
                    except Exception:
                        n_trials, performance = 0, 0.0

                    sessions.append(SessionInfo(
                        eid=eid,
                        subject=sess_info.get("subject", "unknown"),
                        date=str(sess_info.get("start_time", ""))[:10],
                        lab=sess_info.get("lab", "unknown"),
                        n_trials=int(n_trials),
                        performance=float(performance),
                        probes=probes,
                        pids=pids,
                    ))
                except Exception:
                    pass

                checked += 1
                self.progress.emit(
                    30 + int(60 * checked / max(1, min(len(eids), max_sessions * 2))),
                    f"Checked {checked} sessions, found {len(sessions)} with target regions"
                )

            self.progress.emit(100, f"Loaded {len(sessions)} sessions")
            self.finished.emit(sessions)

        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n\n{traceback.format_exc()}")


class UnitsLoadWorker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(list, object)
    error = pyqtSignal(str)

    def __init__(self, one: 'ONE', pid: str, eid: str, probe_name: str = None):
        super().__init__()
        self.one = one
        self.pid = pid
        self.eid = eid
        self.probe_name = probe_name

    def run(self):
        try:
            collection = f"alf/{self.probe_name}" if self.probe_name else None
            self.progress.emit(10, f"Finding spike data in {collection or 'default'}...")

            try:
                datasets = self.one.list_datasets(self.eid)
            except Exception:
                datasets = []

            spikes = None
            load_errors = []

            if spikes is None and collection:
                try:
                    spikes = self.one.load_object(self.eid, "spikes", collection=collection)
                except Exception as e:
                    load_errors.append(f"collection={collection}: {e}")

            if spikes is None:
                try:
                    spikes = self.one.load_object(self.eid, "spikes")
                except Exception as e:
                    load_errors.append(f"no collection: {e}")

            if spikes is None:
                error_details = "\n".join(load_errors) if load_errors else "(no details)"
                self.error.emit(f"Failed to load spikes\n\nTried approaches:\n{error_details}")
                return

            self.progress.emit(50, "Loading clusters/channels...")

            clusters = {}
            channels = {}

            if collection:
                try:
                    clusters = self.one.load_object(self.eid, "clusters", collection=collection) or {}
                except Exception:
                    clusters = {}
                try:
                    channels = self.one.load_object(self.eid, "channels", collection=collection) or {}
                except Exception:
                    channels = {}
            else:
                try:
                    clusters = self.one.load_object(self.eid, "clusters") or {}
                except Exception:
                    clusters = {}
                try:
                    channels = self.one.load_object(self.eid, "channels") or {}
                except Exception:
                    channels = {}

            self.progress.emit(70, "Processing units...")

            spike_times = np.asarray(spikes.get("times", np.array([])))
            spike_clusters = np.asarray(spikes.get("clusters", np.array([])))
            unique_clusters = np.unique(spike_clusters)

            cluster_amps = clusters.get("amps", None)
            cluster_depths = clusters.get("depths", None)
            cluster_channels = clusters.get("channels", None)
            cluster_metrics = clusters.get("metrics", None)

            br = BrainRegions()
            ccf_ids = channels.get("brainLocationIds_ccf_2017", None)
            channel_acronyms = br.id2acronym(ccf_ids).astype(str) if ccf_ids is not None else None

            metrics_df = None
            if cluster_metrics is not None:
                if isinstance(cluster_metrics, pd.DataFrame):
                    metrics_df = cluster_metrics
                elif hasattr(cluster_metrics, "to_frame"):
                    metrics_df = cluster_metrics.to_frame()

            units: List[UnitInfo] = []

            for cluster_id in unique_clusters:
                cluster_id = int(cluster_id)

                mask = (spike_clusters == cluster_id)
                n_spikes = int(np.sum(mask))
                t = spike_times[mask]

                if len(t) > 1:
                    duration = float(t[-1] - t[0])
                    firing_rate = float(n_spikes / duration) if duration > 0 else 0.0
                else:
                    firing_rate = 0.0

                depth = float(cluster_depths[cluster_id]) if (cluster_depths is not None and cluster_id < len(cluster_depths)) else 0.0
                amp_median = float(cluster_amps[cluster_id]) * 1e6 if (cluster_amps is not None and cluster_id < len(cluster_amps)) else 0.0
                peak_channel = int(cluster_channels[cluster_id]) if (cluster_channels is not None and cluster_id < len(cluster_channels)) else 0

                acronym = "unknown"
                if channel_acronyms is not None and 0 <= peak_channel < len(channel_acronyms):
                    acronym = str(channel_acronyms[peak_channel])

                label = 0
                if metrics_df is not None:
                    try:
                        if "cluster_id" in metrics_df.columns:
                            row = metrics_df[metrics_df["cluster_id"] == cluster_id]
                            row = row.iloc[0] if len(row) else None
                        else:
                            row = metrics_df.iloc[cluster_id] if cluster_id < len(metrics_df) else None

                        if row is not None:
                            if "label" in row.index and pd.notna(row["label"]):
                                v = row["label"]
                                if isinstance(v, str):
                                    label = 1 if v.lower().strip() == "good" else 0
                                else:
                                    label = 1 if float(v) >= 1.0 else 0
                            elif "ks2_label" in row.index and pd.notna(row["ks2_label"]):
                                label = 1 if str(row["ks2_label"]).lower().strip() == "good" else 0
                            elif "bitwise_label" in row.index and pd.notna(row["bitwise_label"]):
                                label = 1 if int(row["bitwise_label"]) == 1 else 0

                            if "firing_rate" in row.index and pd.notna(row["firing_rate"]):
                                firing_rate = float(row["firing_rate"])
                    except Exception:
                        pass

                units.append(UnitInfo(
                    cluster_id=cluster_id,
                    pid=self.pid,
                    acronym=acronym,
                    firing_rate=firing_rate,
                    n_spikes=n_spikes,
                    label=label,
                    amp_median=amp_median,
                    depth_um=depth,
                ))

            self.progress.emit(100, f"Loaded {len(units)} units")
            self.finished.emit(units, {"times": spike_times, "clusters": spike_clusters})

        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n\n{traceback.format_exc()}")


# =============================================================================
# Main Window
# =============================================================================

class IBLDataLoaderGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.br = BrainRegions() if HAS_IBL else None
        self.one: Optional["ONE"] = None
        self.sessions: List[SessionInfo] = []
        self.current_units: List[UnitInfo] = []
        self.current_spikes: Optional[Dict] = None
        self.selected_units: List[int] = []

        self.init_ui()
        self.status_bar.showMessage("Ready - Select mode and click 'Connect'")

    def init_ui(self):
        self.setWindowTitle("IBL Timescale Data Loader")
        self.setGeometry(100, 100, 1400, 900)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        splitter.addWidget(self.create_selection_panel())
        splitter.addWidget(self.create_preview_panel())
        splitter.setSizes([500, 900])

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.hide()
        self.status_bar.addPermanentWidget(self.progress_bar)

        self.create_menu_bar()

    def create_menu_bar(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")

        export_parquet_action = QAction("Export Selection to Parquet", self)
        export_parquet_action.triggered.connect(self.export_to_parquet)
        file_menu.addAction(export_parquet_action)

        export_yaml_action = QAction("Export Selection to YAML", self)
        export_yaml_action.triggered.connect(self.export_to_yaml)
        file_menu.addAction(export_yaml_action)

        file_menu.addSeparator()
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        help_menu = menubar.addMenu("Help")
        regions_action = QAction("View Canonical Regions", self)
        regions_action.triggered.connect(self.show_region_browser)
        help_menu.addAction(regions_action)

        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    # -------------------------------------------------------------------------
    # KEY CHANGE: Lab dropdown is now dynamic (any lab), not hardcoded
    # -------------------------------------------------------------------------

    def create_selection_panel(self) -> QWidget:
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

        lab_row = QHBoxLayout()
        lab_row.addWidget(QLabel("Lab:"))
        self.lab_combo = QComboBox()
        # start empty; we fill after connecting
        self.lab_combo.addItems(["All (connect to load labs)"])
        self.lab_combo.setEnabled(False)
        lab_row.addWidget(self.lab_combo)
        filter_layout.addLayout(lab_row)

        subj_row = QHBoxLayout()
        subj_row.addWidget(QLabel("Subject:"))
        self.subject_edit = QLineEdit()
        self.subject_edit.setPlaceholderText("e.g., KS001 (leave empty for all)")
        subj_row.addWidget(self.subject_edit)
        filter_layout.addLayout(subj_row)

        region_row = QHBoxLayout()
        region_row.addWidget(QLabel("Region:"))
        self.region_edit = QLineEdit()
        self.region_edit.setPlaceholderText("e.g., CA1, VISp (comma-separated)")
        self.region_edit.setToolTip("Enter IBL canonical region acronyms")
        region_row.addWidget(self.region_edit)

        self.browse_regions_btn = QPushButton("Browse...")
        self.browse_regions_btn.setMaximumWidth(90)
        self.browse_regions_btn.clicked.connect(self.show_region_browser)
        self.browse_regions_btn.setToolTip("Browse canonical brain regions")
        region_row.addWidget(self.browse_regions_btn)
        filter_layout.addLayout(region_row)

        self.include_descendants_check = QCheckBox("Include child regions (e.g., VIS → VISp, VISl, ...)")
        self.include_descendants_check.setChecked(True)
        filter_layout.addWidget(self.include_descendants_check)

        self.good_only_check = QCheckBox("Good units only (label=1)")
        self.good_only_check.setChecked(True)
        filter_layout.addWidget(self.good_only_check)

        self.load_sessions_btn = QPushButton("Load Sessions")
        self.load_sessions_btn.clicked.connect(self.load_sessions)
        self.load_sessions_btn.setEnabled(False)
        filter_layout.addWidget(self.load_sessions_btn)

        layout.addWidget(filter_group)

        sessions_group = QGroupBox("Sessions")
        sessions_layout = QVBoxLayout(sessions_group)

        self.sessions_table = QTableWidget()
        self.sessions_table.setColumnCount(6)
        self.sessions_table.setHorizontalHeaderLabels(["Subject", "Date", "Lab", "Trials", "Perf", "Probes"])
        self.sessions_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.sessions_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.sessions_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.sessions_table.itemSelectionChanged.connect(self.on_session_selected)
        sessions_layout.addWidget(self.sessions_table)
        layout.addWidget(sessions_group)

        probes_group = QGroupBox("Probes")
        probes_layout = QVBoxLayout(probes_group)
        self.probes_list = QListWidget()
        self.probes_list.itemSelectionChanged.connect(self.on_probe_selected)
        probes_layout.addWidget(self.probes_list)
        layout.addWidget(probes_group)

        return panel

    def create_preview_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        tabs = QTabWidget()
        layout.addWidget(tabs)

        # Units table
        units_widget = QWidget()
        units_layout = QVBoxLayout(units_widget)
        self.units_table = QTableWidget()
        self.units_table.setColumnCount(7)
        self.units_table.setHorizontalHeaderLabels(["ID", "Region", "FR (Hz)", "Spikes", "Label", "Amp (µV)", "Depth (µm)"])
        self.units_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.units_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.units_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.units_table.itemSelectionChanged.connect(self.on_units_selected)
        units_layout.addWidget(self.units_table)

        self.unit_count_label = QLabel("No units loaded")
        units_layout.addWidget(self.unit_count_label)
        tabs.addTab(units_widget, "Units Table")

        # Raster
        raster_widget = QWidget()
        raster_layout = QVBoxLayout(raster_widget)
        self.raster_plot = pg.PlotWidget(title="Spike Raster")
        self.raster_plot.setLabel("left", "Unit")
        self.raster_plot.setLabel("bottom", "Time (s)")
        raster_layout.addWidget(self.raster_plot)

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

        # FR dist
        fr_widget = QWidget()
        fr_layout = QVBoxLayout(fr_widget)
        self.fr_plot = pg.PlotWidget(title="Firing Rate Distribution")
        self.fr_plot.setLabel("left", "Count")
        self.fr_plot.setLabel("bottom", "Firing Rate (Hz)")
        fr_layout.addWidget(self.fr_plot)
        tabs.addTab(fr_widget, "FR Distribution")

        # Region summary
        region_widget = QWidget()
        region_layout = QVBoxLayout(region_widget)
        self.region_table = QTableWidget()
        self.region_table.setColumnCount(4)
        self.region_table.setHorizontalHeaderLabels(["Region", "Total", "Good", "Mean FR"])
        self.region_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        region_layout.addWidget(self.region_table)
        tabs.addTab(region_widget, "Region Summary")

        bottom_group = QGroupBox("Selection Summary")
        bottom_layout = QVBoxLayout(bottom_group)

        self.selection_summary = QTextEdit()
        self.selection_summary.setMaximumHeight(100)
        self.selection_summary.setReadOnly(True)
        bottom_layout.addWidget(self.selection_summary)

        export_row = QHBoxLayout()
        self.export_data_btn = QPushButton("Export Data to Parquet")
        self.export_data_btn.clicked.connect(self.export_to_parquet)
        export_row.addWidget(self.export_data_btn)

        self.export_yaml_btn = QPushButton("Export Config (YAML)")
        self.export_yaml_btn.clicked.connect(self.export_to_yaml)
        export_row.addWidget(self.export_yaml_btn)

        self.copy_selection_btn = QPushButton("Copy Selection")
        self.copy_selection_btn.clicked.connect(self.copy_selection)
        export_row.addWidget(self.copy_selection_btn)

        export_row.addStretch()
        bottom_layout.addLayout(export_row)
        layout.addWidget(bottom_group)

        return panel

    # -------------------------------------------------------------------------
    # Dynamic lab loading
    # -------------------------------------------------------------------------

    def populate_labs(self):
        """
        Populate lab dropdown with *all* labs available from Alyx.
        Called after successful ONE connection.
        """
        if self.one is None:
            return

        self.lab_combo.blockSignals(True)
        self.lab_combo.clear()

        labs: List[str] = []
        try:
            # Alyx 'labs' endpoint returns a list of lab objects (usually with 'name')
            lab_objs = self.one.alyx.rest("labs", "list")
            for obj in lab_objs:
                name = obj.get("name", None)
                if name:
                    labs.append(str(name))
        except Exception:
            labs = []

        labs = sorted(set(labs), key=lambda s: s.lower())

        # Always include "All"
        self.lab_combo.addItem("All")
        if labs:
            self.lab_combo.addItems(labs)
            self.lab_combo.setEnabled(True)
        else:
            # still usable; user can type a lab name in code if needed, but UI shows fallback
            self.lab_combo.addItem("(could not load labs)")
            self.lab_combo.setEnabled(True)

        self.lab_combo.blockSignals(False)

    # -------------------------------------------------------------------------
    # Region browser
    # -------------------------------------------------------------------------

    def show_region_browser(self):
        dialog = RegionBrowserDialog(self, self.region_edit.text())
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.region_edit.setText(dialog.get_selection())

    # -------------------------------------------------------------------------
    # Actions
    # -------------------------------------------------------------------------

    def connect_to_one(self):
        if not HAS_IBL:
            QMessageBox.critical(self, "Error",
                                 "IBL packages not installed.\n\nInstall with:\n  pip install ONE-api iblatlas")
            return

        self.connect_btn.setEnabled(False)
        self.conn_status.setText("🟡 Connecting...")
        self.status_bar.showMessage("Connecting to ONE API...")
        QApplication.processEvents()

        mode = self.mode_combo.currentText()

        try:
            self.one = ONE(base_url=self.url_edit.text(), mode=mode)
            self.conn_status.setText("🟢 Connected")
            self.load_sessions_btn.setEnabled(True)
            self.status_bar.showMessage(f"Connected to ONE API (mode={mode})")

            # NEW: load labs dynamically
            self.populate_labs()

        except Exception as e:
            self.conn_status.setText("🔴 Failed")
            self.status_bar.showMessage(f"Connection failed: {e}")
            QMessageBox.warning(
                self, "Connection Failed",
                f"Could not connect to ONE API.\n\nError: {e}\n\n"
                f"If authentication is needed, run in terminal:\n"
                f"  python -c \"from one.api import ONE; ONE.setup()\"\n"
                f"Or try 'local' mode if you have cached data."
            )
        finally:
            self.connect_btn.setEnabled(True)

    def load_sessions(self):
        if self.one is None:
            return

        self.load_sessions_btn.setEnabled(False)
        self.progress_bar.show()
        self.progress_bar.setValue(0)

        lab_val = self.lab_combo.currentText()
        lab_filter = None if (lab_val == "All" or lab_val.startswith("(")) else lab_val

        filters = {
            "lab": lab_filter,
            "subject": self.subject_edit.text().strip() or None,
            "region": self.region_edit.text().strip() or None,
            "include_descendants": self.include_descendants_check.isChecked(),
        }

        self.session_worker = SessionLoadWorker(self.one, filters)
        self.session_worker.progress.connect(self.on_load_progress)
        self.session_worker.finished.connect(self.on_sessions_loaded)
        self.session_worker.error.connect(self.on_load_error)
        self.session_worker.start()

    def on_load_progress(self, value: int, message: str):
        self.progress_bar.setValue(value)
        self.status_bar.showMessage(message)

    def on_sessions_loaded(self, sessions: List[SessionInfo]):
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
        self.load_sessions_btn.setEnabled(True)
        self.progress_bar.hide()
        self.status_bar.showMessage(f"Error: {error}")
        QMessageBox.warning(self, "Load Error", error)

    def on_session_selected(self):
        rows = self.sessions_table.selectionModel().selectedRows()
        if not rows:
            return

        sess = self.sessions[rows[0].row()]
        self.probes_list.clear()
        for probe, pid in zip(sess.probes, sess.pids):
            item = QListWidgetItem(f"{probe} ({str(pid)[:8]}...)")
            item.setData(Qt.ItemDataRole.UserRole, (pid, sess.eid, probe))
            self.probes_list.addItem(item)

        self.update_selection_summary()

    def on_probe_selected(self):
        items = self.probes_list.selectedItems()
        if not items or self.one is None:
            return

        pid, eid, probe_name = items[0].data(Qt.ItemDataRole.UserRole)

        self.progress_bar.show()
        self.progress_bar.setValue(0)

        self.units_worker = UnitsLoadWorker(self.one, pid, eid, probe_name)
        self.units_worker.progress.connect(self.on_load_progress)
        self.units_worker.finished.connect(self.on_units_loaded)
        self.units_worker.error.connect(self.on_load_error)
        self.units_worker.start()

    def expand_regions(self, acronyms: List[str]) -> Set[str]:
        if not self.br or not acronyms:
            return set(a.upper() for a in acronyms)

        acronyms = [a.strip() for a in acronyms if a and str(a).strip()]
        if not acronyms:
            return set()

        try:
            ids = self.br.acronym2id(acronyms)
            ids = np.atleast_1d(ids)
            ids = ids[~np.isnan(ids)].astype(int)

            all_ids: Set[int] = set()
            for region_id in ids:
                desc = self.br.descendants(np.array([region_id]))
                all_ids.update(desc.astype(int).tolist())

            all_acronyms = set(self.br.id2acronym(np.array(list(all_ids), dtype=int)))
            return {str(a).upper() for a in all_acronyms}
        except Exception:
            return set(a.upper() for a in acronyms)

    def on_units_loaded(self, units: List[UnitInfo], spikes: Dict):
        self.current_units = units
        self.current_spikes = spikes

        region_filter = [r.strip() for r in self.region_edit.text().split(",") if r.strip()]
        if region_filter:
            if self.include_descendants_check.isChecked():
                expanded = self.expand_regions(region_filter)
            else:
                expanded = set(r.upper() for r in region_filter)
            units = [u for u in units if u.acronym.upper() in expanded]

        display_units = [u for u in units if u.label == 1] if self.good_only_check.isChecked() else units

        self.units_table.setRowCount(len(display_units))
        for i, u in enumerate(display_units):
            self.units_table.setItem(i, 0, QTableWidgetItem(str(u.cluster_id)))
            self.units_table.setItem(i, 1, QTableWidgetItem(u.acronym))
            self.units_table.setItem(i, 2, QTableWidgetItem(f"{u.firing_rate:.2f}"))
            self.units_table.setItem(i, 3, QTableWidgetItem(str(u.n_spikes)))

            label_item = QTableWidgetItem(str(u.label))
            if u.label == 1:
                label_item.setBackground(QColor(200, 255, 200))
            self.units_table.setItem(i, 4, label_item)

            self.units_table.setItem(i, 5, QTableWidgetItem(f"{u.amp_median:.1f}"))
            self.units_table.setItem(i, 6, QTableWidgetItem(f"{u.depth_um:.0f}"))

        n_good = sum(1 for u in self.current_units if u.label == 1)
        self.unit_count_label.setText(
            f"Total: {len(self.current_units)} | Good: {n_good} | Displayed: {len(display_units)}"
        )

        self.update_fr_histogram()
        self.update_region_summary()
        self.update_raster_plot()

        self.progress_bar.hide()
        self.update_selection_summary()

    def on_units_selected(self):
        rows = self.units_table.selectionModel().selectedRows()
        self.selected_units = [int(self.units_table.item(r.row(), 0).text()) for r in rows]
        self.update_selection_summary()
        self.update_raster_plot()

    def update_raster_plot(self):
        self.raster_plot.clear()
        if self.current_spikes is None:
            return

        t_start = float(self.raster_start.value())
        t_end = float(self.raster_end.value())

        units_to_plot = self.selected_units[:20] if self.selected_units else [u.cluster_id for u in self.current_units[:20]]

        spike_times = np.asarray(self.current_spikes.get("times", np.array([])))
        spike_clusters = np.asarray(self.current_spikes.get("clusters", np.array([])))

        for i, cluster_id in enumerate(units_to_plot):
            mask = (spike_clusters == cluster_id) & (spike_times >= t_start) & (spike_times <= t_end)
            times = spike_times[mask]
            if len(times) > 0:
                if len(times) > 1000:
                    step = max(1, len(times) // 1000)
                    times = times[::step]
                y = np.ones_like(times) * i
                self.raster_plot.plot(times, y, pen=None, symbol="|", symbolSize=5, symbolBrush="b")

        self.raster_plot.setYRange(-0.5, len(units_to_plot) - 0.5)

    def update_fr_histogram(self):
        self.fr_plot.clear()
        if not self.current_units:
            return

        frs = [u.firing_rate for u in self.current_units if u.firing_rate > 0]
        if not frs:
            return

        log_frs = np.log10(np.array(frs) + 0.1)
        hist, bins = np.histogram(log_frs, bins=30)
        centers = (bins[:-1] + bins[1:]) / 2
        width = bins[1] - bins[0]

        self.fr_plot.addItem(pg.BarGraphItem(x=centers, height=hist, width=width * 0.9, brush=(100, 100, 200, 150)))
        self.fr_plot.setLabel("bottom", "Firing Rate (log10 Hz)")
        self.fr_plot.setLabel("left", "Count")

    def update_region_summary(self):
        if not self.current_units:
            return

        regions: Dict[str, Dict[str, Any]] = {}
        for u in self.current_units:
            regions.setdefault(u.acronym, {"total": 0, "good": 0, "frs": []})
            regions[u.acronym]["total"] += 1
            regions[u.acronym]["good"] += int(u.label >= 1)
            regions[u.acronym]["frs"].append(u.firing_rate)

        self.region_table.setRowCount(len(regions))
        for i, (region, data) in enumerate(sorted(regions.items())):
            self.region_table.setItem(i, 0, QTableWidgetItem(region))
            self.region_table.setItem(i, 1, QTableWidgetItem(str(data["total"])))
            self.region_table.setItem(i, 2, QTableWidgetItem(str(data["good"])))
            mean_fr = float(np.mean(data["frs"])) if data["frs"] else 0.0
            self.region_table.setItem(i, 3, QTableWidgetItem(f"{mean_fr:.2f}"))

    def update_selection_summary(self):
        lines = []
        rows = self.sessions_table.selectionModel().selectedRows()
        if rows:
            sess = self.sessions[rows[0].row()]
            lines.append(f"Session: {sess.subject} / {sess.date} ({str(sess.eid)[:8]}...)")

        items = self.probes_list.selectedItems()
        if items:
            lines.append(f"Probe: {items[0].text()}")

        if self.current_units:
            n_good = sum(1 for u in self.current_units if u.label == 1)
            lines.append(f"Units: {len(self.current_units)} total, {n_good} good")

        if self.selected_units:
            lines.append(f"Selected: {len(self.selected_units)} units")

        self.selection_summary.setText("\n".join(lines))

    def export_to_yaml(self):
        rows = self.sessions_table.selectionModel().selectedRows()
        if not rows:
            QMessageBox.warning(self, "No Selection", "Please select a session first.")
            return

        sess = self.sessions[rows[0].row()]

        regions_list = [r.strip() for r in self.region_edit.text().split(",") if r.strip()]
        regions_yaml = ", ".join(f"\"{r}\"" for r in regions_list) if regions_list else "# all regions"

        lab_val = self.lab_combo.currentText()
        lab_filter = None if (lab_val == "All" or lab_val.startswith("(")) else lab_val

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
  labs:
    include: [{f'"{lab_filter}"' if lab_filter else "# all labs"}]
  regions:
    include: [{regions_yaml}]
    include_descendants: {str(self.include_descendants_check.isChecked()).lower()}

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

        path, _ = QFileDialog.getSaveFileName(self, "Export Config", "ibl_timescale_config.yaml", "YAML Files (*.yaml *.yml)")
        if path:
            with open(path, "w") as f:
                f.write(yaml_content)
            self.status_bar.showMessage(f"Exported config to {path}")

    def copy_selection(self):
        QApplication.clipboard().setText(self.selection_summary.toPlainText())
        self.status_bar.showMessage("Selection copied to clipboard")

    def export_to_parquet(self):
        if not self.current_units:
            QMessageBox.warning(self, "No Data", "Please load units from a probe first.")
            return
        rows = self.sessions_table.selectionModel().selectedRows()
        if not rows:
            QMessageBox.warning(self, "No Session", "Please select a session.")
            return

        sess = self.sessions[rows[0].row()]
        probe_items = self.probes_list.selectedItems()
        probe_data = probe_items[0].data(Qt.ItemDataRole.UserRole) if probe_items else (None, None, None)
        pid = probe_data[0] if probe_data else "unknown"
        probe_name = probe_data[2] if len(probe_data) > 2 else "probe00"

        default_dir = Path("./data")
        default_dir.mkdir(exist_ok=True)
        default_path = default_dir / f"{sess.subject}_{sess.date}_{probe_name}_units.parquet"

        path, _ = QFileDialog.getSaveFileName(self, "Export Units to Parquet", str(default_path),
                                              "Parquet Files (*.parquet);;All Files (*)")
        if not path:
            return

        try:
            data = []
            for u in self.current_units:
                data.append({
                    "subject": str(sess.subject),
                    "eid": str(sess.eid),
                    "pid": str(pid),
                    "cluster_id": int(u.cluster_id),
                    "acronym": str(u.acronym),
                    "lab": str(sess.lab),
                    "session_date": str(sess.date),
                    "probe_name": str(probe_name) if probe_name else "unknown",
                    "label": int(u.label),
                    "n_spikes": int(u.n_spikes),
                    "firing_rate_hz": float(u.firing_rate),
                    "amp_median_uV": float(u.amp_median),
                    "depth_um": float(u.depth_um),
                    "tau_ms": None,
                    "qc_status": "pending",
                    "processing_timestamp": datetime.now().isoformat(),
                })

            df = pd.DataFrame(data)
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path, index=False)

            n_good = sum(1 for u in self.current_units if u.label == 1)
            self.status_bar.showMessage(f"Exported {len(self.current_units)} units ({n_good} good) to {path}")
            QMessageBox.information(self, "Export Successful",
                                    f"Saved {len(self.current_units)} units to:\n{path}\n\nGood units: {n_good}")

        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Error saving file:\n{str(e)}")
            self.status_bar.showMessage(f"Export failed: {e}")

    def show_about(self):
        n_regions = sum(len(v) for v in IBL_CANONICAL_REGIONS.values())
        QMessageBox.about(self, "About IBL Data Loader",
                          f"IBL Timescale Data Loader\n\n"
                          f"Interactive tool for selecting and previewing\n"
                          f"IBL Brainwide Map data for timescale analysis.\n\n"
                          f"Includes {n_regions} canonical regions.\n\n"
                          f"Version 1.2")


# =============================================================================
# Main
# =============================================================================

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = IBLDataLoaderGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
