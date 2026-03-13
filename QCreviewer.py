"""
QSM QC Reviewer v10
=====================
Changes from v9:

1. SPLITTABLE LEFT PANEL
   The three viewer canvases are arranged in a vertical QSplitter (default 50/50):
     Top  row : QSplitter(H) → Raw QSM | Cortical QSM
     Bottom row: QSplitter(H) → Subcortical QSM | Info panel (scrollable)
   Dragging any divider triggers a debounced fit_all.  The per-row H-splitters
   start at 50/50.

2. SCROLLABLE INFO PANEL
   The bottom-right info panel is wrapped in a QScrollArea so it never gets
   clipped when the window is small.

3. ORIENTATION IN TOOLBAR
   All orientation controls (Canonical, View, Flip H, Flip V) have been moved
   from the info panel into a compact QToolBar that sits just below the menu bar.
   Default: LPI (ITK-SNAP) + Axial — correct ITK-SNAP convention on first launch.

4. UI POLISH
   Consistent dark theme with accent colours, rounded group-box titles,
   better spacing, monospace case labels, colour-coded status text.

5. ASYNC LOADING WITH PROGRESS BAR
   File I/O and reorientation happen in a QThread (LoadWorker).
   A QProgressBar embedded in the status bar shows live steps.
   Navigation / case-list are disabled during loading and re-enabled when
   the worker finishes.  The main thread only handles napari layer calls
   (which must stay on the GUI thread).
"""

import os
import sys
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional, Tuple

import nibabel as nib
import nibabel.orientations as nibo
import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation

from qtpy.QtCore import QObject, QEvent, QTimer, Qt, QThread, Signal
from qtpy.QtWidgets import (
    QAction, QApplication, QCheckBox, QComboBox,
    QDialog, QDoubleSpinBox, QFormLayout, QFrame,
    QGroupBox, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QMainWindow,
    QProgressBar, QPushButton, QScrollArea,
    QSizePolicy, QSplitter, QTextEdit, QToolBar,
    QVBoxLayout, QWidget,
)

from napari.components import ViewerModel
from napari.qt import QtViewer


# ─────────────────────────────────────────────────────────────────────────────
# User config
# ─────────────────────────────────────────────────────────────────────────────

CASES_ROOT = r"E:/52594/OneDrive_Hopkins/Research/UKB/qc/data/checkcases/examples/initial256cases/data_in_casefolder/"
OUTPUT_CSV = os.path.join(CASES_ROOT, "qc_results.csv")

FILE_NAMES = {
    "raw_qsm":           "QSM_TOTAL_mcpc3Ds_chi_SFCR+0_Avg_wGDC.nii.gz",
    "segmentation":      "T1_SynthSeg_relabeled_corrected_to_SWI.nii.gz",
    "subcortical_label": "QSM_TOTAL_mcpc3Ds_chi_SFCR_Avg_wGDC_labels_a2.nii.gz",
    "cortical_qsm":      "QSM_TOTAL_mcpc3Ds_chi_SFCR_Avg_wGDC_cortical_dilated.nii.gz",
    "subcortical_qsm":   "QSM_TOTAL_mcpc3Ds_chi_SFCR_Avg_wGDC_subcortical_expanded.nii.gz",
}

SAVE_GENERATED_MASKED_QSM = False
SUBCORTICAL_MARGIN        = 3
CORTICAL_DILATION_ITER    = 2
DEFAULT_LEVEL             = 0.0
DEFAULT_WINDOW            = 0.4
DEFAULT_LOW               = DEFAULT_LEVEL - DEFAULT_WINDOW / 2.0
DEFAULT_HIGH              = DEFAULT_LEVEL + DEFAULT_WINDOW / 2.0
SEGMENTATION_OPACITY      = 0.35
DEFAULT_CORT_SEG_VISIBLE  = False
DEFAULT_SUB_SEG_VISIBLE   = False

MOTION_SCORES = [
    "0 – No motion",
    "1 – Mild motion",
    "2 – Motion affecting internal structure",
    "3 – Severe motion",
]

ROI_LABEL_SYNTHSEG_COMBINED = {
    "Fron": [33, 41, 43, 47, 48, 49, 56, 57, 61, 67, 75, 77, 81, 82, 83, 90, 91, 95],
    "Temp": [31, 36, 38, 44, 59, 62, 63, 65, 70, 72, 78, 93, 96, 97],
    "Pari": [37, 46, 51, 53, 58, 60, 71, 80, 85, 87, 92, 94],
    "Occi": [34, 40, 42, 50, 54, 68, 74, 76, 84, 88],
    "Cing": [32, 39, 52, 55, 66, 73, 86, 89],
    "Hipp": [13, 27],
    "Amyg": [14, 28],
    "Accu": [16, 29],
}
_ALL_CORTICAL_LABELS = sorted(
    {lbl for labels in ROI_LABEL_SYNTHSEG_COMBINED.values() for lbl in labels}
)

# ── Orientation tables (unchanged from v9) ────────────────────────────────────

CANONICAL_OPTIONS: Dict[str, Optional[Tuple[str, str, str]]] = {
    "LPI (ITK-SNAP)": ('L', 'P', 'I'),
    "RAS":            ('R', 'A', 'S'),
    "Native":         None,
}
DEFAULT_CANONICAL = "LPI (ITK-SNAP)"

ORIENTATIONS: Dict[str, Dict] = {
    "Axial":    {"order": (2, 1, 0), "axis": 2},
    "Coronal":  {"order": (1, 2, 0), "axis": 1},
    "Sagittal": {"order": (0, 2, 1), "axis": 0},
}
DEFAULT_ORIENTATION = "Axial"

_AUTO_FLIP: Dict[Tuple[str, str], Tuple[bool, bool]] = {
    ("LPI (ITK-SNAP)", "Axial"):    (False, False),
    ("LPI (ITK-SNAP)", "Coronal"):  (False, False),
    ("LPI (ITK-SNAP)", "Sagittal"): (False, False),
    ("RAS",            "Axial"):    (False, True),
    ("RAS",            "Coronal"):  (True,  True),
    ("RAS",            "Sagittal"): (True,  True),
    ("Native",         "Axial"):    (False, False),
    ("Native",         "Coronal"):  (False, False),
    ("Native",         "Sagittal"): (False, False),
}

_VIEW_DIR_LABELS: Dict[str, Tuple[str, str, str, str]] = {
    "Axial":    ("R", "L", "A", "P"),
    "Coronal":  ("R", "L", "S", "I"),
    "Sagittal": ("A", "P", "S", "I"),
}

# ── Dark-theme style constants ────────────────────────────────────────────────

_C_BG        = "#1e2128"
_C_PANEL     = "#252830"
_C_HEADER    = "#2b2f3a"
_C_DIR_BAR   = "#1c1f2b"
_C_BORDER    = "#3a3f4b"
_C_TEXT      = "#d4d8e2"
_C_DIM       = "#8891a0"
_C_ACCENT    = "#4a9eff"
_C_SUCCESS   = "#56b870"
_C_WARN      = "#e08b3a"
_C_SYNC_ON   = "#4a9eff"

_GLOBAL_CSS = f"""
QMainWindow, QWidget {{ background: {_C_BG}; color: {_C_TEXT}; font-family: Segoe UI, Arial, sans-serif; }}
QToolBar {{ background: {_C_HEADER}; border-bottom: 1px solid {_C_BORDER}; spacing: 6px; padding: 3px 8px; }}
QToolBar QLabel {{ color: {_C_DIM}; font-size: 11px; }}
QToolBar QComboBox {{
    background: {_C_PANEL}; color: {_C_TEXT}; border: 1px solid {_C_BORDER};
    border-radius: 3px; padding: 2px 6px; font-size: 11px; min-width: 120px;
}}
QToolBar QComboBox::drop-down {{ border: none; width: 18px; }}
QToolBar QCheckBox {{ color: {_C_TEXT}; font-size: 11px; }}
QSplitter::handle {{ background: {_C_BORDER}; }}
QSplitter::handle:horizontal {{ width: 4px; }}
QSplitter::handle:vertical   {{ height: 4px; }}
QGroupBox {{
    border: 1px solid {_C_BORDER}; border-radius: 5px;
    margin-top: 18px; padding-top: 6px;
    font-size: 11px; font-weight: 600; color: {_C_ACCENT};
}}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; }}
QComboBox {{
    background: {_C_PANEL}; color: {_C_TEXT}; border: 1px solid {_C_BORDER};
    border-radius: 3px; padding: 3px 8px; font-size: 11px;
}}
QComboBox QAbstractItemView {{ background: {_C_PANEL}; color: {_C_TEXT}; selection-background-color: {_C_ACCENT}; }}
QListWidget {{
    background: {_C_PANEL}; color: {_C_TEXT}; border: 1px solid {_C_BORDER};
    border-radius: 3px; font-size: 11px;
}}
QListWidget::item:selected {{ background: {_C_ACCENT}; color: white; }}
QListWidget::item:hover {{ background: {_C_HEADER}; }}
QTextEdit {{
    background: {_C_PANEL}; color: {_C_TEXT}; border: 1px solid {_C_BORDER};
    border-radius: 3px; font-size: 11px; padding: 4px;
}}
QPushButton {{
    background: {_C_PANEL}; color: {_C_TEXT}; border: 1px solid {_C_BORDER};
    border-radius: 4px; padding: 5px 12px; font-size: 11px;
}}
QPushButton:hover  {{ background: {_C_HEADER}; border-color: {_C_ACCENT}; }}
QPushButton:pressed {{ background: {_C_ACCENT}; color: white; }}
QPushButton:disabled {{ color: {_C_DIM}; border-color: {_C_BORDER}; }}
QCheckBox {{ color: {_C_TEXT}; font-size: 11px; spacing: 5px; }}
QDoubleSpinBox {{
    background: {_C_PANEL}; color: {_C_TEXT}; border: 1px solid {_C_BORDER};
    border-radius: 3px; padding: 2px 6px; font-size: 11px;
}}
QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{
    background: {_C_BG}; width: 8px; border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background: {_C_BORDER}; border-radius: 4px; min-height: 20px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}
QStatusBar {{ background: {_C_HEADER}; color: {_C_DIM}; font-size: 10px; }}
QProgressBar {{
    background: {_C_PANEL}; border: 1px solid {_C_BORDER}; border-radius: 3px;
    text-align: center; color: {_C_TEXT}; font-size: 10px; max-height: 14px;
}}
QProgressBar::chunk {{ background: {_C_ACCENT}; border-radius: 2px; }}
QMenuBar {{ background: {_C_HEADER}; color: {_C_TEXT}; }}
QMenuBar::item:selected {{ background: {_C_ACCENT}; }}
QMenu {{ background: {_C_PANEL}; color: {_C_TEXT}; border: 1px solid {_C_BORDER}; }}
QMenu::item:selected {{ background: {_C_ACCENT}; }}
"""

_SAVE_BTN_CSS = f"""
QPushButton {{
    background: {_C_ACCENT}; color: white; border: none;
    border-radius: 4px; padding: 7px 14px; font-size: 12px; font-weight: 600;
}}
QPushButton:hover  {{ background: #5aabff; }}
QPushButton:pressed {{ background: #3a8eef; }}
QPushButton:disabled {{ background: {_C_BORDER}; color: {_C_DIM}; }}
"""

_NAV_BTN_CSS = f"""
QPushButton {{
    background: {_C_PANEL}; color: {_C_TEXT}; border: 1px solid {_C_BORDER};
    border-radius: 4px; padding: 5px 16px; font-size: 11px;
}}
QPushButton:hover  {{ background: {_C_HEADER}; border-color: {_C_ACCENT}; color: {_C_ACCENT}; }}
QPushButton:disabled {{ color: {_C_DIM}; border-color: {_C_BORDER}; }}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CasePaths:
    case_id:           str
    case_dir:          str
    raw_qsm:           str
    segmentation:      str
    subcortical_label: str
    cortical_qsm:      Optional[str] = None
    subcortical_qsm:   Optional[str] = None


@dataclass
class QCRecord:
    case_id:         str
    cortex_score:    str = ""
    subcortex_score: str = ""
    notes:           str = ""


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def find_cases(root: str, file_names: Dict[str, str]) -> List[CasePaths]:
    cases: List[CasePaths] = []
    if not os.path.isdir(root):
        raise FileNotFoundError(f"CASES_ROOT not found: {root}")
    required = ["raw_qsm", "segmentation", "subcortical_label"]
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if not os.path.isdir(d):
            continue
        paths = {k: os.path.join(d, v) for k, v in file_names.items()}
        if not all(os.path.exists(paths[k]) for k in required):
            continue
        cases.append(CasePaths(
            case_id=name, case_dir=d,
            raw_qsm=paths["raw_qsm"],
            segmentation=paths["segmentation"],
            subcortical_label=paths["subcortical_label"],
            cortical_qsm=paths["cortical_qsm"]
                if os.path.exists(paths["cortical_qsm"]) else None,
            subcortical_qsm=paths["subcortical_qsm"]
                if os.path.exists(paths["subcortical_qsm"]) else None,
        ))
    return cases


def read_existing_results(csv_path: str) -> Dict[str, QCRecord]:
    results: Dict[str, QCRecord] = {}
    if not os.path.exists(csv_path):
        return results
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        rec = QCRecord(
            case_id=str(row.get("case_id", "")),
            cortex_score=str(row.get("cortex_score", "")),
            subcortex_score=str(row.get("subcortex_score", "")),
            notes=str(row.get("notes", "")),
        )
        if rec.case_id:
            results[rec.case_id] = rec
    return results


def write_results(csv_path: str, records: Dict[str, QCRecord]) -> None:
    rows = [asdict(v) for _, v in sorted(records.items(), key=lambda x: x[0])]
    pd.DataFrame(
        rows, columns=["case_id", "cortex_score", "subcortex_score", "notes"]
    ).to_csv(csv_path, index=False)


# ─────────────────────────────────────────────────────────────────────────────
# Reorientation helpers
# ─────────────────────────────────────────────────────────────────────────────

def reorient_image(img: nib.Nifti1Image,
                   target_axcodes: Optional[Tuple[str, str, str]]) -> nib.Nifti1Image:
    if target_axcodes is None:
        return img
    curr_ornt = nibo.io_orientation(img.affine)
    targ_ornt = nibo.axcodes2ornt(target_axcodes)
    transform = nibo.ornt_transform(curr_ornt, targ_ornt)
    return img.as_reoriented(transform)


def native_dir_labels(affine: np.ndarray,
                      orient_name: str) -> Tuple[str, str, str, str]:
    try:
        axcodes = nibo.aff2axcodes(affine)
        cfg      = ORIENTATIONS[orient_name]
        order    = cfg["order"]
        row_axis = order[-2]
        col_axis = order[-1]
        col_code = axcodes[col_axis]
        row_code = axcodes[row_axis]
        opp = {'R':'L','L':'R','A':'P','P':'A','S':'I','I':'S'}
        return opp.get(col_code,'?'), col_code, opp.get(row_code,'?'), row_code
    except Exception:
        return ('?', '?', '?', '?')


# ─────────────────────────────────────────────────────────────────────────────
# QSM generation  (pure data, safe to run in worker thread)
# ─────────────────────────────────────────────────────────────────────────────

def ensure_3d(arr: np.ndarray, name: str) -> np.ndarray:
    if arr.ndim != 3:
        raise ValueError(f"{name} is not 3D, shape={arr.shape}")
    return arr


def generate_subcortical_qsm(qsm, label, margin=3):
    mask = label > 0
    if not np.any(mask):
        return np.zeros_like(qsm)
    coords = np.where(mask)
    slices = tuple(
        slice(max(0, coords[ax].min() - margin),
              min(mask.shape[ax], coords[ax].max() + margin + 1))
        for ax in range(3)
    )
    cube = np.zeros_like(mask, dtype=np.uint8)
    cube[slices] = 1
    return qsm * cube


def generate_cortical_qsm(qsm, seg, roi_dict, dilation_iter=2):
    labels = sorted({l for v in roi_dict.values() for l in v})
    return qsm * binary_dilation(np.isin(seg, labels), iterations=dilation_iter)


def load_case_data(case: CasePaths,
                   canonical_key: str,
                   save_generated: bool,
                   subcortical_margin: int,
                   cortical_dilation_iter: int,
                   progress_cb: Callable[[int, str], None]):
    """
    Pure data loading — runs in worker thread.
    progress_cb(percent, message) called at each step.
    Returns a dict consumed by _apply_loaded_data().
    """
    target_axcodes = CANONICAL_OPTIONS[canonical_key]

    progress_cb(5, "Loading raw QSM…")
    raw_img_native = nib.load(case.raw_qsm)
    native_axcodes = nibo.aff2axcodes(raw_img_native.affine)
    raw_img        = reorient_image(raw_img_native, target_axcodes)
    qsm_data       = np.asarray(raw_img.get_fdata(), dtype=np.float32)
    reoriented_affine = raw_img.affine

    progress_cb(25, "Loading segmentation…")
    seg_img  = reorient_image(nib.load(case.segmentation), target_axcodes)
    seg_data = np.asarray(seg_img.get_fdata(), dtype=np.float32)

    progress_cb(40, "Loading subcortical labels…")
    sub_lbl_img  = reorient_image(nib.load(case.subcortical_label), target_axcodes)
    sub_lbl_data = np.asarray(sub_lbl_img.get_fdata()).astype(np.int32)

    cort_ok = case.cortical_qsm is not None and os.path.exists(case.cortical_qsm)
    sub_ok  = case.subcortical_qsm is not None and os.path.exists(case.subcortical_qsm)

    progress_cb(55, "Loading cortical QSM…")
    if cort_ok:
        cort_data = np.asarray(
            reorient_image(nib.load(case.cortical_qsm), target_axcodes).get_fdata(),
            dtype=np.float32)
    else:
        progress_cb(60, "Generating cortical mask…")
        cort_data = generate_cortical_qsm(
            qsm_data, seg_data, ROI_LABEL_SYNTHSEG_COMBINED, cortical_dilation_iter
        ).astype(np.float32)
        if save_generated:
            p = os.path.join(case.case_dir, FILE_NAMES["cortical_qsm"])
            nib.save(nib.Nifti1Image(cort_data, raw_img.affine, raw_img.header), p)
            case.cortical_qsm = p; cort_ok = True

    progress_cb(75, "Loading subcortical QSM…")
    if sub_ok:
        sub_data = np.asarray(
            reorient_image(nib.load(case.subcortical_qsm), target_axcodes).get_fdata(),
            dtype=np.float32)
    else:
        progress_cb(80, "Generating subcortical mask…")
        sub_data = generate_subcortical_qsm(
            qsm_data, sub_lbl_data, subcortical_margin
        ).astype(np.float32)
        if save_generated:
            p = os.path.join(case.case_dir, FILE_NAMES["subcortical_qsm"])
            nib.save(nib.Nifti1Image(sub_data, raw_img.affine, raw_img.header), p)
            case.subcortical_qsm = p; sub_ok = True

    progress_cb(92, "Building seg overlays…")
    cortical_seg = np.where(
        np.isin(seg_data.astype(np.int32), _ALL_CORTICAL_LABELS),
        seg_data.astype(np.int32), 0)

    progress_cb(98, "Ready")
    return dict(
        raw=ensure_3d(qsm_data,      "raw_qsm"),
        cort=ensure_3d(cort_data,    "cortical_qsm"),
        sub=ensure_3d(sub_data,      "subcortical_qsm"),
        cortical_seg=ensure_3d(cortical_seg, "cortical_seg"),
        subcortical_seg=ensure_3d(sub_lbl_data, "subcortical_label"),
        cort_ok=cort_ok, sub_ok=sub_ok,
        native_axcodes=native_axcodes,
        reoriented_affine=reoriented_affine,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Background loader  (QThread)
# ─────────────────────────────────────────────────────────────────────────────

class LoadWorker(QThread):
    progress = Signal(int, str)   # (percent 0-100, message)
    finished = Signal(object)     # dict from load_case_data
    error    = Signal(str)

    def __init__(self, case, canonical_key, save_generated,
                 subcortical_margin, cortical_dilation_iter):
        super().__init__()
        self._case   = case
        self._ck     = canonical_key
        self._save   = save_generated
        self._sm     = subcortical_margin
        self._cdi    = cortical_dilation_iter

    def run(self):
        try:
            result = load_case_data(
                self._case, self._ck, self._save, self._sm, self._cdi,
                lambda pct, msg: self.progress.emit(pct, msg))
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# Dims helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_slice(v: ViewerModel, axis: int) -> int:
    if axis >= len(v.dims.current_step):
        return 0
    return int(round(v.dims.current_step[axis]))


def set_slice(v: ViewerModel, z: int, axis: int) -> None:
    if axis >= len(v.dims.nsteps):
        return
    n = int(v.dims.nsteps[axis])
    v.dims.set_current_step(axis, max(0, min(z, n - 1)))


# ─────────────────────────────────────────────────────────────────────────────
# Wheel event filter
# ─────────────────────────────────────────────────────────────────────────────

class WheelScrollFilter(QObject):
    def __init__(self, on_scroll: Callable[[int], None], parent: QObject = None):
        super().__init__(parent)
        self._on_scroll = on_scroll

    def eventFilter(self, obj, event) -> bool:
        if event.type() == QEvent.Wheel:
            dy = event.angleDelta().y()
            if dy != 0:
                self._on_scroll(-1 if dy > 0 else 1)
            return True
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Contrast dialog
# ─────────────────────────────────────────────────────────────────────────────

class ContrastDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent, Qt.Tool)
        self.setWindowTitle("Contrast & Overlay")
        self.setMinimumWidth(300)
        lay = QVBoxLayout(self)
        lay.setSpacing(10)
        form = QFormLayout()
        form.setSpacing(8)

        self.level_spin = QDoubleSpinBox()
        self.level_spin.setDecimals(4); self.level_spin.setRange(-10, 10)
        self.level_spin.setSingleStep(0.01); self.level_spin.setValue(DEFAULT_LEVEL)

        self.window_spin = QDoubleSpinBox()
        self.window_spin.setDecimals(4); self.window_spin.setRange(0.0001, 20)
        self.window_spin.setSingleStep(0.01); self.window_spin.setValue(DEFAULT_WINDOW)

        self.seg_opacity_spin = QDoubleSpinBox()
        self.seg_opacity_spin.setDecimals(2); self.seg_opacity_spin.setRange(0.0, 1.0)
        self.seg_opacity_spin.setSingleStep(0.05)
        self.seg_opacity_spin.setValue(SEGMENTATION_OPACITY)

        self.sync_cb = QCheckBox("Apply to all three QSM views")
        self.sync_cb.setChecked(True)

        form.addRow("Level:",       self.level_spin)
        form.addRow("Window:",      self.window_spin)
        form.addRow("Seg opacity:", self.seg_opacity_spin)
        form.addRow(self.sync_cb)
        lay.addLayout(form)

        self._apply_cb  = None
        self._seg_op_cb = None

        btn = QPushButton("Apply Contrast")
        btn.clicked.connect(lambda: self._apply_cb and self._apply_cb(
            self.level_spin.value(), self.window_spin.value(), self.sync_cb.isChecked()))
        lay.addWidget(btn)
        close = QPushButton("Close")
        close.clicked.connect(self.hide)
        lay.addWidget(close)

        self.seg_opacity_spin.valueChanged.connect(
            lambda v: self._seg_op_cb and self._seg_op_cb(v))

    def reset(self):
        self.level_spin.setValue(DEFAULT_LEVEL)
        self.window_spin.setValue(DEFAULT_WINDOW)


# ─────────────────────────────────────────────────────────────────────────────
# ImageCanvas
# ─────────────────────────────────────────────────────────────────────────────

_DIR_CSS  = (f"font-size:10px; color:{_C_DIM}; background:{_C_DIR_BAR};"
             " padding:2px 6px; letter-spacing:1px;")
_SYNC_CSS = (f"QCheckBox {{ color:#aabbcc; font-size:10px; }}"
             f"QCheckBox::indicator {{ width:12px; height:12px; }}")


class ImageCanvas(QWidget):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # title bar
        hdr = QWidget()
        hdr.setStyleSheet(f"background:{_C_HEADER};")
        hdr.setFixedHeight(28)
        hl = QHBoxLayout(hdr)
        hl.setContentsMargins(8, 0, 8, 0); hl.setSpacing(6)
        lbl = QLabel(title)
        lbl.setStyleSheet(f"font-weight:600; font-size:11px; color:{_C_TEXT};")
        self.sync_cb = QCheckBox("🔗 Sync")
        self.sync_cb.setChecked(True)
        self.sync_cb.setStyleSheet(_SYNC_CSS)
        self.sync_cb.setToolTip("Uncheck to scroll this view independently")
        hl.addWidget(lbl, 1)
        hl.addWidget(self.sync_cb)

        # direction bar
        dir_bar = QWidget()
        dir_bar.setStyleSheet(f"background:{_C_DIR_BAR};")
        dir_bar.setFixedHeight(18)
        dl = QHBoxLayout(dir_bar)
        dl.setContentsMargins(6, 0, 6, 0); dl.setSpacing(0)
        self._lbl_left  = QLabel("← ?"); self._lbl_left.setStyleSheet(_DIR_CSS)
        self._lbl_mid   = QLabel("↑?  ↓?"); self._lbl_mid.setStyleSheet(_DIR_CSS)
        self._lbl_right = QLabel("? →"); self._lbl_right.setStyleSheet(_DIR_CSS)
        self._lbl_left.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        self._lbl_mid.setAlignment(Qt.AlignVCenter | Qt.AlignCenter)
        self._lbl_right.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
        dl.addWidget(self._lbl_left, 1)
        dl.addWidget(self._lbl_mid,  1)
        dl.addWidget(self._lbl_right,1)

        # napari viewer
        self.viewer_model = ViewerModel(title=title)
        self.qt_viewer    = QtViewer(self.viewer_model)
        self.qt_viewer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._canvas_native: Optional[QWidget] = self._find_native()
        self._wheel_filter: Optional[WheelScrollFilter] = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(hdr)
        lay.addWidget(dir_bar)
        lay.addWidget(self.qt_viewer, 1)

        self.viewer_model.layers.selection.events.active.connect(
            self._on_layer_active)

    def _find_native(self):
        for attr in ("canvas", "_canvas", "native"):
            obj = getattr(self.qt_viewer, attr, None)
            if obj is None: continue
            if hasattr(obj, "width"): return obj
            sub = getattr(obj, "native", None)
            if sub and hasattr(sub, "width"): return sub
        return None

    def _on_layer_active(self, event=None):
        try: self.viewer_model.camera.mouse_zoom = False
        except AttributeError: pass

    @property
    def sync_enabled(self): return self.sync_cb.isChecked()

    def set_direction_labels(self, left, right, top, bottom):
        self._lbl_left.setText(f"← {left}")
        self._lbl_right.setText(f"{right} →")
        self._lbl_mid.setText(f"↑ {top}   ↓ {bottom}")

    def install_wheel_filter(self, on_scroll):
        target = self._canvas_native or self.qt_viewer
        self._wheel_filter = WheelScrollFilter(on_scroll, parent=self)
        target.installEventFilter(self._wheel_filter)

    @property
    def viewer(self): return self.viewer_model

    def clear_layers(self): self.viewer.layers.clear()
    def add_image(self, *a, **kw): return self.viewer.add_image(*a, **kw)
    def add_labels(self, *a, **kw): return self.viewer.add_labels(*a, **kw)

    def lock_scroll_mode(self):
        try: self.viewer_model.camera.mouse_zoom = False
        except AttributeError: pass

    def set_view(self, dims_order, scroll_axis, data_shape, flip_v, flip_h):
        self.viewer.dims.ndisplay = 2
        self.viewer.dims.order    = dims_order
        set_slice(self.viewer, data_shape[scroll_axis] // 2, scroll_axis)
        try:
            self.viewer.camera.flip = (flip_v, flip_h)
        except Exception:
            try: self.viewer.camera.flip = (flip_v, flip_h, False)
            except Exception: pass

    def fit_to_shape(self, data_shape, dims_order):
        def _do():
            try:
                ax_row = dims_order[-2]; ax_col = dims_order[-1]
                row_size = data_shape[ax_row]; col_size = data_shape[ax_col]
                self.viewer.camera.center = (row_size / 2.0, col_size / 2.0)
                nat = self._canvas_native
                vw = max(1, nat.width())  if nat else 400
                vh = max(1, nat.height()) if nat else 400
                self.viewer.camera.zoom = 0.90 * min(vw / col_size, vh / row_size)
            except Exception:
                self.viewer.reset_view()
        QTimer.singleShot(0, _do)


# ─────────────────────────────────────────────────────────────────────────────
# Main window
# ─────────────────────────────────────────────────────────────────────────────

class ReviewerMainWindow(QMainWindow):

    def __init__(self, cases: List[CasePaths], output_csv: str):
        super().__init__()
        self.setWindowTitle("QSM QC Reviewer")
        self.resize(1900, 1020)
        self.setStyleSheet(_GLOBAL_CSS)

        self.cases         = cases
        self.output_csv    = output_csv
        self.results       = read_existing_results(output_csv)
        self.case_index    = 0
        self.current_saved = True
        self._syncing      = False
        self._worker: Optional[LoadWorker] = None

        # Orientation state
        cfg = ORIENTATIONS[DEFAULT_ORIENTATION]
        self._scroll_axis: int  = cfg["axis"]
        self._dims_order: tuple = cfg["order"]
        self._canonical_key     = DEFAULT_CANONICAL
        self._flip_v            = False
        self._flip_h            = False

        # Data
        self.raw_data = self.cortical_data = self.subcortical_data = None
        self._reoriented_affine = None

        # Layers
        self.raw_layer = self.seg_cortical_layer = None
        self.seg_subcortical_layer = self.cortical_layer = self.subcortical_layer = None

        # Canvases
        self.raw_canvas         = ImageCanvas("Raw QSM + Segmentation")
        self.cortical_canvas    = ImageCanvas("Cortical QSM")
        self.subcortical_canvas = ImageCanvas("Subcortical QSM")

        # Contrast dialog
        self.contrast_dlg = ContrastDialog(self)
        self.contrast_dlg._apply_cb  = self.apply_contrast
        self.contrast_dlg._seg_op_cb = self._update_seg_opacity

        # Fit-all debounce timer (splitter drag)
        self._fit_timer = QTimer(self)
        self._fit_timer.setSingleShot(True)
        self._fit_timer.setInterval(80)
        self._fit_timer.timeout.connect(self._fit_all)

        self._build_menu()
        self._build_toolbar()
        self._build_ui()
        self._build_statusbar()
        self._bind_shortcuts()
        self._connect_slice_sync()
        self._install_wheel_filters()
        self.load_case(0)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _all_canvases(self):
        return [self.raw_canvas, self.cortical_canvas, self.subcortical_canvas]

    def _canvas_data_pairs(self):
        return [(self.raw_canvas, self.raw_data),
                (self.cortical_canvas, self.cortical_data),
                (self.subcortical_canvas, self.subcortical_data)]

    # ── menu ─────────────────────────────────────────────────────────────────

    def _build_menu(self):
        vm = self.menuBar().addMenu("View")
        act = QAction("Contrast && Overlay…", self)
        act.setShortcut("Ctrl+L")
        act.triggered.connect(self.contrast_dlg.show)
        vm.addAction(act)

        tm = self.menuBar().addMenu("Tools")
        self._act_save_gen = QAction(
            "Save generated QSM files", self,
            checkable=True, checked=SAVE_GENERATED_MASKED_QSM)
        tm.addAction(self._act_save_gen)

    # ── toolbar  (Orientation controls) ──────────────────────────────────────

    def _build_toolbar(self):
        tb = QToolBar("Orientation", self)
        tb.setMovable(False)
        tb.setFloatable(False)
        self.addToolBar(Qt.TopToolBarArea, tb)

        tb.addWidget(QLabel("  Canonical: "))
        self.canonical_combo = QComboBox()
        self.canonical_combo.addItems(list(CANONICAL_OPTIONS.keys()))
        self.canonical_combo.setCurrentText(DEFAULT_CANONICAL)
        self.canonical_combo.setToolTip(
            "LPI (ITK-SNAP): patient Left on screen Right, Anterior at top\n"
            "RAS: same display, different internal axis order\n"
            "Native: raw voxel order, use Flip controls to adjust")
        tb.addWidget(self.canonical_combo)

        tb.addSeparator()
        tb.addWidget(QLabel("  View: "))
        self.orient_combo = QComboBox()
        self.orient_combo.addItems(list(ORIENTATIONS.keys()))
        self.orient_combo.setCurrentText(DEFAULT_ORIENTATION)
        tb.addWidget(self.orient_combo)

        tb.addSeparator()
        self.flip_h_cb = QCheckBox(" Flip H")
        self.flip_v_cb = QCheckBox(" Flip V")
        tb.addWidget(self.flip_h_cb)
        tb.addWidget(self.flip_v_cb)

        # connections
        self.canonical_combo.currentTextChanged.connect(
            lambda _: self.load_case(self.case_index))
        self.orient_combo.currentTextChanged.connect(
            lambda _: self._apply_orientation())
        self.flip_h_cb.toggled.connect(lambda _: self._apply_orientation())
        self.flip_v_cb.toggled.connect(lambda _: self._apply_orientation())

    # ── status bar ────────────────────────────────────────────────────────────

    def _build_statusbar(self):
        sb = self.statusBar()
        self._status_msg = QLabel("Ready")
        self._status_msg.setStyleSheet(f"color:{_C_DIM}; font-size:10px; padding:0 8px;")
        sb.addWidget(self._status_msg, 1)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setFixedWidth(220)
        self._progress_bar.setVisible(False)
        sb.addPermanentWidget(self._progress_bar)

    def _set_status(self, msg: str, color: str = _C_DIM):
        self._status_msg.setStyleSheet(f"color:{color}; font-size:10px; padding:0 8px;")
        self._status_msg.setText(msg)

    # ── layout ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        # ── outer left/right splitter ─────────────────────────────────────
        self._main_splitter = QSplitter(Qt.Horizontal)
        self._main_splitter.setChildrenCollapsible(False)

        # ── left: vertical splitter (top row + bottom row) ────────────────
        self._v_splitter = QSplitter(Qt.Vertical)
        self._v_splitter.setChildrenCollapsible(False)

        # top row: Raw | Cortical
        self._top_splitter = QSplitter(Qt.Horizontal)
        self._top_splitter.setChildrenCollapsible(False)
        self._top_splitter.addWidget(self.raw_canvas)
        self._top_splitter.addWidget(self.cortical_canvas)

        # bottom row: Subcortical | Info (scrollable)
        self._bot_splitter = QSplitter(Qt.Horizontal)
        self._bot_splitter.setChildrenCollapsible(False)
        self._bot_splitter.addWidget(self.subcortical_canvas)
        self._bot_splitter.addWidget(self._build_info_scroll())

        self._v_splitter.addWidget(self._top_splitter)
        self._v_splitter.addWidget(self._bot_splitter)

        self._main_splitter.addWidget(self._v_splitter)
        self._main_splitter.addWidget(self._build_qc_panel())
        self._main_splitter.setChildrenCollapsible(False)

        root.addWidget(self._main_splitter)

        # Connect splitter moves → debounced fit
        for spl in (self._v_splitter, self._top_splitter, self._bot_splitter,
                    self._main_splitter):
            spl.splitterMoved.connect(lambda *_: self._fit_timer.start())

        # Deferred size setup
        QTimer.singleShot(0, self._init_splitter_sizes)

    def _init_splitter_sizes(self):
        w = self._main_splitter.width()
        if w > 10:
            self._main_splitter.setSizes([int(w * 0.62), int(w * 0.38)])
        h = self._v_splitter.height()
        if h > 10:
            self._v_splitter.setSizes([h // 2, h // 2])
        # equal horizontal splits
        for spl in (self._top_splitter, self._bot_splitter):
            s = spl.width()
            if s > 10:
                spl.setSizes([s // 2, s // 2])

    # ── info panel (scrollable) ───────────────────────────────────────────────

    def _build_info_scroll(self) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setMinimumWidth(200)
        scroll.setWidget(self._build_info_panel())
        return scroll

    def _build_info_panel(self) -> QWidget:
        panel = QWidget()
        panel.setStyleSheet(f"QWidget {{ background: {_C_PANEL}; }}")
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(10)

        # ── Case info ──────────────────────────────────────────────────────
        gb = QGroupBox("Case")
        vl = QVBoxLayout(gb); vl.setSpacing(4)
        self.case_label = QLabel("-")
        self.case_label.setStyleSheet(
            f"font-family: Consolas, monospace; font-weight:700;"
            f" font-size:14px; color:{_C_ACCENT};")
        self.status_label = QLabel("Status: unsaved")
        self.source_label = QLabel("Sources: -")
        self.source_label.setStyleSheet(f"color:{_C_DIM}; font-size:10px;")
        self.orient_label = QLabel("Native: -")
        self.orient_label.setStyleSheet(f"color:{_C_DIM}; font-size:10px;")
        for w in (self.case_label, self.status_label,
                  self.source_label, self.orient_label):
            vl.addWidget(w)
        lay.addWidget(gb)

        # ── Overlay toggles ───────────────────────────────────────────────
        ob = QGroupBox("Overlay")
        ol = QVBoxLayout(ob); ol.setSpacing(6)
        self.cort_seg_cb = QCheckBox("Cortical labels (SynthSeg ROIs)")
        self.cort_seg_cb.setChecked(DEFAULT_CORT_SEG_VISIBLE)
        self.sub_seg_cb  = QCheckBox("Subcortical labels (label file)")
        self.sub_seg_cb.setChecked(DEFAULT_SUB_SEG_VISIBLE)
        ol.addWidget(self.cort_seg_cb)
        ol.addWidget(self.sub_seg_cb)
        lay.addWidget(ob)

        # ── Hotkeys ───────────────────────────────────────────────────────
        hb = QGroupBox("Hotkeys")
        hl = QVBoxLayout(hb)
        hotkey_lbl = QLabel(
            "Wheel    scroll slices (respects Sync)\n"
            "Ctrl+L   Contrast & Overlay\n"
            "C        Toggle cortical overlay\n"
            "S        Toggle subcortical overlay\n"
            "N / P    Next / Prev case (auto-save)\n"
            "Ctrl+S   Save")
        hotkey_lbl.setStyleSheet(
            f"font-family: Consolas, monospace; font-size:10px; color:{_C_DIM};")
        hl.addWidget(hotkey_lbl)
        lay.addWidget(hb)

        lay.addStretch(1)

        self.cort_seg_cb.toggled.connect(
            lambda v: self._update_seg_visibility("cortical", v))
        self.sub_seg_cb.toggled.connect(
            lambda v: self._update_seg_visibility("subcortical", v))
        return panel

    # ── QC panel ──────────────────────────────────────────────────────────────

    def _build_qc_panel(self) -> QWidget:
        panel = QWidget()
        panel.setStyleSheet(f"QWidget {{ background: {_C_PANEL}; }}")
        panel.setMinimumWidth(280)
        panel.setMaximumWidth(440)
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(10)

        # Scores
        sg = QGroupBox("Motion QC Scores")
        sf = QFormLayout(sg); sf.setSpacing(8)
        self.cortex_combo    = QComboBox()
        self.subcortex_combo = QComboBox()
        for cb in (self.cortex_combo, self.subcortex_combo):
            cb.addItem("")
            cb.addItems(MOTION_SCORES)
        sf.addRow("Cortex:",    self.cortex_combo)
        sf.addRow("Subcortex:", self.subcortex_combo)
        lay.addWidget(sg)

        # Notes
        ng = QGroupBox("Notes")
        nl = QVBoxLayout(ng)
        self.notes_edit = QTextEdit()
        self.notes_edit.setPlaceholderText("Optional notes…")
        self.notes_edit.setMinimumHeight(80)
        self.notes_edit.setMaximumHeight(140)
        nl.addWidget(self.notes_edit)
        lay.addWidget(ng)

        # Save
        self._btn_save = QPushButton("💾   Save   (Ctrl+S)")
        self._btn_save.setStyleSheet(_SAVE_BTN_CSS)
        self._btn_save.clicked.connect(self.save_current_case)
        lay.addWidget(self._btn_save)

        # Navigation
        nav = QGroupBox("Navigation")
        nl2 = QHBoxLayout(nav); nl2.setSpacing(8)
        self.btn_prev = QPushButton("◀  Prev")
        self.btn_next = QPushButton("Next  ▶")
        for b in (self.btn_prev, self.btn_next):
            b.setStyleSheet(_NAV_BTN_CSS)
        self.btn_prev.clicked.connect(self.prev_case)
        self.btn_next.clicked.connect(self.next_case)
        nl2.addWidget(self.btn_prev)
        nl2.addWidget(self.btn_next)
        lay.addWidget(nav)

        # Case list
        lg = QGroupBox("Case List")
        ll = QVBoxLayout(lg)
        self.case_list = QListWidget()
        for c in self.cases:
            self.case_list.addItem(QListWidgetItem(c.case_id))
        self.case_list.setMinimumHeight(180)
        self.case_list.itemClicked.connect(self.on_case_clicked)
        ll.addWidget(self.case_list)
        lay.addWidget(lg, 1)

        self.cortex_combo.currentTextChanged.connect(self._mark_unsaved)
        self.subcortex_combo.currentTextChanged.connect(self._mark_unsaved)
        self.notes_edit.textChanged.connect(self._mark_unsaved)
        return panel

    # ── wheel filters ─────────────────────────────────────────────────────────

    def _install_wheel_filters(self):
        for canvas in self._all_canvases():
            def make_cb(c):
                def _on_scroll(delta):
                    self._on_canvas_wheel(c, delta)
                return _on_scroll
            canvas.install_wheel_filter(make_cb(canvas))

    def _on_canvas_wheel(self, source: ImageCanvas, delta: int):
        if self._worker and self._worker.isRunning():
            return
        sa    = self._scroll_axis
        new_z = get_slice(source.viewer, sa) + delta
        if not source.sync_enabled:
            set_slice(source.viewer, new_z, sa)
            return
        self._syncing = True
        try:
            for c in self._all_canvases():
                if c.sync_enabled:
                    set_slice(c.viewer, new_z, sa)
        finally:
            self._syncing = False

    # ── slice sync ────────────────────────────────────────────────────────────

    def _connect_slice_sync(self):
        def make_handler(src: ImageCanvas):
            def _h(event=None):
                if self._syncing or not src.sync_enabled:
                    return
                sa = self._scroll_axis
                if (event is not None and hasattr(event, "value")
                        and event.value is not None and sa < len(event.value)):
                    z = int(round(event.value[sa]))
                else:
                    z = get_slice(src.viewer, sa)
                self._syncing = True
                try:
                    for c in self._all_canvases():
                        if c is not src and c.sync_enabled:
                            set_slice(c.viewer, z, sa)
                finally:
                    self._syncing = False
            return _h

        for c in self._all_canvases():
            c.viewer.dims.events.current_step.connect(make_handler(c))

    def _set_all_slices_force(self, z: int):
        for c in self._all_canvases():
            set_slice(c.viewer, z, self._scroll_axis)

    # ── orientation ───────────────────────────────────────────────────────────

    def _apply_orientation(self, force_mid: bool = False):
        orient_name   = self.orient_combo.currentText()
        canonical_key = self.canonical_combo.currentText()
        cfg = ORIENTATIONS.get(orient_name, ORIENTATIONS[DEFAULT_ORIENTATION])
        self._scroll_axis   = cfg["axis"]
        self._dims_order    = cfg["order"]
        self._canonical_key = canonical_key

        auto_fv, auto_fh = _AUTO_FLIP.get((canonical_key, orient_name), (False, False))
        fv = auto_fv ^ self.flip_v_cb.isChecked()
        fh = auto_fh ^ self.flip_h_cb.isChecked()
        self._flip_v, self._flip_h = fv, fh

        if canonical_key == "Native" and self._reoriented_affine is not None:
            left, right, top, bottom = native_dir_labels(
                self._reoriented_affine, orient_name)
            if fh: left, right = right, left
            if fv: top, bottom = bottom, top
        else:
            left, right, top, bottom = _VIEW_DIR_LABELS.get(
                orient_name, ('?','?','?','?'))

        for canvas, data in self._canvas_data_pairs():
            if data is None: continue
            canvas.set_view(self._dims_order, self._scroll_axis, data.shape, fv, fh)
            canvas.lock_scroll_mode()
            canvas.set_direction_labels(left, right, top, bottom)
            canvas.fit_to_shape(data.shape, self._dims_order)

        if force_mid and self.raw_data is not None:
            mid = self.raw_data.shape[self._scroll_axis] // 2
            self._set_all_slices_force(mid)

    # ── fit / resize ──────────────────────────────────────────────────────────

    def _fit_all(self):
        for canvas, data in self._canvas_data_pairs():
            if data is not None:
                canvas.fit_to_shape(data.shape, self._dims_order)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_timer.start()

    # ── shortcuts ─────────────────────────────────────────────────────────────

    def _bind_shortcuts(self):
        for c in self._all_canvases():
            c.viewer.bind_key("C")(lambda _: self.cort_seg_cb.toggle())
            c.viewer.bind_key("S")(lambda _: self.sub_seg_cb.toggle())
            c.viewer.bind_key("N")(lambda _: self.next_case())
            c.viewer.bind_key("P")(lambda _: self.prev_case())
            c.viewer.bind_key("Control-S")(lambda _: self.save_current_case())

    # ── misc ─────────────────────────────────────────────────────────────────

    def current_case(self): return self.cases[self.case_index]
    def current_case_id(self): return self.current_case().case_id

    def _mark_unsaved(self, *_):
        self.current_saved = False
        self.status_label.setText("Status:  ✏ unsaved")
        self.status_label.setStyleSheet(f"color:{_C_WARN}; font-size:11px;")

    def _update_seg_visibility(self, which, checked):
        if which == "cortical" and self.seg_cortical_layer:
            self.seg_cortical_layer.visible = checked
        elif which == "subcortical" and self.seg_subcortical_layer:
            self.seg_subcortical_layer.visible = checked

    def _update_seg_opacity(self, value):
        for lyr in (self.seg_cortical_layer, self.seg_subcortical_layer):
            if lyr is not None: lyr.opacity = float(value)

    def apply_contrast(self, level, window, apply_all=True):
        lo, hi = level - window / 2, level + window / 2
        layers = [self.raw_layer]
        if apply_all: layers += [self.cortical_layer, self.subcortical_layer]
        for lyr in layers:
            if lyr is not None: lyr.contrast_limits = (lo, hi)

    @staticmethod
    def _score_to_label(score):
        if not score: return ""
        for label in MOTION_SCORES:
            if label[0] == score: return label
        return ""

    @staticmethod
    def _label_to_score(label): return label[0] if label else ""

    def _collect_record(self):
        return QCRecord(
            case_id=self.current_case_id(),
            cortex_score=self._label_to_score(self.cortex_combo.currentText()),
            subcortex_score=self._label_to_score(self.subcortex_combo.currentText()),
            notes=self.notes_edit.toPlainText().strip())

    # ── save / navigation ─────────────────────────────────────────────────────

    def save_current_case(self):
        rec = self._collect_record()
        self.results[rec.case_id] = rec
        write_results(self.output_csv, self.results)
        self.current_saved = True
        self.status_label.setText("Status:  ✔ saved")
        self.status_label.setStyleSheet(f"color:{_C_SUCCESS}; font-size:11px;")

    def next_case(self):
        if self.case_index >= len(self.cases) - 1 or self._is_loading(): return
        self.save_current_case()
        self.load_case(self.case_index + 1)

    def prev_case(self):
        if self.case_index <= 0 or self._is_loading(): return
        self.save_current_case()
        self.load_case(self.case_index - 1)

    def on_case_clicked(self, item):
        if self._is_loading(): return
        for i, c in enumerate(self.cases):
            if c.case_id == item.text() and i != self.case_index:
                self.save_current_case()
                self.load_case(i)
                return

    def _is_loading(self):
        return self._worker is not None and self._worker.isRunning()

    def _set_nav_enabled(self, enabled: bool):
        self.btn_prev.setEnabled(enabled and self.case_index > 0)
        self.btn_next.setEnabled(enabled and self.case_index < len(self.cases) - 1)
        self._btn_save.setEnabled(enabled)
        self.case_list.setEnabled(enabled)
        self.canonical_combo.setEnabled(enabled)
        self.orient_combo.setEnabled(enabled)

    # ── load case (async) ─────────────────────────────────────────────────────

    def load_case(self, index: int):
        # Cancel any running worker
        if self._worker and self._worker.isRunning():
            self._worker.quit()
            self._worker.wait(1000)

        self.case_index = index
        canonical_key   = self.canonical_combo.currentText() \
            if hasattr(self, "canonical_combo") else DEFAULT_CANONICAL

        # Clear canvases immediately so old data disappears
        for c in self._all_canvases():
            c.clear_layers()

        # Show progress
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(True)
        self._set_status(f"Loading {self.cases[index].case_id}…", _C_ACCENT)
        self._set_nav_enabled(False)

        # Start worker
        self._worker = LoadWorker(
            case=self.cases[index],
            canonical_key=canonical_key,
            save_generated=self._act_save_gen.isChecked(),
            subcortical_margin=SUBCORTICAL_MARGIN,
            cortical_dilation_iter=CORTICAL_DILATION_ITER,
        )
        self._worker.progress.connect(self._on_load_progress)
        self._worker.finished.connect(self._on_load_finished)
        self._worker.error.connect(self._on_load_error)
        self._worker.start()

    def _on_load_progress(self, pct: int, msg: str):
        self._progress_bar.setValue(pct)
        self._set_status(msg, _C_ACCENT)

    def _on_load_error(self, msg: str):
        self._progress_bar.setVisible(False)
        self._set_status(f"Error: {msg}", _C_WARN)
        self._set_nav_enabled(True)

    def _on_load_finished(self, data: dict):
        """Called on GUI thread when worker is done. Apply layers + orientation."""
        self._progress_bar.setValue(100)

        raw  = data["raw"]
        cort = data["cort"]
        sub  = data["sub"]

        self.raw_data             = raw
        self.cortical_data        = cort
        self.subcortical_data     = sub
        self._reoriented_affine   = data["reoriented_affine"]

        opacity = self.contrast_dlg.seg_opacity_spin.value()

        self.raw_layer = self.raw_canvas.add_image(
            raw, name="raw_qsm", colormap="gray",
            contrast_limits=(DEFAULT_LOW, DEFAULT_HIGH))
        self.seg_cortical_layer = self.raw_canvas.add_labels(
            data["cortical_seg"], name="cortical_labels",
            opacity=opacity, visible=self.cort_seg_cb.isChecked())
        self.seg_subcortical_layer = self.raw_canvas.add_labels(
            data["subcortical_seg"], name="subcortical_labels",
            opacity=opacity, visible=self.sub_seg_cb.isChecked())
        self.cortical_layer = self.cortical_canvas.add_image(
            cort, name="cortical_qsm", colormap="gray",
            contrast_limits=(DEFAULT_LOW, DEFAULT_HIGH))
        self.subcortical_layer = self.subcortical_canvas.add_image(
            sub, name="subcortical_qsm", colormap="gray",
            contrast_limits=(DEFAULT_LOW, DEFAULT_HIGH))

        self._apply_orientation(force_mid=True)
        QTimer.singleShot(120, self._fit_all)

        # Restore QC fields
        case = self.cases[self.case_index]
        rec  = self.results.get(case.case_id, QCRecord(case_id=case.case_id))
        for w in (self.cortex_combo, self.subcortex_combo, self.notes_edit):
            w.blockSignals(True)
        self.cortex_combo.setCurrentText(self._score_to_label(rec.cortex_score))
        self.subcortex_combo.setCurrentText(self._score_to_label(rec.subcortex_score))
        self.notes_edit.setPlainText(rec.notes)
        for w in (self.cortex_combo, self.subcortex_combo, self.notes_edit):
            w.blockSignals(False)

        saved = case.case_id in self.results
        self.case_label.setText(case.case_id)
        if saved:
            self.status_label.setText("Status:  ✔ saved")
            self.status_label.setStyleSheet(f"color:{_C_SUCCESS}; font-size:11px;")
        else:
            self.status_label.setText("Status:  ✏ unsaved")
            self.status_label.setStyleSheet(f"color:{_C_WARN}; font-size:11px;")

        self.source_label.setText(
            f"Cortical: {'file' if data['cort_ok'] else 'gen'}  "
            f"  Subcortical: {'file' if data['sub_ok'] else 'gen'}")
        nat_str  = " → ".join(data["native_axcodes"])
        ck       = self.canonical_combo.currentText()
        reor_str = f"  (→ {ck})" if ck != "Native" else ""
        self.orient_label.setText(f"Native: {nat_str}{reor_str}")

        self.current_saved = saved
        self.case_list.setCurrentRow(self.case_index)
        self.contrast_dlg.reset()
        self._set_nav_enabled(True)
        self._progress_bar.setVisible(False)
        self._set_status(f"Loaded  {case.case_id}", _C_SUCCESS)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    cases = find_cases(CASES_ROOT, FILE_NAMES)
    if not cases:
        raise RuntimeError("No valid cases found. Check CASES_ROOT and FILE_NAMES.")

    # ── Must be set BEFORE QApplication is instantiated ──────────────────
    # AA_ShareOpenGLContexts: lets multiple vispy/napari canvases share one
    #   GL context pool instead of fighting over makeCurrent().
    # AA_UseDesktopOpenGL:    forces native desktop GL (not ANGLE/software),
    #   which is required for vispy on Windows with multiple canvases.
    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
    QApplication.setAttribute(Qt.AA_UseDesktopOpenGL)
    # ─────────────────────────────────────────────────────────────────────

    app = QApplication.instance() or QApplication(sys.argv)
    win = ReviewerMainWindow(cases=cases, output_csv=OUTPUT_CSV)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()