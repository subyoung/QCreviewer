"""
QSM QC Reviewer v21
New features: Label descriptions can be loaded from external text files in ITK-SNAP format, with a fallback search for bundled defaults.  See the "Change Label Descriptions" dialog for details and instructions.
=====================
A napari-based viewer for quality control of QSM processing results, designed to be used in conjunction with the UK Biobank QSM pipeline.  This viewer is intended
to be used by trained human raters to visually assess the quality of QSM outputs and assign categorical quality scores, which can then be exported in a CSV file for downstream analysis.  The viewer provides a multi-panel layout with linked navigation, allowing users to easily compare different image contrasts and segmentations for each case.  The UI includes options for adjusting brightness/contrast, toggling segmentation overlays, and adding free-form notes for each case.  The viewer is built using the napari framework for fast multi-dimensional image visualization, and PyQt for the UI components.  It is designed to be flexible and extensible, allowing for future additions such as new QC criteria or support for additional file formats.

Ziyang Xu, 2026
"""
import os
import sys
import bisect
import shutil
import platform
import re
from pathlib import Path
from dataclasses import dataclass, asdict, fields
from typing import Callable, Dict, List, Optional, Tuple
from collections import OrderedDict

_IS_MACOS   = sys.platform == "darwin"
_IS_WINDOWS = sys.platform == "win32"
_SYSTEM_FONT = ".AppleSystemUIFont, Helvetica Neue, Arial" if _IS_MACOS else "Segoe UI, Arial"

# ═══════════════════════════════════════════════════════════════════════════════
#  TUNABLE SIZING CONSTANTS
#  All visual dimensions are defined here.  Change these to adjust the UI.
#  Two values per constant: (Windows, macOS/Linux)
# ═══════════════════════════════════════════════════════════════════════════════

# -- Font sizes (pt) ----------------------------------------------------------
_BASE_PT         = 7 if _IS_WINDOWS else 11  # primary UI font
_SMALL_PT        =  5 if _IS_WINDOWS else 10  # secondary/dim labels
_STATUS_PT       =  6 if _IS_WINDOWS else  9  # status bar, progress

# -- Canvas header bar --------------------------------------------------------
_HDR_H           = 20 if _IS_WINDOWS else 42
_HDR_COMBO_MIN_H = 8 if _IS_WINDOWS else 28
_HDR_COMBO_MAX_H = 10 if _IS_WINDOWS else 34
_HDR_MARGIN_V    =  0 if _IS_WINDOWS else  5  # stable: hl.setContentsMargins(8,0,8,0)

# -- Direction bar (R/L/A/P labels) ------------------------------------------
_DIR_H           = 12 if _IS_WINDOWS else 26
_DIR_PT          =  6 if _IS_WINDOWS else 10

# -- Global widget sizing -----------------------------------------------------
_GB_MARGIN_TOP   = 8 if _IS_WINDOWS else 22
_GB_PAD_TOP      =  2 if _IS_WINDOWS else  8
_COMBO_MIN_H     = 15 if _IS_WINDOWS else 24
_ITEM_PAD        =  2 if _IS_WINDOWS else  3
_ITEM_MIN_H      = 15 if _IS_WINDOWS else 22
_COMBO_ITEM_PAD  =  1 if _IS_WINDOWS else  4

# -- Panel layout -------------------------------------------------------------
_PANEL_MARGIN    = 1 if _IS_WINDOWS else 10
_PANEL_SPACING   = 1 if _IS_WINDOWS else  8
_STACKED_SPACING =  1 if _IS_WINDOWS else  3

# -- Specific widget heights --------------------------------------------------
_NOTES_MIN_H     = 45 if _IS_WINDOWS else 60
_NOTES_MAX_H     = 75 if _IS_WINDOWS else 110
_CASELIST_MIN_H  = 80 if _IS_WINDOWS else 120
import nibabel as nib
import nibabel.orientations as nibo
import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation
from qtpy.QtCore import QObject, QEvent, QTimer, Qt, QSize, Signal, QThread, QRect, QPoint
from qtpy.QtGui import QFont, QColor, QPainter, QPainterPath, QPen, QPixmap, QIcon
from qtpy.QtWidgets import (
    QAbstractItemView, QAction, QApplication, QCheckBox, QComboBox,
    QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame,
    QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QListView, QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
    QProgressBar, QPushButton, QScrollArea,
    QSizePolicy, QSplitter, QTextEdit,
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
    "cortical_qsm_cube": "QSM_TOTAL_mcpc3Ds_chi_SFCR_Avg_wGDC_cortical_expanded.nii.gz",
    "subcortical_qsm":   "QSM_TOTAL_mcpc3Ds_chi_SFCR_Avg_wGDC_subcortical_expanded.nii.gz",
}
SEGMENTATION_TASK = False  # if True, show segmentation accuracy labeling section
SAVE_GENERATED_MASKED_QSM = True
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
CORTICAL_LABELS_FILENAME = "cortical_labels.txt"
SUBCORTICAL_LABELS_FILENAME = "subcortical_labels.txt"

_LABEL_DESCRIPTION_OVERRIDE_PATHS: Dict[str, Optional[Path]] = {
    "cortical": None,
    "subcortical": None,
}

def _app_base_dir() -> Path:
    try:
        return Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
    except Exception:
        return Path(__file__).resolve().parent

def _pyinstaller_temp_dir() -> Optional[Path]:
    try:
        base = getattr(sys, "_MEIPASS", None)
        return Path(base).resolve() if base else None
    except Exception:
        return None

def _default_label_file_candidates(filename: str) -> List[Path]:
    candidates: List[Path] = []
    seen = set()

    def _add(path_like):
        if not path_like:
            return
        try:
            path = Path(path_like).resolve()
        except Exception:
            return
        key = str(path)
        if key in seen:
            return
        seen.add(key)
        candidates.append(path)

    app_dir = _app_base_dir()
    meipass_dir = _pyinstaller_temp_dir()
    script_dir = Path(__file__).resolve().parent

    _add(app_dir / filename)
    _add(app_dir / "resources" / filename)
    if meipass_dir is not None:
        _add(meipass_dir / filename)
        _add(meipass_dir / "resources" / filename)
    _add(script_dir / filename)
    _add(script_dir / "resources" / filename)
    return candidates

def get_default_label_description_paths() -> Dict[str, Optional[Path]]:
    resolved: Dict[str, Optional[Path]] = {"cortical": None, "subcortical": None}
    for key, filename in (
        ("cortical", CORTICAL_LABELS_FILENAME),
        ("subcortical", SUBCORTICAL_LABELS_FILENAME),
    ):
        for cand in _default_label_file_candidates(filename):
            if cand.exists():
                resolved[key] = cand
                break
    return resolved

def get_effective_label_description_paths() -> Dict[str, Optional[Path]]:
    defaults = get_default_label_description_paths()
    resolved: Dict[str, Optional[Path]] = {}
    for key in ("cortical", "subcortical"):
        override = _LABEL_DESCRIPTION_OVERRIDE_PATHS.get(key)
        if override is not None:
            try:
                override = Path(override).expanduser().resolve()
            except Exception:
                override = None
        resolved[key] = override if (override is not None and override.exists()) else defaults.get(key)
    return resolved

def set_label_description_override_paths(cortical_path: Optional[str] = None,
                                         subcortical_path: Optional[str] = None,
                                         use_default: bool = False) -> None:
    global _LABEL_DESCRIPTION_OVERRIDE_PATHS
    if use_default:
        _LABEL_DESCRIPTION_OVERRIDE_PATHS = {"cortical": None, "subcortical": None}
        return

    def _norm(path_str: Optional[str]) -> Optional[Path]:
        raw = (path_str or "").strip()
        if not raw:
            return None
        try:
            return Path(raw).expanduser().resolve()
        except Exception:
            return None

    _LABEL_DESCRIPTION_OVERRIDE_PATHS = {
        "cortical": _norm(cortical_path),
        "subcortical": _norm(subcortical_path),
    }

def using_default_label_description_paths() -> bool:
    return all(v is None for v in _LABEL_DESCRIPTION_OVERRIDE_PATHS.values())

def _parse_itksnap_visible_label_file(path: Path) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith('#'):
                    continue
                m = re.match(r'^(-?\d+)\s+\d+\s+\d+\s+\d+\s+([0-9]*\.?[0-9]+)\s+(\d+)\s+(\d+)\s+"(.*)"\s*$', line)
                if not m:
                    continue
                idx = int(m.group(1))
                vis = int(m.group(3))
                label_name = m.group(5).strip()
                if vis == 1 and idx > 0 and label_name:
                    mapping[idx] = label_name
    except FileNotFoundError:
        pass
    except Exception:
        pass
    return mapping

def load_hover_label_maps() -> Tuple[Dict[int, str], Dict[int, str]]:
    paths = get_effective_label_description_paths()
    cortical_path = paths.get("cortical")
    subcortical_path = paths.get("subcortical")
    cortical = _parse_itksnap_visible_label_file(cortical_path) if cortical_path else {}
    subcortical = _parse_itksnap_visible_label_file(subcortical_path) if subcortical_path else {}
    return cortical, subcortical

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
    ("RAS",            "Axial"):    (True,  True),
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
QMainWindow, QWidget {{
    background: {_C_BG}; color: {_C_TEXT};
    font-family: {_SYSTEM_FONT}; font-size: {_BASE_PT}pt;
}}
QSplitter::handle           {{ background: {_C_BORDER}; }}
QSplitter::handle:horizontal {{ width: 5px; }}
QSplitter::handle:vertical   {{ height: 5px; }}
QGroupBox {{
    border: 1px solid {_C_BORDER}; border-radius: 5px;
    margin-top: {_GB_MARGIN_TOP}px; padding-top: {_GB_PAD_TOP}px;
    font-size: {_BASE_PT}pt; font-weight: 600; color: {_C_ACCENT};
}}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; }}
QLabel           {{ font-size: {_BASE_PT}pt; }}
QComboBox {{
    background: {_C_PANEL}; color: {_C_TEXT}; border: 1px solid {_C_BORDER};
    border-radius: 4px; padding: {'4px 10px' if _IS_WINDOWS else '3px 8px'}; font-size: {_BASE_PT}pt;
    min-height: {_COMBO_MIN_H}px;
}}
QComboBox:hover {{ border-color: {_C_ACCENT}; }}
QComboBox QAbstractItemView {{
    background: {_C_PANEL}; color: {_C_TEXT};
    selection-background-color: {_C_ACCENT}; font-size: {_BASE_PT}pt;
}}
QComboBox QAbstractItemView::item {{
    padding: {_COMBO_ITEM_PAD}px 10px; min-height: {_ITEM_MIN_H}px;
}}
QComboBox QAbstractItemView::item:hover {{ background: {_C_HEADER}; color: {_C_TEXT}; }}
QListWidget {{
    background: {_C_PANEL}; color: {_C_TEXT}; border: 1px solid {_C_BORDER};
    border-radius: 4px; font-size: {_BASE_PT}pt;
}}
QListWidget::item           {{ padding: {_ITEM_PAD}px 6px; min-height: {_ITEM_MIN_H}px; }}
QListWidget::item:selected  {{ background: {_C_ACCENT}; color: white; }}
QListWidget::item:hover     {{ background: {_C_HEADER}; }}
QTextEdit {{
    background: {_C_PANEL}; color: {_C_TEXT}; border: 1px solid {_C_BORDER};
    border-radius: 4px; font-size: {_BASE_PT}pt; padding: {'5px' if _IS_WINDOWS else '3px'};
}}
QPushButton {{
    background: {_C_PANEL}; color: {_C_TEXT}; border: 1px solid {_C_BORDER};
    border-radius: 4px; padding: {'6px 16px' if _IS_WINDOWS else '4px 12px'}; font-size: {_BASE_PT}pt;
}}
QPushButton:hover   {{ background: {_C_HEADER}; border-color: {_C_ACCENT}; }}
QPushButton:pressed {{ background: {_C_ACCENT}; color: white; }}
QPushButton:disabled {{ color: {_C_DIM}; border-color: {_C_BORDER}; }}
QCheckBox           {{ color: {_C_TEXT}; font-size: {_BASE_PT}pt; spacing: 6px; }}
{"QCheckBox:hover     { color: white; }" if not _IS_WINDOWS else ""}
QDoubleSpinBox {{
    background: {_C_PANEL}; color: {_C_TEXT}; border: 1px solid {_C_BORDER};
    border-radius: 4px; padding: 3px 8px; font-size: {_BASE_PT}pt;
}}
{"QLineEdit {" if not _IS_WINDOWS else ""}
{"    background: " + _C_PANEL + "; color: " + _C_TEXT + "; border: 1px solid " + _C_BORDER + ";" if not _IS_WINDOWS else ""}
{"    border-radius: 4px; padding: 3px 8px; font-size: " + str(_BASE_PT) + "pt;" if not _IS_WINDOWS else ""}
{"}" if not _IS_WINDOWS else ""}
{"QLineEdit:focus { border-color: " + _C_ACCENT + "; }" if not _IS_WINDOWS else ""}
QScrollArea        {{ border: none; background: transparent; }}
QScrollBar:vertical {{
    background: {_C_BG}; width: 8px; border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background: {_C_BORDER}; border-radius: 4px; min-height: {'24px' if _IS_WINDOWS else '20px'};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}
QStatusBar         {{ background: {_C_HEADER}; color: {_C_DIM}; font-size: {_STATUS_PT}pt; }}
QProgressBar {{
    background: {_C_PANEL}; border: 1px solid {_C_BORDER}; border-radius: 3px;
    text-align: center; color: {_C_TEXT}; font-size: {_STATUS_PT}pt; max-height: {'16px' if _IS_WINDOWS else '14px'};
}}
QProgressBar::chunk {{ background: {_C_ACCENT}; border-radius: 2px; }}
QMenuBar            {{ background: {_C_HEADER}; color: {_C_TEXT}; font-size: {_BASE_PT}pt; }}
{"QMenuBar::item      { padding: 4px 10px; }" if not _IS_WINDOWS else ""}
QMenuBar::item:selected {{ background: {_C_ACCENT}; }}
QMenu               {{ background: {_C_PANEL}; color: {_C_TEXT};
                       border: 1px solid {_C_BORDER}; font-size: {_BASE_PT}pt; }}
{"QMenu::item         { padding: 5px 28px 5px 28px; }" if not _IS_WINDOWS else ""}
{"QMenu::item:checked { padding: 5px 28px 5px 10px; }" if not _IS_WINDOWS else ""}
QMenu::item:selected {{ background: {_C_ACCENT}; }}
{"QDialog { background: " + _C_BG + "; color: " + _C_TEXT + "; }" if not _IS_WINDOWS else ""}
{"QDialogButtonBox QPushButton { min-width: 70px; padding: 4px 12px; }" if not _IS_WINDOWS else ""}
"""
_SAVE_BTN_CSS = f"""
QPushButton {{
    background: {_C_ACCENT}; color: white; border: none;
    border-radius: 4px; padding: {'9px 18px' if _IS_WINDOWS else '6px 14px'}; font-size: {_BASE_PT+1}pt; font-weight: 600;
}}
QPushButton:hover   {{ background: #5aabff; }}
QPushButton:pressed {{ background: #3a8eef; }}
QPushButton:disabled {{ background: {_C_BORDER}; color: {_C_DIM}; }}
"""
_NAV_BTN_CSS = f"""
QPushButton {{
    background: {_C_PANEL}; color: {_C_TEXT}; border: 1px solid {_C_BORDER};
    border-radius: 4px; padding: {'7px 20px' if _IS_WINDOWS else '4px 12px'}; font-size: {_BASE_PT}pt;
}}
QPushButton:hover   {{ background: {_C_HEADER}; border-color: {_C_ACCENT}; color: {_C_ACCENT}; }}
QPushButton:disabled {{ color: {_C_DIM}; border-color: {_C_BORDER}; }}
"""
_QC_ERROR_STYLE = f"""
QComboBox {{
    background: {_C_PANEL}; color: {_C_TEXT};
    border: 2px solid #d65c5c; border-radius: 4px; padding: 4px 10px; font-size: {_BASE_PT}pt;
}}
QComboBox QAbstractItemView {{
    background: {_C_PANEL}; color: {_C_TEXT};
    selection-background-color: {_C_ACCENT}; font-size: {_BASE_PT}pt;
}}
"""
ZOOM_PRESETS = [25, 50, 75, 100, 125, 150, 200, 300]

# ── Styled combo-box that always opens its popup BELOW the button ─────────────
#
# On macOS, the default QComboBox.showPopup() positions the popup so the
# *currently selected item* sits at the cursor, meaning items above the
# selection appear above the button.  Overriding showPopup() and calling
# QComboBox.showPopup() after resetting the internal scroll-to-current logic
# is not reliable; the cleanest cross-platform fix is to calculate the
# geometry ourselves and call QAbstractItemView.setGeometry() directly.
#
# We also install a plain QListView so that Qt renders the popup with our
# CSS hover rules on every platform (macOS native popup ignores CSS).

_COMBO_POPUP_CSS = f"""
QListView {{
    background: {_C_PANEL}; color: {_C_TEXT};
    border: 1px solid {_C_BORDER}; border-radius: 4px;
    outline: none; padding: 4px 0;
}}
QListView::item {{
    padding: 2px 10px; min-height: 18px;
    color: {_C_TEXT};
}}
QListView::item:selected, QListView::item:selected:hover {{
    background: {_C_ACCENT}; color: white;
}}
QListView::item:hover {{
    background: {_C_HEADER}; color: white;
}}
"""

class StyledComboBox(QComboBox):
    """
    QComboBox subclass that:
    - Always opens its popup BELOW the widget (never above, never centred).
    - Uses a plain QListView so CSS hover styling works on macOS.
    - Exposes a convenient class method for construction with items.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMaxVisibleItems(20)
        lv = QListView()
        lv.setMouseTracking(True)
        lv.setStyleSheet(_COMBO_POPUP_CSS)
        self.setView(lv)

    # ------------------------------------------------------------------
    # Force popup to appear BELOW the combo box button on all platforms
    # ------------------------------------------------------------------
    def showPopup(self):
        # Let Qt do its default setup (model, delegate, size hints …)
        # but then immediately correct the position.
        super().showPopup()
        popup = self.findChild(QAbstractItemView)   # the floating popup frame
        if popup is None:
            return
        container = popup.window()   # the QFrame that wraps the QListView
        if container is None:
            return

        # Global position of bottom-left corner of this combo box
        bottom_left = self.mapToGlobal(QPoint(0, self.height()))

        # Keep the same width and height Qt calculated
        cw = max(container.width(), self.width())
        ch = container.height()

        # Clamp to screen so we don't go off the bottom
        screen = QApplication.screenAt(bottom_left)
        if screen is None:
            screen = QApplication.primaryScreen()
        screen_geom = screen.availableGeometry()

        x = bottom_left.x()
        y = bottom_left.y()

        # If no room below, flip above (fallback only)
        if y + ch > screen_geom.bottom():
            y = self.mapToGlobal(QPoint(0, 0)).y() - ch

        # Clamp horizontal so popup doesn't go off-screen right
        if x + cw > screen_geom.right():
            x = screen_geom.right() - cw

        x = max(screen_geom.left(), x)
        y = max(screen_geom.top(), y)

        container.setGeometry(QRect(x, y, cw, ch))


def _make_combo(items: Optional[List[str]] = None,
                placeholder: Optional[str] = None,
                min_width: int = 0,
                max_width: int = 0,
                expanding: bool = True) -> StyledComboBox:
    """Build a StyledComboBox with optional item list and size constraints."""
    cb = StyledComboBox()
    if placeholder is not None:
        cb.addItem(placeholder)
    if items:
        cb.addItems(items)
    if min_width:
        cb.setMinimumWidth(min_width)
    if max_width:
        cb.setMaximumWidth(max_width)
    policy = QSizePolicy.Expanding if expanding else QSizePolicy.Fixed
    cb.setSizePolicy(policy, QSizePolicy.Fixed)
    return cb


@dataclass
class CasePaths:
    case_id:           str
    case_dir:          str
    raw_qsm:           str
    segmentation:      str
    subcortical_label: str
    cortical_qsm:      Optional[str] = None
    cortical_qsm_cube: Optional[str] = None
    subcortical_qsm:   Optional[str] = None
@dataclass
class QCRecord:
    case_id:           str
    cortex_score:      str = ""
    subcortex_score:   str = ""
    cort_label_ok:     str = ""   # "0"=good, "1"=bad, ""=not assessed
    sub_label_ok:      str = ""
    notes:             str = ""
    marked_for_review: str = ""   # "1"=flagged, ""=not flagged
@dataclass
class AppConfig:
    cases_root: str
    output_csv: str
    file_names: Dict[str, str]
    save_generated_qsm: bool = SAVE_GENERATED_MASKED_QSM
    show_seg_in_derived_views: bool = False


def _make_app_icon() -> QIcon:
    """Use napari's app icon when available; otherwise fall back to the local Q logo."""
    try:
        import importlib.resources as ilr
        try:
            import napari.resources
            root = ilr.files(napari.resources)
            candidates = [
                "logo.png",
                "logo_silhouette.png",
                "napari_logo.png",
                "icon.png",
                "icon.icns",
                "icon.ico",
            ]
            for name in candidates:
                try:
                    res = root / name
                    if res.is_file():
                        return QIcon(str(res))
                except Exception:
                    pass
            try:
                for res in root.rglob('*'):
                    if res.name.lower() in {"logo.png", "napari_logo.png", "icon.png", "icon.ico"}:
                        return QIcon(str(res))
            except Exception:
                pass
        except Exception:
            pass
    except Exception:
        pass
    return QIcon(_make_logo_pixmap(64))

def _make_logo_pixmap(size: int = 22) -> QPixmap:
    pix = QPixmap(size, size)
    pix.fill(Qt.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.Antialiasing)
    rect = pix.rect().adjusted(1, 1, -1, -1)
    path = QPainterPath()
    path.addRoundedRect(float(rect.x()), float(rect.y()), float(rect.width()), float(rect.height()), 6.0, 6.0)
    painter.fillPath(path, QColor(_C_ACCENT))
    painter.setPen(QPen(QColor("#9bc7ff"), 1.2))
    painter.drawPath(path)
    font = QFont("Segoe UI", max(7, size // 3))
    font.setBold(True)
    painter.setFont(font)
    painter.setPen(QColor("white"))
    painter.drawText(rect, Qt.AlignCenter, "Q")
    painter.end()
    return pix

class LabelDescriptionConfigDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Change Label Descriptions")
        self.setWindowIcon(_make_app_icon())
        self.setModal(True)
        self.setMinimumWidth(560 if _IS_WINDOWS else 680)
        self.setStyleSheet(_GLOBAL_CSS)

        paths = get_effective_label_description_paths()
        defaults = get_default_label_description_paths()

        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 12, 16, 12)
        lay.setSpacing(10)

        title = QLabel("Label description files")
        title.setStyleSheet(f"font-size:{_BASE_PT + 2}pt; font-weight:700; color:{_C_ACCENT};")
        lay.addWidget(title)

        hint = QLabel(
            "Choose custom cortical and subcortical label description files, or use the default bundled files."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color:{_C_DIM}; font-size:{_SMALL_PT}pt;")
        lay.addWidget(hint)

        self.use_default_cb = QCheckBox("Use default label descriptions")
        self.use_default_cb.setChecked(using_default_label_description_paths())
        lay.addWidget(self.use_default_cb)

        box = QGroupBox("Files")
        box_lay = QVBoxLayout(box)
        box_lay.setContentsMargins(10, 12, 10, 10)
        box_lay.setSpacing(8)

        self.cortical_edit = QLineEdit(str(paths.get("cortical") or ""))
        self.subcortical_edit = QLineEdit(str(paths.get("subcortical") or ""))

        for label_text, edit, browse_cb in (
            ("Cortical labels:", self.cortical_edit, self._browse_cortical),
            ("Subcortical labels:", self.subcortical_edit, self._browse_subcortical),
        ):
            row = QHBoxLayout()
            row.setSpacing(8)
            lbl = QLabel(label_text)
            lbl.setMinimumWidth(130)
            lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            btn = QPushButton("Browse…")
            btn.setMinimumWidth(84)
            btn.clicked.connect(browse_cb)
            row.addWidget(lbl)
            row.addWidget(edit, 1)
            row.addWidget(btn)
            box_lay.addLayout(row)

        self.default_info = QLabel()
        self.default_info.setWordWrap(True)
        self.default_info.setStyleSheet(f"color:{_C_DIM}; font-size:{_SMALL_PT}pt;")
        default_cort = str(defaults.get("cortical") or "(not found)")
        default_sub = str(defaults.get("subcortical") or "(not found)")
        self.default_info.setText(
            f"Default cortical: {default_cort}\nDefault subcortical: {default_sub}"
        )
        box_lay.addWidget(self.default_info)
        lay.addWidget(box)

        btns = QDialogButtonBox()
        self.apply_btn = btns.addButton("Apply", QDialogButtonBox.AcceptRole)
        cancel_btn = btns.addButton("Cancel", QDialogButtonBox.RejectRole)
        self.apply_btn.clicked.connect(self._on_apply)
        cancel_btn.clicked.connect(self.reject)
        lay.addWidget(btns)

        self.use_default_cb.toggled.connect(self._on_use_default_toggled)
        self._on_use_default_toggled(self.use_default_cb.isChecked())

    def _browse_file(self, edit: QLineEdit, title: str):
        start = edit.text().strip()
        if not start:
            defaults = get_default_label_description_paths()
            start = str((defaults.get("cortical") or defaults.get("subcortical") or _app_base_dir()))
        path, _ = QFileDialog.getOpenFileName(
            self,
            title,
            start,
            "Text files (*.txt);;All files (*)",
        )
        if path:
            edit.setText(path)

    def _browse_cortical(self):
        self._browse_file(self.cortical_edit, "Select cortical label description file")

    def _browse_subcortical(self):
        self._browse_file(self.subcortical_edit, "Select subcortical label description file")

    def _on_use_default_toggled(self, checked: bool):
        self.cortical_edit.setEnabled(not checked)
        self.subcortical_edit.setEnabled(not checked)

    def _on_apply(self):
        if self.use_default_cb.isChecked():
            self.accept()
            return

        missing = []
        for name, edit in (("cortical", self.cortical_edit), ("subcortical", self.subcortical_edit)):
            path = edit.text().strip()
            if not path:
                missing.append(name)
                continue
            if not Path(path).expanduser().exists():
                QMessageBox.warning(self, "File not found", f"The selected {name} label file does not exist:\n{path}")
                return

        if missing:
            QMessageBox.warning(
                self,
                "Missing files",
                "Please select both cortical and subcortical label description files, or enable 'Use default label descriptions'.",
            )
            return
        self.accept()

    def selected_paths(self) -> Dict[str, Optional[str]]:
        if self.use_default_cb.isChecked():
            return {"use_default": True, "cortical": None, "subcortical": None}
        return {
            "use_default": False,
            "cortical": self.cortical_edit.text().strip() or None,
            "subcortical": self.subcortical_edit.text().strip() or None,
        }


class StartupConfigDialog(QDialog):
    def __init__(self, parent=None, defaults_root: str = CASES_ROOT,
                 defaults_output_csv: str = OUTPUT_CSV,
                 defaults_file_names: Optional[Dict[str, str]] = None):
        super().__init__(parent)
        self.setWindowTitle("QSM QC Reviewer - Path Setup")
        self.setWindowIcon(_make_app_icon())
        # Size to fit within the screen, no larger than needed
        screen = QApplication.primaryScreen()
        if _IS_WINDOWS:
            # Windows: fixed sizes matching stable version
            self.setMinimumWidth(500)
        else:
            # macOS: adaptive sizing
            self.setMinimumWidth(640)
            if screen is not None:
                sg = screen.availableGeometry()
                dlg_w = min(int(sg.width() * 0.75), 960)
                dlg_h = min(int(sg.height() * 0.85), 780)
            else:
                dlg_w, dlg_h = (860, 680)
            self.resize(dlg_w, dlg_h)
            self.setSizeGripEnabled(True)   # user can resize
        self.setModal(True)
        self.setStyleSheet(_GLOBAL_CSS)
        self._defaults_file_names = dict(defaults_file_names or FILE_NAMES)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 8, 16, 12) #the first is left, the second is top, the third is right, and the fourth is bottom
        lay.setSpacing(6) # vertical spacing between sections
        title_row = QHBoxLayout()
        logo = QLabel()
        logo.setPixmap(_make_app_icon().pixmap(28, 28) if _IS_WINDOWS else _make_app_icon().pixmap(36, 36))
        title = QLabel("Launch Settings")
        title.setStyleSheet(f"font-size: {'15' if _IS_WINDOWS else str(_BASE_PT+15)}pt; font-weight: 700; color: {_C_ACCENT};")
        # subtitle = QLabel("Set the data folder, CSV file name, display options, and the six file names before entering the labeling UI.")
        # subtitle.setStyleSheet(f"color: {_C_DIM};{'font-size:' + str(_BASE_PT) + 'pt;' if not _IS_WINDOWS else ''}")
        # subtitle.setWordWrap(True)
        title_col = QVBoxLayout()
        if not _IS_WINDOWS:
            title_col.setSpacing(2)
        title_col.addWidget(title)
        # title_col.addWidget(subtitle)
        title_row.addWidget(logo)
        title_row.addSpacing(8)
        title_row.addLayout(title_col, 1)
        lay.addLayout(title_row)
        box = QGroupBox("Paths")
        box_lay = QVBoxLayout(box)
        box_lay.setSpacing(10)

        # Data folder row - explicit horizontal layout to guarantee same-line alignment
        folder_row = QHBoxLayout()
        folder_row.setSpacing(8)
        folder_lbl = QLabel("Data folder:")
        folder_lbl.setMinimumWidth(120)
        folder_lbl.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        folder_lbl.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
        self.root_edit = QLineEdit(defaults_root)
        self.root_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        browse_btn = QPushButton("Browse…")
        browse_btn.setMinimumWidth(80)
        browse_btn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        browse_btn.clicked.connect(self._browse_root)
        folder_row.addWidget(folder_lbl)
        folder_row.addWidget(self.root_edit, 1)
        folder_row.addWidget(browse_btn)
        box_lay.addLayout(folder_row)

        # CSV file name row
        csv_row = QHBoxLayout()
        csv_row.setSpacing(8)
        csv_lbl = QLabel("CSV file name:")
        csv_lbl.setMinimumWidth(120)
        csv_lbl.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        csv_lbl.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
        csv_default_name = os.path.basename(defaults_output_csv) if defaults_output_csv else "qc_results.csv"
        self.csv_edit = QLineEdit(csv_default_name)
        self.csv_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        csv_row.addWidget(csv_lbl)
        csv_row.addWidget(self.csv_edit, 1)
        # Invisible spacer to align with folder row; matches browse_btn minimum
        _csv_spacer = QWidget()
        _csv_spacer.setMinimumWidth(80)
        _csv_spacer.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        csv_row.addWidget(_csv_spacer)
        box_lay.addLayout(csv_row)
        lay.addWidget(box)
        files_box = QGroupBox("Data file names")
        files_grid = QGridLayout(files_box)
        files_grid.setHorizontalSpacing(10 if _IS_WINDOWS else 12)
        files_grid.setVerticalSpacing(8 if _IS_WINDOWS else 10)
        if not _IS_WINDOWS:
            files_grid.setColumnMinimumWidth(0, 160)
        labels = [
            ("raw_qsm", "Raw QSM:"),
            ("segmentation", "Segmentation:"),
            ("subcortical_label", "Subcortical label:"),
            ("cortical_qsm", "Cortical QSM:"),
            ("cortical_qsm_cube", "Cortical QSM (expanded):"),
            ("subcortical_qsm", "Subcortical QSM:"),
        ]
        self.file_edits: Dict[str, QLineEdit] = {}
        for row, (key, label_text) in enumerate(labels):
            lbl = QLabel(label_text)
            lbl.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
            edit = QLineEdit(self._defaults_file_names.get(key, ""))
            edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self.file_edits[key] = edit
            files_grid.addWidget(lbl, row, 0)
            files_grid.addWidget(edit, row, 1)
        lay.addWidget(files_box)
        opt_box = QGroupBox("Startup options")
        opt_lay = QVBoxLayout(opt_box)
        self.save_generated_cb = QCheckBox("Save generated QSM files")
        self.save_generated_cb.setChecked(SAVE_GENERATED_MASKED_QSM)
        self.show_seg_derived_cb = QCheckBox("Show segmentation in cortical/subcortical QSM views")
        self.show_seg_derived_cb.setChecked(False)
        opt_lay.addWidget(self.save_generated_cb)
        opt_lay.addWidget(self.show_seg_derived_cb)
        lay.addWidget(opt_box)
        btns = QDialogButtonBox()
        self.continue_btn = btns.addButton("Continue", QDialogButtonBox.AcceptRole)
        reset_btn = btns.addButton("Reset defaults", QDialogButtonBox.ResetRole)
        cancel_btn = btns.addButton("Cancel", QDialogButtonBox.RejectRole)
        self.continue_btn.clicked.connect(self._on_continue)
        reset_btn.clicked.connect(self._reset_defaults)
        cancel_btn.clicked.connect(self.reject)
        lay.addWidget(btns)
        self.config: Optional[AppConfig] = None
    @staticmethod
    def _wrap_layout(inner_layout):
        w = QWidget()
        w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        w.setLayout(inner_layout)
        return w
    def _browse_root(self):
        path = QFileDialog.getExistingDirectory(self, "Select data folder", self.root_edit.text().strip() or CASES_ROOT)
        if path:
            self.root_edit.setText(path)
    def _reset_defaults(self):
        self.root_edit.setText(CASES_ROOT)
        self.csv_edit.setText(os.path.basename(OUTPUT_CSV))
        for key, edit in self.file_edits.items():
            edit.setText(FILE_NAMES.get(key, ""))
        self.save_generated_cb.setChecked(SAVE_GENERATED_MASKED_QSM)
        self.show_seg_derived_cb.setChecked(False)
    def _on_continue(self):
        root = self.root_edit.text().strip()
        csv_name = self.csv_edit.text().strip()
        if not root:
            QMessageBox.warning(self, "Missing data folder", "Please set the data folder path.")
            return
        if not os.path.isdir(root):
            QMessageBox.warning(self, "Invalid data folder", f"Folder does not exist:\n{root}")
            return
        if not csv_name:
            QMessageBox.warning(self, "Missing CSV name", "Please set the CSV save file name.")
            return
        if os.path.basename(csv_name) != csv_name:
            QMessageBox.warning(self, "CSV name only", "Please enter a file name only, not a full path.")
            return
        if not csv_name.lower().endswith(".csv"):
            csv_name += ".csv"
            self.csv_edit.setText(csv_name)
        file_names = {k: e.text().strip() for k, e in self.file_edits.items()}
        missing = [k for k, v in file_names.items() if not v]
        if missing:
            QMessageBox.warning(self, "Missing file names", "Please fill in all six data file names.")
            return
        self.config = AppConfig(
            cases_root=root,
            output_csv=os.path.join(root, csv_name),
            file_names=file_names,
            save_generated_qsm=self.save_generated_cb.isChecked(),
            show_seg_in_derived_views=self.show_seg_derived_cb.isChecked(),
        )
        self.accept()
# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────
def iter_valid_cases(root: str, file_names: Dict[str, str]):
    if not os.path.isdir(root):
        raise FileNotFoundError(f"CASES_ROOT not found: {root}")
    required = ["raw_qsm", "segmentation", "subcortical_label"]
    with os.scandir(root) as it:
        for entry in it:
            try:
                if not entry.is_dir():
                    continue
                d = entry.path
                paths = {k: os.path.join(d, v) for k, v in file_names.items()}
                if not all(os.path.exists(paths[k]) for k in required):
                    continue
                yield CasePaths(
                    case_id=entry.name, case_dir=d,
                    raw_qsm=paths["raw_qsm"],
                    segmentation=paths["segmentation"],
                    subcortical_label=paths["subcortical_label"],
                    cortical_qsm=paths["cortical_qsm"] if os.path.exists(paths["cortical_qsm"]) else None,
                    cortical_qsm_cube=paths["cortical_qsm_cube"] if os.path.exists(paths["cortical_qsm_cube"]) else None,
                    subcortical_qsm=paths["subcortical_qsm"] if os.path.exists(paths["subcortical_qsm"]) else None,
                )
            except OSError:
                continue

def find_cases(root: str, file_names: Dict[str, str]) -> List[CasePaths]:
    cases = list(iter_valid_cases(root, file_names))
    cases.sort(key=lambda c: c.case_id)
    return cases


class CaseScanWorker(QObject):
    case_found = Signal(object)
    finished = Signal(int)
    failed = Signal(str)

    def __init__(self, *, root: str, file_names: Dict[str, str]):
        super().__init__()
        self.root = root
        self.file_names = dict(file_names)

    def run(self):
        total = 0
        try:
            for case in iter_valid_cases(self.root, self.file_names):
                self.case_found.emit(case)
                total += 1
            self.finished.emit(total)
        except Exception as exc:
            self.failed.emit(str(exc))
def _normalize_saved_choice(val: str) -> str:
    s = ("" if val is None else str(val)).strip()
    if not s:
        return ""
    head = s[0]
    if head in {"0", "1", "2", "3"}:
        return head
    if s.lower() in {"good", "bad"}:
        return "1" if s.lower() == "bad" else "0"
    return s

def read_existing_results(csv_path: str) -> Dict[str, QCRecord]:
    results: Dict[str, QCRecord] = {}
    if not os.path.exists(csv_path):
        return results
    try:
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    except Exception:
        df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        def _safe_str(val):
            if val != val or str(val).lower() == "nan":
                return ""
            return str(val).strip()
        rec = QCRecord(
            case_id=_safe_str(row.get("case_id", "")),
            cortex_score=_normalize_saved_choice(_safe_str(row.get("cortex_score", ""))),
            subcortex_score=_normalize_saved_choice(_safe_str(row.get("subcortex_score", ""))),
            cort_label_ok=_normalize_saved_choice(_safe_str(row.get("cort_label_ok", ""))),
            sub_label_ok=_normalize_saved_choice(_safe_str(row.get("sub_label_ok", ""))),
            notes=_safe_str(row.get("notes", "")),
            marked_for_review=("1" if _safe_str(row.get("marked_for_review", "")).lower() in {"1", "true", "yes", "y", "flagged", "review"} else ""),
        )
        if rec.case_id:
            results[rec.case_id] = rec
    return results
def write_results(csv_path: str, records: Dict[str, QCRecord]) -> None:
    rows = [asdict(v) for _, v in sorted(records.items(), key=lambda x: x[0])]
    pd.DataFrame(
        rows, columns=[f.name for f in fields(QCRecord)]
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
        cube = np.zeros_like(label, dtype=np.uint8)
        return np.zeros_like(qsm), cube
    coords = np.where(mask)
    slices = tuple(
        slice(max(0, coords[ax].min() - margin),
              min(mask.shape[ax], coords[ax].max() + margin + 1))
        for ax in range(3)
    )
    cube = np.zeros_like(mask, dtype=np.uint8)
    cube[slices] = 1
    return qsm * cube, cube

def generate_cortical_qsm(qsm, seg, roi_dict, dilation_iter=2):
    labels = sorted({l for v in roi_dict.values() for l in v})
    return qsm * binary_dilation(np.isin(seg, labels), iterations=dilation_iter)

def generate_cortical_qsm_cube(qsm, seg, roi_dict, cube_mask, dilation_iter=2):
    labels = sorted({l for v in roi_dict.values() for l in v})
    cortical_roi_mask = binary_dilation(np.isin(seg, labels), iterations=dilation_iter)
    cube_outside_mask = ~(np.asarray(cube_mask) > 0)
    combined_mask = cortical_roi_mask | cube_outside_mask
    return qsm * combined_mask
def load_case_data(case: CasePaths,
                   canonical_key: str,
                   save_generated: bool,
                   subcortical_margin: int,
                   cortical_dilation_iter: int,
                   display_mode: str,
                   progress_cb: Callable[[int, str], None],
                   file_names: Dict[str, str]):
    """Heavy data loading / generation, safe to run in a worker thread."""
    target_axcodes = CANONICAL_OPTIONS[canonical_key]

    # Warm cloud-synced (OneDrive/iCloud) stubs before heavy reads
    all_paths = [
        case.raw_qsm, case.segmentation, case.subcortical_label,
        case.cortical_qsm, case.cortical_qsm_cube, case.subcortical_qsm,
    ]
    _warm_onedrive_files([p for p in all_paths if p], timeout=10.0)

    progress_cb(5, "Loading raw QSM…")
    raw_img_native = nib.load(case.raw_qsm)
    native_axcodes = nibo.aff2axcodes(raw_img_native.affine)
    raw_img = reorient_image(raw_img_native, target_axcodes)
    qsm_data = np.asarray(raw_img.get_fdata(dtype=np.float32), dtype=np.float32)
    reoriented_affine = raw_img.affine
    zooms = tuple(float(np.linalg.norm(raw_img.affine[:3, i])) for i in range(3))

    progress_cb(22, "Loading segmentation…")
    seg_img = reorient_image(nib.load(case.segmentation), target_axcodes)
    seg_data = np.asarray(seg_img.get_fdata(), dtype=np.int32)

    progress_cb(36, "Loading subcortical labels…")
    sub_lbl_img = reorient_image(nib.load(case.subcortical_label), target_axcodes)
    sub_lbl_data = np.asarray(sub_lbl_img.get_fdata(), dtype=np.int32)

    cort_roi_ok = bool(case.cortical_qsm and os.path.exists(case.cortical_qsm))
    cort_cube_ok = bool(case.cortical_qsm_cube and os.path.exists(case.cortical_qsm_cube))
    sub_ok = bool(case.subcortical_qsm and os.path.exists(case.subcortical_qsm))
    generated_paths = {}

    progress_cb(50, "Loading subcortical QSM…")
    cube_mask = None
    if sub_ok:
        sub_data = np.asarray(
            reorient_image(nib.load(case.subcortical_qsm), target_axcodes).get_fdata(dtype=np.float32),
            dtype=np.float32,
        )
        cube_mask = (sub_data != 0).astype(np.uint8)
    else:
        progress_cb(60, "Generating subcortical mask…")
        sub_data, cube_mask = generate_subcortical_qsm(qsm_data, sub_lbl_data, subcortical_margin)
        sub_data = sub_data.astype(np.float32, copy=False)
        if save_generated:
            p = os.path.join(case.case_dir, file_names["subcortical_qsm"])
            nib.save(nib.Nifti1Image(sub_data, raw_img.affine, raw_img.header), p)
            generated_paths["subcortical_qsm"] = p
            sub_ok = True

    progress_cb(70, "Loading cortical QSMs…")
    if cort_roi_ok:
        cort_roi_data = np.asarray(
            reorient_image(nib.load(case.cortical_qsm), target_axcodes).get_fdata(dtype=np.float32),
            dtype=np.float32,
        )
    else:
        progress_cb(76, "Generating cortical ROI QSM…")
        cort_roi_data = generate_cortical_qsm(
            qsm_data, seg_data, ROI_LABEL_SYNTHSEG_COMBINED, cortical_dilation_iter
        ).astype(np.float32, copy=False)

    if cort_cube_ok:
        cort_cube_data = np.asarray(
            reorient_image(nib.load(case.cortical_qsm_cube), target_axcodes).get_fdata(dtype=np.float32),
            dtype=np.float32,
        )
    else:
        progress_cb(84, "Generating cortical expanded QSM…")
        if cube_mask is None:
            _, cube_mask = generate_subcortical_qsm(qsm_data, sub_lbl_data, subcortical_margin)
        cort_cube_data = generate_cortical_qsm_cube(
            qsm_data, seg_data, ROI_LABEL_SYNTHSEG_COMBINED, cube_mask, cortical_dilation_iter
        ).astype(np.float32, copy=False)

    if save_generated:
        if not cort_roi_ok:
            p = os.path.join(case.case_dir, file_names["cortical_qsm"])
            nib.save(nib.Nifti1Image(cort_roi_data, raw_img.affine, raw_img.header), p)
            generated_paths["cortical_qsm"] = p
            cort_roi_ok = True
        if not cort_cube_ok:
            p = os.path.join(case.case_dir, file_names["cortical_qsm_cube"])
            nib.save(nib.Nifti1Image(cort_cube_data, raw_img.affine, raw_img.header), p)
            generated_paths["cortical_qsm_cube"] = p
            cort_cube_ok = True

    cort_data = cort_roi_data if display_mode == "ROI only" else cort_cube_data
    progress_cb(92, "Building seg overlays…")
    cortical_seg = np.where(np.isin(seg_data, _ALL_CORTICAL_LABELS), seg_data, 0).astype(np.int32, copy=False)
    progress_cb(98, "Ready…")
    return dict(
        raw=ensure_3d(qsm_data, "raw_qsm"),
        cort=ensure_3d(cort_data, "cortical_qsm"),
        cort_roi=ensure_3d(cort_roi_data, "cortical_qsm_roi"),
        cort_cube=ensure_3d(cort_cube_data, "cortical_qsm_cube"),
        sub=ensure_3d(sub_data, "subcortical_qsm"),
        cortical_seg=ensure_3d(cortical_seg, "cortical_seg"),
        subcortical_seg=ensure_3d(sub_lbl_data.astype(np.int32, copy=False), "subcortical_label"),
        cube_mask=ensure_3d(cube_mask.astype(np.uint8, copy=False), "cube_mask"),
        cort_ok=(cort_roi_ok if display_mode == "ROI only" else cort_cube_ok),
        cort_roi_ok=cort_roi_ok,
        cort_cube_ok=cort_cube_ok,
        sub_ok=sub_ok,
        native_axcodes=native_axcodes,
        reoriented_affine=reoriented_affine,
        zooms=zooms,
        generated_paths=generated_paths,
    )


def _warm_onedrive_files(paths: List[str], timeout: float = 8.0):
    """
    Touch each file path to trigger OneDrive/cloud stub download.
    On Windows with OneDrive Files On-Demand, simply calling os.path.getsize()
    or opening the file causes the OS to download it.
    On macOS with iCloud Drive / OneDrive, similar stubs exist.
    Runs in the calling thread; safe to call from a worker thread.
    Silently ignores errors.
    """
    import time
    deadline = time.monotonic() + timeout
    for p in paths:
        if time.monotonic() > deadline:
            break
        if not p or not os.path.exists(p):
            continue
        try:
            # Reading 512 bytes is enough to trigger cloud download
            with open(p, 'rb') as f:
                f.read(512)
        except OSError:
            try:
                os.path.getsize(p)
            except OSError:
                pass


class LoadWorker(QObject):
    progressed = Signal(int, str)
    finished = Signal(dict)
    failed = Signal(str)

    def __init__(self, *, case: CasePaths, canonical_key: str, save_generated: bool,
                 subcortical_margin: int, cortical_dilation_iter: int,
                 display_mode: str, file_names: Dict[str, str]):
        super().__init__()
        self.case = case
        self.canonical_key = canonical_key
        self.save_generated = save_generated
        self.subcortical_margin = subcortical_margin
        self.cortical_dilation_iter = cortical_dilation_iter
        self.display_mode = display_mode
        self.file_names = dict(file_names)

    def run(self):
        try:
            data = load_case_data(
                case=self.case,
                canonical_key=self.canonical_key,
                save_generated=self.save_generated,
                subcortical_margin=self.subcortical_margin,
                cortical_dilation_iter=self.cortical_dilation_iter,
                display_mode=self.display_mode,
                progress_cb=lambda pct, msg: self.progressed.emit(int(pct), str(msg)),
                file_names=self.file_names,
            )
            self.finished.emit(data)
        except Exception as exc:
            self.failed.emit(str(exc))
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
    def __init__(self, on_wheel: Callable[[object], None], parent: QObject = None):
        super().__init__(parent)
        self._on_wheel = on_wheel
    def eventFilter(self, obj, event) -> bool:
        if event.type() == QEvent.Wheel:
            self._on_wheel(event)
            return True
        return False
# ─────────────────────────────────────────────────────────────────────────────
# Contrast dialog
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
_DIR_CSS  = (f"font-size:{_DIR_PT}pt; color:{_C_DIM}; background:{_C_DIR_BAR};"
             " padding:2px 8px; letter-spacing:1px;")
_SYNC_CSS = (f"QCheckBox {{ color:#aabbcc; font-size:{_SMALL_PT}pt; }}"
             f"QCheckBox::indicator {{ width:13px; height:13px; }}")

# Shared stylesheet for ALL combos that live in a canvas title bar.
# Overrides the global 11pt/5px-padding rule so both zoom and mode combos
# are exactly the same visual size regardless of parent stylesheet cascade.
_HDR_COMBO_CSS = (
    f"QComboBox {{"
    f"  background:{_C_PANEL}; color:{_C_TEXT};"
    f"  border:1px solid {_C_BORDER}; border-radius:4px;"
    f"  font-size:{_BASE_PT}pt; padding:{'1px 6px' if _IS_WINDOWS else '1px 6px'};"
    + (f"  min-height:{_HDR_COMBO_MIN_H}px; max-height:{_HDR_COMBO_MAX_H}px;" if not _IS_WINDOWS else "")
    + f"}}"
    f"QComboBox:hover {{ border-color:{_C_ACCENT}; }}"
)
class ImageCanvas(QWidget):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._zoom_mode = "fit"   # "fit" or float multiplier relative to fit
        self._updating_zoom_ui = False
        # title bar
        # ── title bar ─────────────────────────────────────────────────────
        # All interactive widgets in the header share one height so they
        # sit on the same baseline. We give the row enough room for 11pt text
        # with comfortable padding, then let Qt centre items vertically.
        _COMBO_MIN = _HDR_COMBO_MIN_H
        _COMBO_MAX = _HDR_COMBO_MAX_H

        hdr = QWidget()
        hdr.setStyleSheet(f"background:{_C_HEADER};")
        hdr.setFixedHeight(_HDR_H)
        hl = QHBoxLayout(hdr)
        if _IS_WINDOWS:
            hl.setContentsMargins(4, 0, 4, 0) 
        else:
            hl.setContentsMargins(8, _HDR_MARGIN_V, 8, _HDR_MARGIN_V)
            hl.setAlignment(Qt.AlignVCenter)
        hl.setSpacing(6)

        self.title_label = QLabel(title)
        self.title_label.setStyleSheet(f"font-weight:600; font-size:{_BASE_PT}pt; color:{_C_TEXT};")
        if not _IS_WINDOWS:
            self.title_label.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)

        self.title_right_layout = QHBoxLayout()
        self.title_right_layout.setContentsMargins(0, 0, 0, 0)
        self.title_right_layout.setSpacing(3 if _IS_WINDOWS else 0)
        if not _IS_WINDOWS:
            self.title_right_layout.setAlignment(Qt.AlignVCenter)

        zoom_lbl = QLabel("Zoom")
        zoom_lbl.setStyleSheet(f"color:{_C_DIM}; font-size:{_SMALL_PT}pt;")
        if not _IS_WINDOWS:
            zoom_lbl.setAlignment(Qt.AlignVCenter)

        self.zoom_combo = StyledComboBox()
        self.zoom_combo.setEditable(True)
        self.zoom_combo.setInsertPolicy(QComboBox.NoInsert)
        self.zoom_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        if _IS_WINDOWS:
            self.zoom_combo.setMinimumWidth(70)
            self.zoom_combo.setMaximumWidth(168)
        else:
            self.zoom_combo.setMinimumWidth(90)
            self.zoom_combo.setMaximumWidth(145)
            self.zoom_combo.setMinimumHeight(_COMBO_MIN)
            self.zoom_combo.setMaximumHeight(_COMBO_MAX)
            self.zoom_combo.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed) 
        self.zoom_combo.setStyleSheet(_HDR_COMBO_CSS)
        self.zoom_combo.addItem("Autofit")
        for pct in ZOOM_PRESETS:
            self.zoom_combo.addItem(f"{pct}%")
        self.zoom_combo.setCurrentText("Autofit")
        self.zoom_combo.lineEdit().editingFinished.connect(self._on_zoom_combo_edited)
        self.zoom_combo.currentTextChanged.connect(self._on_zoom_combo_changed)

        self.sync_cb = QCheckBox("\U0001f517 Sync")
        self.sync_cb.setChecked(True)
        self.sync_cb.setStyleSheet(_SYNC_CSS)
        self.sync_cb.setToolTip("Uncheck to scroll this view independently")

        hl.addWidget(self.title_label)
        hl.addLayout(self.title_right_layout)
        hl.addStretch(1)
        hl.addWidget(zoom_lbl)
        hl.addWidget(self.zoom_combo)
        hl.addWidget(self.sync_cb)

        # Store combo height bounds for set_title_right_widget
        self._hdr_combo_min = _COMBO_MIN
        self._hdr_combo_max = _COMBO_MAX
        self._dir_bar = QWidget()
        self._dir_bar.setStyleSheet(f"background:{_C_DIR_BAR};")
        self._dir_bar.setFixedHeight(0)
        self._dir_bar.hide()
        # napari viewer
        self.viewer_model = ViewerModel(title=title)
        self.qt_viewer    = QtViewer(self.viewer_model)
        self.qt_viewer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._canvas_native: Optional[QWidget] = self._find_native()
        self._wheel_filter: Optional[WheelScrollFilter] = None
        self._last_fit_signature = None
        self._dims_patched = False
        # Custom slice-info label (replaces napari's clipping spinbox)
        self._slice_info_lbl: Optional[QLabel] = None

        # Patch dims bar widgets after first layer insertion
        self.viewer_model.layers.events.inserted.connect(
            lambda _e: QTimer.singleShot(80, self._patch_napari_dims_bar)
        )

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(hdr)
        lay.addWidget(self.qt_viewer, 1)

        self.viewer_model.layers.selection.events.active.connect(
            self._on_layer_active)
        # Connect dims current_step to update our slice label
        self.viewer_model.dims.events.current_step.connect(
            self._on_dims_step_changed)
        self._hover_overlay_parent = self._canvas_native if self._canvas_native is not None else self.qt_viewer
        self._hover_info_label = QLabel(self._hover_overlay_parent)
        self._hover_info_label.setStyleSheet(
            f"background: rgba(10, 12, 16, 180); color:{_C_TEXT}; "
            f"border:1px solid {_C_BORDER}; border-radius:4px; "
            f"padding:{2 if _IS_WINDOWS else 4}px {5 if _IS_WINDOWS else 8}px; "
            f"font-size:{_SMALL_PT}pt;"
        )
        self._hover_info_label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._hover_info_label.hide()

        self._intensity_info_label = QLabel(self._hover_overlay_parent)
        self._intensity_info_label.setStyleSheet(
            f"background: rgba(10, 12, 16, 180); color:{_C_TEXT}; "
            f"border:1px solid {_C_BORDER}; border-radius:4px; "
            f"padding:{2 if _IS_WINDOWS else 4}px {5 if _IS_WINDOWS else 8}px; "
            f"font-size:{_SMALL_PT}pt;"
        )
        self._intensity_info_label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._intensity_info_label.hide()
        self._intensity_layer = None

        self._dir_overlay_visible = True
        self._dir_overlay_labels = {}
        dir_style = (
            f"background: transparent; color: #f2b000; border: none; "
            f"font-size:{_DIR_PT}pt; font-weight:600; padding:0px;"
        )
        for key in ("left", "right", "top", "bottom"):
            lbl = QLabel(self._hover_overlay_parent)
            lbl.setStyleSheet(dir_style)
            lbl.setAttribute(Qt.WA_TransparentForMouseEvents, True)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.hide()
            self._dir_overlay_labels[key] = lbl

        self._hover_sources = []
        self._hover_event_source = self._canvas_native or self.qt_viewer
        if self._hover_event_source is not None:
            try:
                self._hover_event_source.setMouseTracking(True)
            except Exception:
                pass
            self._hover_event_source.installEventFilter(self)
        self.qt_viewer.installEventFilter(self)
        self._reposition_hover_info_label()
        self._update_direction_overlay_visibility()
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

    def _on_dims_step_changed(self, event=None):
        """Update napari dims bar and forcefully defend spinbox width."""
        self._patch_napari_dims_bar()

        from qtpy.QtWidgets import QWidget, QAbstractSpinBox
        
        # 1. 终极防御机制：无情碾压 Napari 内部的动态尺寸计算
        # 每次切片变动时，强行重置 SpinBox 的最小和最大宽度，彻底封死它缩水的可能！
        for child in self.qt_viewer.findChildren(QWidget):
            if isinstance(child, QAbstractSpinBox) or 'spinbox' in type(child).__name__.lower():
                child.setMinimumWidth(50 if _IS_WINDOWS else 60)
                child.setMaximumWidth(80)

        # 2. 更新右侧总数 Label，使其配合输入框显示为 "/ 47" 的优雅格式
        if getattr(self, '_slice_info_lbl', None) is not None:
            try:
                dims = self.viewer_model.dims
                scroll_ax = getattr(self, '_scroll_axis_hint', None)
                if scroll_ax is None:
                    for ax in range(len(dims.nsteps)):
                        if dims.nsteps[ax] > 1:
                            scroll_ax = ax
                            break
                if scroll_ax is not None and scroll_ax < len(dims.current_step):
                    total = int(dims.nsteps[scroll_ax])
                    self._slice_info_lbl.setText(f"/ {total}")
            except Exception:
                pass
        self.refresh_hover_label()

    def _patch_napari_dims_bar(self):
        """
        一次性基础样式修补。
        """
        if getattr(self, '_dims_patched', False):
            return

        from qtpy.QtWidgets import QWidget, QLabel, QPushButton, QAbstractSpinBox, QAbstractSlider, QSizePolicy
        from qtpy.QtCore import Qt
        from qtpy.QtGui import QIcon

        found_slider = False

        for child in self.qt_viewer.findChildren(QWidget):
            try:
                cn = type(child).__name__.lower()
                tt = (child.toolTip() or '').lower()

                # 隐藏废弃的坐标轴文本 ("edit to change")
                if 'edit to change' in tt or 'dimension name' in tt:
                    child.hide()
                    child.setFixedSize(0, 0)
                    continue

                # 完美修复播放按钮
                if isinstance(child, QPushButton):
                    child.setIcon(QIcon())
                    child.setText('▶')
                    child.setStyleSheet(
                        f"QPushButton {{ font-size:{_BASE_PT+2}pt; padding:0px; margin:0px; "
                        f"background:{_C_HEADER}; color:{_C_TEXT}; "
                        f"border:1px solid {_C_BORDER}; border-radius:3px; }}"
                        f"QPushButton:hover {{ background:{_C_ACCENT}; color:white; }}"
                    )
                    child.setFixedSize(28 if _IS_WINDOWS else 32, 22 if _IS_WINDOWS else 24)
                    continue

                # 设置 SpinBox (数字输入框) 的暗黑UI样式
                if isinstance(child, QAbstractSpinBox) or 'spinbox' in cn:
                    child.setStyleSheet(
                        f"background:{_C_PANEL}; color:{_C_TEXT}; "
                        f"border:1px solid {_C_BORDER}; border-radius:3px;"
                    )
                    continue

                # 强化滑动条的拉伸属性，逼迫它去占满两端多余的空白
                if isinstance(child, QAbstractSlider) and 'scrollbar' not in cn:
                    sp = child.sizePolicy()
                    sp.setHorizontalPolicy(QSizePolicy.Expanding)
                    child.setSizePolicy(sp)
                    found_slider = True
                    continue

                # 抓取最右侧的 Label，准备在上面的函数中改造为 "/ 总数" 格式
                if isinstance(child, QLabel) and child.isVisible():
                    txt = child.text().strip()
                    if txt.isdigit() or '/' in txt:
                        child.setMinimumWidth(45)
                        child.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                        child.setStyleSheet(
                            f"color:{_C_DIM}; font-size:{_BASE_PT}pt; "
                            f"font-family:Consolas,Menlo,monospace; padding-right:4px;"
                        )
                        self._slice_info_lbl = child

            except Exception:
                pass

        if found_slider:
            self._dims_patched = True

    def set_hover_label_source(self, layer, label_map: Optional[Dict[int, str]],
                               visibility_getter: Optional[Callable[[], bool]] = None):
        sources = []
        if layer is not None and label_map:
            sources.append({
                'layer': layer,
                'label_map': dict(label_map or {}),
                'visibility_getter': visibility_getter,
            })
        self._hover_sources = sources
        self.clear_hover_info()

    def set_hover_label_sources(self, sources):
        normalized = []
        for src in (sources or []):
            if not isinstance(src, dict):
                continue
            layer = src.get('layer')
            label_map = dict(src.get('label_map') or {})
            if layer is None or not label_map:
                continue
            normalized.append({
                'layer': layer,
                'label_map': label_map,
                'visibility_getter': src.get('visibility_getter'),
            })
        self._hover_sources = normalized
        self.clear_hover_info()

    def clear_hover_info(self):
        if getattr(self, '_hover_info_label', None) is not None:
            self._hover_info_label.clear()
            self._hover_info_label.hide()
        if getattr(self, '_intensity_info_label', None) is not None:
            self._intensity_info_label.hide()

    def set_intensity_layer(self, layer):
        self._intensity_layer = layer

    def _sample_intensity_value(self):
        layer = getattr(self, '_intensity_layer', None)
        if layer is None:
            return None
        world_pos = getattr(getattr(self.viewer_model, 'cursor', None), 'position', None)
        if world_pos is None:
            return None
        data = getattr(layer, 'data', None)
        if data is None:
            return None
        data_pos = None
        if hasattr(layer, 'world_to_data'):
            try:
                data_pos = layer.world_to_data(world_pos)
            except Exception:
                data_pos = None
        if data_pos is None:
            data_pos = world_pos
        arr = np.asarray(data_pos, dtype=float).reshape(-1)
        ndim = int(getattr(data, 'ndim', 0) or 0)
        if ndim <= 0:
            return None
        if arr.size != ndim:
            current = list(getattr(self.viewer_model.dims, 'current_step', ()) or ())
            if len(current) < ndim:
                current.extend([0] * (ndim - len(current)))
            for i in range(min(arr.size, ndim)):
                current[i] = arr[i]
            arr = np.asarray(current[:ndim], dtype=float)
        shape = tuple(int(v) for v in getattr(data, 'shape', ()))
        if len(shape) != ndim:
            return None
        idx = []
        for axis, val in enumerate(arr):
            if not np.isfinite(val):
                return None
            vox = int(np.floor(float(val) + 1e-6))
            if vox < 0 or vox >= shape[axis]:
                return None
            idx.append(vox)
        try:
            return float(data[tuple(idx)])
        except Exception:
            return None

    def _refresh_intensity_label(self):
        int_lbl = getattr(self, '_intensity_info_label', None)
        if int_lbl is None:
            return
        val = self._sample_intensity_value()
        if val is None:
            int_lbl.hide()
        else:
            text = f"{val:.4f}"
            if int_lbl.text() != text:
                int_lbl.setText(text)
                int_lbl.adjustSize()
            int_lbl.show()

    def set_direction_overlay_visible(self, visible: bool):
        self._dir_overlay_visible = bool(visible)
        self._update_direction_overlay_visibility()

    def _update_direction_overlay_visibility(self):
        labels = getattr(self, '_dir_overlay_labels', None) or {}
        for lbl in labels.values():
            if self._dir_overlay_visible and bool(lbl.text()):
                lbl.show()
                lbl.raise_()
            else:
                lbl.hide()
        self._reposition_direction_overlay_labels()

    def _reposition_hover_info_label(self):
        seg_lbl = getattr(self, '_hover_info_label', None)
        int_lbl = getattr(self, '_intensity_info_label', None)
        if seg_lbl is None and int_lbl is None:
            return
        margin = 6 if _IS_WINDOWS else 10
        gap = 4
        parent = getattr(self, '_hover_overlay_parent', None) or self.qt_viewer
        pw = max(0, parent.width())
        ph = max(0, parent.height())
        # Position seg label at bottom-left
        if seg_lbl is not None:
            try:
                seg_lbl.adjustSize()
            except Exception:
                pass
            x = margin
            y = max(margin, ph - seg_lbl.height() - margin)
            seg_lbl.move(x, y)
            seg_lbl.raise_()
        # Position intensity label: above seg label when visible, else at bottom-left
        if int_lbl is not None:
            try:
                int_lbl.adjustSize()
            except Exception:
                pass
            x = margin
            if seg_lbl is not None and seg_lbl.isVisible():
                y = max(margin, seg_lbl.y() - int_lbl.height() - gap)
            else:
                y = max(margin, ph - int_lbl.height() - margin)
            int_lbl.move(x, y)
            int_lbl.raise_()
        self._reposition_direction_overlay_labels()

    def _reposition_direction_overlay_labels(self):
        labels = getattr(self, '_dir_overlay_labels', None) or {}
        if not labels:
            return
        parent = getattr(self, '_hover_overlay_parent', None) or self.qt_viewer
        pw = max(0, parent.width())
        ph = max(0, parent.height())
        if pw <= 0 or ph <= 0:
            return
        edge_margin = 3 if _IS_WINDOWS else 6
        center_x = pw // 2
        center_y = ph // 2
        positions = {
            'left': (edge_margin, center_y),
            'right': (pw - edge_margin, center_y),
            'top': (center_x, edge_margin),
            'bottom': (center_x, ph - edge_margin),
        }
        for key, lbl in labels.items():
            try:
                lbl.adjustSize()
            except Exception:
                pass
            x, y = positions[key]
            if key == 'left':
                px = x
                py = y - lbl.height() // 2
            elif key == 'right':
                px = x - lbl.width()
                py = y - lbl.height() // 2
            elif key == 'top':
                px = x - lbl.width() // 2
                py = y
            else:
                px = x - lbl.width() // 2
                py = y - lbl.height()
            lbl.move(max(0, px), max(0, py))
            lbl.raise_()

    def _hover_source_enabled(self, src) -> bool:
        layer = src.get('layer')
        label_map = src.get('label_map') or {}
        if layer is None or not label_map:
            return False
        visibility_getter = src.get('visibility_getter')
        if visibility_getter is not None:
            try:
                if not bool(visibility_getter()):
                    return False
            except Exception:
                return False
        try:
            if not bool(getattr(layer, 'visible', False)):
                return False
        except Exception:
            return False
        return True

    def _sample_hover_label_value(self):
        sources = getattr(self, '_hover_sources', None) or []
        if not sources:
            return None
        world_pos = getattr(getattr(self.viewer_model, 'cursor', None), 'position', None)
        if world_pos is None:
            return None
        for source in sources:
            if not self._hover_source_enabled(source):
                continue
            layer = source.get('layer')
            label_map = source.get('label_map') or {}
            data = getattr(layer, 'data', None)
            if data is None:
                continue
            data_pos = None
            if hasattr(layer, 'world_to_data'):
                try:
                    data_pos = layer.world_to_data(world_pos)
                except Exception:
                    data_pos = None
            if data_pos is None:
                data_pos = world_pos
            arr = np.asarray(data_pos, dtype=float).reshape(-1)
            ndim = int(getattr(data, 'ndim', 0) or 0)
            if ndim <= 0:
                continue
            if arr.size != ndim:
                current = list(getattr(self.viewer_model.dims, 'current_step', ()) or ())
                if len(current) < ndim:
                    current.extend([0] * (ndim - len(current)))
                for i in range(min(arr.size, ndim)):
                    current[i] = arr[i]
                arr = np.asarray(current[:ndim], dtype=float)
            idx = []
            shape = tuple(int(v) for v in getattr(data, 'shape', ()))
            if len(shape) != ndim:
                continue
            valid = True
            for axis, val in enumerate(arr):
                if not np.isfinite(val):
                    valid = False
                    break
                vox = int(np.floor(float(val) + 1e-6))
                if vox < 0 or vox >= shape[axis]:
                    valid = False
                    break
                idx.append(vox)
            if not valid:
                continue
            try:
                label_val = int(data[tuple(idx)])
            except Exception:
                continue
            if label_val <= 0:
                continue
            label_name = label_map.get(label_val)
            if not label_name:
                continue
            return label_val, label_name
        return None

    def refresh_hover_label(self):
        result = self._sample_hover_label_value()
        seg_lbl = getattr(self, '_hover_info_label', None)
        if seg_lbl is not None:
            if result is None:
                seg_lbl.clear()
                seg_lbl.hide()
            else:
                label_val, label_name = result
                text = f"{label_val} · {label_name}"
                if seg_lbl.text() != text:
                    seg_lbl.setText(text)
                    seg_lbl.adjustSize()
                seg_lbl.show()
                seg_lbl.raise_()
        self._refresh_intensity_label()
        self._reposition_hover_info_label()

    def eventFilter(self, obj, event):
        hover_parent = getattr(self, '_hover_overlay_parent', None)
        if obj in {getattr(self, '_hover_event_source', None), self.qt_viewer, hover_parent}:
            et = event.type()
            if et == QEvent.MouseMove:
                self.refresh_hover_label()
            elif et in (QEvent.Leave, QEvent.HoverLeave):
                self.clear_hover_info()
            elif et == QEvent.Resize:
                self._reposition_hover_info_label()
        return super().eventFilter(obj, event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._reposition_hover_info_label()

    def set_title_right_widget(self, widget: Optional[QWidget]):
        """Insert a widget (e.g. cortical mode combo) into the title bar.
        Applies the same height bounds and stylesheet as the zoom combo."""
        while self.title_right_layout.count():
            item = self.title_right_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
        if widget is not None:
            if isinstance(widget, QComboBox):
                if not _IS_WINDOWS:
                    widget.setMinimumHeight(self._hdr_combo_min)
                    widget.setMaximumHeight(self._hdr_combo_max)
                widget.setStyleSheet(_HDR_COMBO_CSS)
            self.title_right_layout.addWidget(widget)

    @property
    def sync_enabled(self): return self.sync_cb.isChecked()
    @staticmethod
    def _parse_zoom_text(text: str):
        s = (text or "").strip()
        if not s:
            return None
        s_low = s.lower()
        if s_low in {"fit", "fit window", "fitwindow", "window", "auto", "autofit"}:
            return "fit"
        try:
            if s.endswith('%'):
                value = float(s[:-1].strip())
            else:
                value = float(s)
                if value <= 10:
                    value *= 100.0
            if value <= 0:
                return None
            return value / 100.0
        except Exception:
            return None
    @staticmethod
    def _format_zoom_mode(mode) -> str:
        if mode == "fit":
            return "Autofit"
        try:
            pct = float(mode) * 100.0
            if abs(pct - round(pct)) < 1e-6:
                return f"{int(round(pct))}%"
            return f"{pct:.1f}%"
        except Exception:
            return "Autofit"
    def _set_zoom_combo_text(self, text: str):
        self._updating_zoom_ui = True
        self.zoom_combo.setCurrentText(text)
        self._updating_zoom_ui = False
    def _on_zoom_combo_changed(self, text: str):
        if self._updating_zoom_ui:
            return
        parsed = self._parse_zoom_text(text)
        if parsed is None:
            return
        self._zoom_mode = parsed
        self._set_zoom_combo_text(self._format_zoom_mode(parsed))
        self._emit_zoom_changed()
    def _on_zoom_combo_edited(self):
        text = self.zoom_combo.currentText()
        parsed = self._parse_zoom_text(text)
        if parsed is None:
            self._set_zoom_combo_text(self._format_zoom_mode(self._zoom_mode))
            return
        self._zoom_mode = parsed
        self._set_zoom_combo_text(self._format_zoom_mode(parsed))
        self._emit_zoom_changed()
    def _emit_zoom_changed(self):
        parent = self.parent()
        while parent is not None and not hasattr(parent, '_on_canvas_zoom_changed'):
            parent = parent.parent()
        if parent is not None:
            parent._on_canvas_zoom_changed(self)
    def set_zoom_mode(self, mode, apply: bool = False):
        parsed = mode if mode == "fit" else self._parse_zoom_text(str(mode))
        if parsed is None:
            parsed = "fit"
        self._zoom_mode = parsed
        self._set_zoom_combo_text(self._format_zoom_mode(parsed))
        if apply:
            self._emit_zoom_changed()
    def step_zoom_by(self, step: int, apply: bool = False):
        if self._zoom_mode == "fit":
            idx = ZOOM_PRESETS.index(100)
        else:
            curr_pct = float(self._zoom_mode) * 100.0
            idx = min(range(len(ZOOM_PRESETS)), key=lambda i: abs(ZOOM_PRESETS[i] - curr_pct))
        idx = max(0, min(idx + step, len(ZOOM_PRESETS) - 1))
        self.set_zoom_mode(ZOOM_PRESETS[idx] / 100.0, apply=apply)
    def set_direction_labels(self, left, right, top, bottom):
        mapping = {
            'left': str(left or '?'),
            'right': str(right or '?'),
            'top': str(top or '?'),
            'bottom': str(bottom or '?'),
        }
        for key, text in mapping.items():
            lbl = self._dir_overlay_labels.get(key)
            if lbl is not None:
                lbl.setText(text)
                try:
                    lbl.adjustSize()
                except Exception:
                    pass
        self._update_direction_overlay_visibility()
    def install_wheel_filter(self, on_wheel):
        target = self._canvas_native or self.qt_viewer
        self._wheel_filter = WheelScrollFilter(on_wheel, parent=self)
        target.installEventFilter(self._wheel_filter)
    @property
    def viewer(self): return self.viewer_model
    def clear_layers(self):
        self.viewer.layers.clear()
        self.clear_hover_info()
        self._hover_sources = []
        self._intensity_layer = None

    def add_image(self, *a, **kw): return self.viewer.add_image(*a, **kw)
    def add_labels(self, *a, **kw): return self.viewer.add_labels(*a, **kw)
    def lock_scroll_mode(self):
        try: self.viewer_model.camera.mouse_zoom = False
        except AttributeError: pass
    def set_view(self, dims_order, scroll_axis, data_shape, flip_v, flip_h):
        self._scroll_axis_hint = scroll_axis   # used by _on_dims_step_changed
        self.viewer.dims.ndisplay = 2
        self.viewer.dims.order    = dims_order
        set_slice(self.viewer, data_shape[scroll_axis] // 2, scroll_axis)
        try:
            self.viewer.camera.flip = (flip_v, flip_h)
        except Exception:
            try: self.viewer.camera.flip = (flip_v, flip_h, False)
            except Exception: pass
        # Refresh slice label immediately after orientation change
        self._on_dims_step_changed()
        self.clear_hover_info()
    def fit_to_shape(self, data_shape, dims_order, zooms=None, force: bool = False):
        """Fit camera to ~90 % of canvas, then apply current zoom mode."""
        try:
            nat = self._canvas_native
            vw = max(1, nat.width()) if nat else 400
            vh = max(1, nat.height()) if nat else 400
            signature = (tuple(data_shape), tuple(dims_order), tuple(zooms) if zooms is not None else None, vw, vh, self._zoom_mode)
            if (not force) and signature == self._last_fit_signature:
                return
            self._last_fit_signature = signature
            ax_row = dims_order[-2]; ax_col = dims_order[-1]
            if zooms is not None:
                phys_row = data_shape[ax_row] * zooms[ax_row]
                phys_col = data_shape[ax_col] * zooms[ax_col]
            else:
                phys_row = float(data_shape[ax_row])
                phys_col = float(data_shape[ax_col])
            self.viewer.camera.center = (phys_row / 2.0, phys_col / 2.0)
            fit_zoom = 0.90 * min(vw / max(1e-6, phys_col), vh / max(1e-6, phys_row))
            self.viewer.camera.zoom = fit_zoom if self._zoom_mode == "fit" else fit_zoom * float(self._zoom_mode)
            self._reposition_hover_info_label()
        except Exception:
            self.viewer.reset_view()
# ─────────────────────────────────────────────────────────────────────────────
# QC Grading Criteria floating overlay
# ─────────────────────────────────────────────────────────────────────────────
class _QCRulesOverlay(QFrame):
    """Floating panel that shows QC grading criteria, triggered by hovering
    the entry label in the info panel.  It is a child of the central widget
    and is positioned in the centre of the window."""

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.setInterval(280)
        self._hide_timer.timeout.connect(self.hide)

        self.setObjectName("_QCRulesOverlay")
        self.setStyleSheet(
            "#_QCRulesOverlay {"
            f"  background: {_C_PANEL};"
            f"  border: 1px solid {_C_ACCENT}66;"
            "  border-radius: 8px;"
            "}"
        )
        pad = 4 if _IS_WINDOWS else 10
        lay = QVBoxLayout(self)
        lay.setContentsMargins(pad, pad, pad, pad)
        lay.setSpacing(3 if _IS_WINDOWS else 6)

        # Title
        title = QLabel("QC Grading Criteria")
        title.setStyleSheet(
            f"font-size: {_BASE_PT + (0 if _IS_WINDOWS else 2)}pt; "
            f"font-weight: bold; color: {_C_TEXT}; "
            f"padding: {2 if _IS_WINDOWS else 4}px {4 if _IS_WINDOWS else 6}px; "
            "border: none; background: transparent;"
        )
        lay.addWidget(title)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color: {_C_BORDER};")
        lay.addWidget(sep)

        browser = QTextEdit()
        browser.setReadOnly(True)
        browser.setHtml(self._build_html())
        _content_pt = _BASE_PT + 1 if _IS_WINDOWS else _SMALL_PT + 2
        browser.setStyleSheet(
            f"QTextEdit {{ background: transparent; border: none; "
            f"color: {_C_TEXT}; font-size: {_content_pt}pt; }}"
            "QScrollBar:vertical { width: 8px; }"
        )
        browser.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        browser.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        lay.addWidget(browser)

        self.hide()

    @staticmethod
    def _build_html() -> str:
        bg_hdr = "#1e2128"
        c_acc  = _C_ACCENT
        c_brd  = _C_BORDER
        c_txt  = _C_TEXT
        c_dim  = _C_DIM
        fpt      = _BASE_PT + 1 if _IS_WINDOWS else _SMALL_PT + 3
        cell_pad = "5px 8px" if _IS_WINDOWS else "6px 10px"
        col_gn   = 52 if _IS_WINDOWS else 60
        col_gl   = 155 if _IS_WINDOWS else 168
        return f"""
<style type="text/css">
  body  {{ margin:0; color:{c_txt}; font-size:{fpt}pt; }}
  table {{ border-collapse:collapse; width:100%; }}
  th    {{ background:{bg_hdr}; color:{c_acc}; font-weight:bold;
           border:1px solid {c_brd}; padding:{cell_pad}; text-align:center; }}
  td    {{ border:1px solid {c_brd}; padding:{cell_pad}; vertical-align:top; color:{c_txt}; }}
  td.gn {{ text-align:center; font-weight:bold; color:{c_acc}; font-size:{fpt+2}pt;
           white-space:nowrap; }}
  td.gl {{ color:{c_dim}; font-style:italic; }}
  ul    {{ margin:2px 0 0 0; padding-left:{16 if _IS_WINDOWS else 18}px; }}
  li    {{ margin:{2 if _IS_WINDOWS else 3}px 0; }}
</style>
<table>
<tr>
  <th width="{col_gn}">Grade</th>
  <th width="{col_gl}">&nbsp;</th>
  <th>Cortical</th>
  <th>Subcortical</th>
</tr>
<tr>
  <td class="gn">0</td>
  <td class="gl">no motion artifact</td>
  <td>No cortical ROI affected</td>
  <td><b>No artifact affecting subcortical ROIs</b>
    <ul>
      <li>No target subcortical structures are affected.</li>
      <li>Boundaries are preserved.</li>
      <li>Quantitative ROI extraction is trustworthy.</li>
    </ul>
  </td>
</tr>
<tr>
  <td class="gn">1</td>
  <td class="gl">mild motion artifact</td>
  <td><ul>
    <li>Artifact is focal or <b>mild</b></li>
    <li>Cortical ROIs remain trustworthy</li>
    <li><b>Minor edge blur</b>/streaking only</li>
  </ul></td>
  <td><b>Mild artifact affecting only limited/small ROIs</b>
    <ul>
      <li>Major high-priority structures remain unaffected and trustworthy,
        such as <b>caudate, putamen, globus pallidus, thalamus, dentate</b>.</li>
      <li>Artifact affects only <b>small and/or artifact-prone structures</b>,
        such as <b>accumbens, substantia nigra, red nucleus, pulvinar</b>,
        or affects only a small portion of a target ROI.</li>
      <li>Overall quantitative extraction for the main subcortical analysis
        remains trustworthy.</li>
    </ul>
  </td>
</tr>
<tr>
  <td class="gn">2</td>
  <td class="gl">motion artifact affecting image contrast of internal structures</td>
  <td><ul>
    <li>Artifact affects <b>one or more meaningful cortical regions</b></li>
    <li>Cortical ribbon or local ROI interpretation becomes questionable</li>
    <li><b>Values may be biased</b></li>
  </ul></td>
  <td><b>Artifact affecting important ROIs, with possible bias</b>
    <ul>
      <li><b>Multiple ROIs</b>, or at least <b>one major/high-priority ROI</b>,
        are affected.</li>
      <li>Boundaries may still be partly visible, but measurements may be
        biased or less reliable.</li>
      <li>Deep anatomy is still interpretable overall, but quantitative ROI
        extraction should be used with caution.</li>
    </ul>
  </td>
</tr>
<tr>
  <td class="gn">3</td>
  <td class="gl">severe motion artifact affecting both superficial and deep structures</td>
  <td><ul>
    <li><b>Widespread</b> cortical ROIs are not trustworthy</li>
    <li>Ribbon anatomy is <b>severely</b> degraded</li>
    <li><b>Quantitative cortical analysis is invalid</b></li>
  </ul></td>
  <td><b>Severe artifact, subcortical analysis unreliable</b>
    <ul>
      <li>Multiple ROIs are not trustworthy.</li>
      <li>Boundaries are lost, deep anatomy is severely distorted, or several
        key ROIs cannot be confidently identified.</li>
      <li>Quantitative ROI extraction is likely scientifically invalid.</li>
    </ul>
  </td>
</tr>
</table>"""

    def show_centered(self):
        """Compute centre of parent, resize, and show."""
        self._hide_timer.stop()
        parent = self.parent()
        if parent is None:
            return
        pw, ph = parent.width(), parent.height()
        if _IS_WINDOWS:
            w = min(max(int(pw * 0.76), 520), 1040)
            h = min(max(int(ph * 0.68), 360), 680)
        else:
            w = min(max(int(pw * 0.80), 560), 1100)
            h = min(max(int(ph * 0.72), 400), 740)
        self.resize(w, h)
        self.move(max(0, (pw - w) // 2), max(0, (ph - h) // 2))
        self.raise_()
        self.show()

    def schedule_hide(self):
        self._hide_timer.start()

    def cancel_hide(self):
        self._hide_timer.stop()

    # Keep visible while cursor is inside the popup itself
    def enterEvent(self, event):
        self._hide_timer.stop()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.hide()
        super().leaveEvent(event)

# ─────────────────────────────────────────────────────────────────────────────
# Main window
# ─────────────────────────────────────────────────────────────────────────────
class ReviewerMainWindow(QMainWindow):
    def __init__(self, cases: List[CasePaths], output_csv: str, file_names: Dict[str, str],
                 save_generated_qsm: bool = SAVE_GENERATED_MASKED_QSM,
                 show_seg_in_derived_views: bool = False):
        super().__init__()
        self.setWindowTitle("QSM QC Reviewer")
        self.setWindowIcon(_make_app_icon())

        # Window sizing: Windows uses fixed size matching stable version,
        # macOS uses adaptive screen-percentage sizing.
        if _IS_WINDOWS:
            screnn = QApplication.primaryScreen()
            if screnn is not None:
                w = min(int(screnn.size().width() * 0.85), 1600)
                h = min(int(screnn.size().height() * 0.78), 860)
            else:
                w, h = (1280, 760)
            self.resize(w, h)
        else:
            screen = QApplication.primaryScreen()
            if screen is not None:
                sg = screen.availableGeometry()
                w = min(int(sg.width()  * 0.85), 1600)
                h = min(int(sg.height() * 0.78), 860)
            else:
                w, h = (1280, 760)
            self.resize(w, h)

        self.setStyleSheet(_GLOBAL_CSS)
        self.cases         = cases
        self.output_csv    = output_csv
        self.file_names    = dict(file_names)
        self._initial_save_generated_qsm = bool(save_generated_qsm)
        self._show_seg_in_derived_views = bool(show_seg_in_derived_views)
        self.results       = read_existing_results(output_csv)
        self.case_index    = 0
        self.current_saved = True
        self._syncing      = False
        self._loading      = False
        self._load_thread: Optional[QThread] = None
        self._load_worker: Optional[LoadWorker] = None
        self._pending_load_index: Optional[int] = None
        self._data_cache: OrderedDict = OrderedDict()
        self._scan_thread: Optional[QThread] = None
        self._scan_worker: Optional[CaseScanWorker] = None
        self._scan_completed = False
        self._cache_limit = 3
        self._reinit_gen: int = 0          # incremented on every reinit; invalidates stale prefetch results
        self._current_config: Optional[AppConfig] = None   # set after first start_case_scan call
        self._wheel_pending_steps: Dict[int, Tuple[ImageCanvas, int]] = {}
        self._wheel_timer = QTimer(self)
        self._wheel_timer.setSingleShot(True)
        self._wheel_timer.setInterval(12)
        self._wheel_timer.timeout.connect(self._flush_wheel_scroll)
        # Orientation state
        cfg = ORIENTATIONS[DEFAULT_ORIENTATION]
        self._scroll_axis: int  = cfg["axis"]
        self._dims_order: tuple = cfg["order"]
        self._canonical_key     = DEFAULT_CANONICAL
        self._flip_v            = False
        self._flip_h            = False
        # Data
        self.raw_data = self.cortical_data = self.subcortical_data = None
        self.cortical_roi_data = self.cortical_cube_data = self.cube_mask_data = None
        self._reoriented_affine = None
        self._zooms: Optional[tuple] = None
        self._cortical_roi_source_ok = False
        self._cortical_cube_source_ok = False
        self._subcortical_source_ok = False
        self._cortical_hover_label_map, self._subcortical_hover_label_map = load_hover_label_maps()
        # Layers
        self.raw_layer = self.seg_cortical_layer = None
        self.seg_subcortical_layer = self.cortical_layer = self.subcortical_layer = None
        self.cortical_seg_layer = self.subcortical_seg_layer = None
        # Canvases
        self.raw_canvas         = ImageCanvas("Raw QSM + Segmentation")
        self.cortical_canvas    = ImageCanvas("Cortical QSM")
        self.subcortical_canvas = ImageCanvas("Subcortical QSM")
        self.cortical_mode_combo = _make_combo(
            ["ROI only", "All regions outside subcortical"],
            min_width=150, max_width=200, expanding=False)
        self.cortical_mode_combo.setCurrentText("All regions outside subcortical")
        self.cortical_mode_combo.currentTextChanged.connect(self._on_cortical_display_mode_changed)
        self.cortical_canvas.set_title_right_widget(self.cortical_mode_combo)
        # Contrast dialog
        self.contrast_dlg = ContrastDialog(self)
        self.contrast_dlg._apply_cb  = self.apply_contrast
        self.contrast_dlg._seg_op_cb = self._update_seg_opacity
        # Fit-all debounce timer (splitter drag)
        self._fit_timer = QTimer(self)
        self._fit_timer.setSingleShot(True)
        self._fit_timer.setInterval(120)
        self._fit_timer.timeout.connect(self._fit_all)
        self._build_menu()
        self._build_toolbar()
        self._build_ui()
        self._build_statusbar()
        self._bind_shortcuts()
        self._connect_slice_sync()
        self._install_wheel_filters()
        if self.cases:
            self.load_case(0)
        else:
            self._set_nav_enabled(False)
            QTimer.singleShot(0, lambda: self._set_status("Scanning case list…", _C_ACCENT))
    # ── helpers ───────────────────────────────────────────────────────────────
    def _all_canvases(self):
        return [self.raw_canvas, self.cortical_canvas, self.subcortical_canvas]
    def _canvas_data_pairs(self):
        return [(self.raw_canvas, self.raw_data),
                (self.cortical_canvas, self.cortical_data),
                (self.subcortical_canvas, self.subcortical_data)]

    def _cache_key_for(self, case: CasePaths, canonical_key: str):
        return (
            case.case_id,
            canonical_key,
            int(SUBCORTICAL_MARGIN),
            int(CORTICAL_DILATION_ITER),
            tuple(sorted(self.file_names.items())),
            bool(self._act_save_gen.isChecked()) if hasattr(self, '_act_save_gen') else False,
        )

    def _remember_cache(self, key, data: dict):
        self._data_cache[key] = data
        self._data_cache.move_to_end(key)
        while len(self._data_cache) > self._cache_limit:
            self._data_cache.popitem(last=False)

    def _cleanup_load_thread(self):
        thread = self._load_thread
        worker = self._load_worker
        self._load_worker = None
        self._load_thread = None
        if thread is not None:
            thread.quit()
            thread.wait(1000)
            thread.deleteLater()
        if worker is not None:
            worker.deleteLater()

    def _update_image_layer(self, layer_attr: str, canvas: ImageCanvas, data, *, name: str, scale, contrast_limits=None):
        layer = getattr(self, layer_attr, None)
        if layer is None:
            layer = canvas.add_image(data, name=name, colormap='gray', contrast_limits=contrast_limits or (DEFAULT_LOW, DEFAULT_HIGH), scale=scale)
            setattr(self, layer_attr, layer)
        else:
            layer.data = data
            layer.scale = scale
            if contrast_limits is not None:
                layer.contrast_limits = contrast_limits
            try:
                layer.visible = True
            except Exception:
                pass
        return layer

    def _update_label_layer(self, layer_attr: str, canvas: ImageCanvas, data, *, name: str, scale, opacity: float, visible: bool):
        layer = getattr(self, layer_attr, None)
        if layer is None:
            layer = canvas.add_labels(data, name=name, opacity=opacity, visible=visible, scale=scale)
            setattr(self, layer_attr, layer)
        else:
            layer.data = data
            layer.scale = scale
            layer.opacity = opacity
            layer.visible = visible
        return layer
    def _derived_cortical_hover_visible(self) -> bool:
        try:
            return bool(self._show_seg_in_derived_views and self.cort_seg_cb.isChecked() and self.cortical_seg_layer is not None and self.cortical_seg_layer.visible)
        except Exception:
            return False

    def _derived_subcortical_hover_visible(self) -> bool:
        try:
            return bool(self._show_seg_in_derived_views and self.sub_seg_cb.isChecked() and self.subcortical_seg_layer is not None and self.subcortical_seg_layer.visible)
        except Exception:
            return False

    def _raw_cortical_hover_visible(self) -> bool:
        try:
            return bool(self.cort_seg_cb.isChecked() and self.seg_cortical_layer is not None and self.seg_cortical_layer.visible)
        except Exception:
            return False

    def _raw_subcortical_hover_visible(self) -> bool:
        try:
            return bool(self.sub_seg_cb.isChecked() and self.seg_subcortical_layer is not None and self.seg_subcortical_layer.visible)
        except Exception:
            return False

    def _bind_hover_label_overlays(self):
        self.raw_canvas.set_intensity_layer(self.raw_layer)
        self.raw_canvas.set_hover_label_sources([
            {
                'layer': self.seg_subcortical_layer,
                'label_map': self._subcortical_hover_label_map,
                'visibility_getter': self._raw_subcortical_hover_visible,
            },
            {
                'layer': self.seg_cortical_layer,
                'label_map': self._cortical_hover_label_map,
                'visibility_getter': self._raw_cortical_hover_visible,
            },
        ])
        self.cortical_canvas.set_intensity_layer(self.cortical_layer)
        self.cortical_canvas.set_hover_label_source(
            self.cortical_seg_layer,
            self._cortical_hover_label_map,
            visibility_getter=self._derived_cortical_hover_visible,
        )
        self.subcortical_canvas.set_intensity_layer(self.subcortical_layer)
        self.subcortical_canvas.set_hover_label_source(
            self.subcortical_seg_layer,
            self._subcortical_hover_label_map,
            visibility_getter=self._derived_subcortical_hover_visible,
        )

    def _refresh_hover_label_overlays(self):
        for canvas in (self.raw_canvas, self.cortical_canvas, self.subcortical_canvas):
            try:
                canvas.refresh_hover_label()
            except Exception:
                canvas.clear_hover_info()

    # ── menu ─────────────────────────────────────────────────────────────────
    def _build_menu(self):
        from qtpy.QtWidgets import QActionGroup

        # ── File menu ─────────────────────────────────────────────────────
        fm = self.menuBar().addMenu("File")
        act_paths = QAction("Change file paths…", self)
        act_paths.setShortcut("Ctrl+,")
        act_paths.setToolTip("Reopen the path setup dialog without closing the main window")
        act_paths.triggered.connect(self._on_change_paths)
        fm.addAction(act_paths)

        # ── View menu ─────────────────────────────────────────────────────
        vm = self.menuBar().addMenu("View")

        act_contrast = QAction("Contrast && Overlay…", self)
        act_contrast.setShortcut("Ctrl+L")
        act_contrast.triggered.connect(self.contrast_dlg.show)
        vm.addAction(act_contrast)

        act_label_desc = QAction("Change label descriptions…", self)
        act_label_desc.triggered.connect(self._on_change_label_descriptions)
        vm.addAction(act_label_desc)

        vm.addSeparator()
        self._act_show_seg_derived = QAction(
            "Show segmentation in cortical/subcortical QSM", self,
            checkable=True, checked=self._show_seg_in_derived_views)
        self._act_show_seg_derived.toggled.connect(self._on_show_seg_derived_toggled)
        vm.addAction(self._act_show_seg_derived)

        self._act_show_view_directions = QAction(
            "Show view directions", self, checkable=True, checked=True)
        self._act_show_view_directions.toggled.connect(self._on_show_view_directions_toggled)
        vm.addAction(self._act_show_view_directions)

        # ── Canonical (reorientation) sub-menu ────────────────────────────
        vm.addSeparator()
        canonical_menu = vm.addMenu("Canonical orientation")
        canonical_group = QActionGroup(self)
        canonical_group.setExclusive(True)
        self._canonical_actions: Dict[str, QAction] = {}
        for key in CANONICAL_OPTIONS:
            act = QAction(key, self, checkable=True)
            act.setChecked(key == DEFAULT_CANONICAL)
            canonical_group.addAction(act)
            canonical_menu.addAction(act)
            self._canonical_actions[key] = act
        canonical_group.triggered.connect(self._on_canonical_action)

        # ── View orientation sub-menu ─────────────────────────────────────
        orient_menu = vm.addMenu("View orientation")
        orient_group = QActionGroup(self)
        orient_group.setExclusive(True)
        self._orient_actions: Dict[str, QAction] = {}
        for key in ORIENTATIONS:
            act = QAction(key, self, checkable=True)
            act.setChecked(key == DEFAULT_ORIENTATION)
            orient_group.addAction(act)
            orient_menu.addAction(act)
            self._orient_actions[key] = act
        orient_group.triggered.connect(self._on_orient_action)

        # ── Tools menu ────────────────────────────────────────────────────
        tm = self.menuBar().addMenu("Tools")
        self._act_save_gen = QAction(
            "Save generated QSM files", self,
            checkable=True, checked=self._initial_save_generated_qsm)
        tm.addAction(self._act_save_gen)

    # ── Orientation menu callbacks ─────────────────────────────────────────────
    def _on_canonical_action(self, action: QAction):
        """Called when a Canonical orientation menu item is triggered."""
        # canonical_key is read directly from the checked action in _apply_orientation
        if not self._is_loading():
            self.load_case(self.case_index)

    def _on_orient_action(self, action: QAction):
        """Called when a View orientation menu item is triggered."""
        if not self._is_loading():
            self._apply_orientation()

    def _on_show_view_directions_toggled(self, checked: bool):
        for canvas in self._all_canvases():
            try:
                canvas.set_direction_overlay_visible(checked)
            except Exception:
                pass

    def _current_canonical_key(self) -> str:
        for key, act in self._canonical_actions.items():
            if act.isChecked():
                return key
        return DEFAULT_CANONICAL

    def _current_orient_name(self) -> str:
        for key, act in self._orient_actions.items():
            if act.isChecked():
                return key
        return DEFAULT_ORIENTATION

    def _build_toolbar(self):
        """No standalone toolbar — orientation controls moved to View menu."""
        pass

    # ── File → Change file paths ──────────────────────────────────────────────
    def _on_change_paths(self):
        """Open the path-setup dialog without closing the main window."""
        cfg = self._current_config
        dlg = StartupConfigDialog(
            parent=self,
            defaults_root      = cfg.cases_root  if cfg else CASES_ROOT,
            defaults_output_csv= cfg.output_csv  if cfg else OUTPUT_CSV,
            defaults_file_names= cfg.file_names  if cfg else FILE_NAMES,
        )
        dlg.setWindowTitle("QSM QC Reviewer – Change File Paths")
        # Show the "Continue" button as "Apply"
        if hasattr(dlg, 'continue_btn'):
            dlg.continue_btn.setText("Apply")
        if dlg.exec_() != QDialog.Accepted or dlg.config is None:
            return
        self._apply_config_change(dlg.config)


    def _on_change_label_descriptions(self):
        dlg = LabelDescriptionConfigDialog(self)
        if dlg.exec_() != QDialog.Accepted:
            return
        selected = dlg.selected_paths()
        set_label_description_override_paths(
            cortical_path=selected.get("cortical"),
            subcortical_path=selected.get("subcortical"),
            use_default=bool(selected.get("use_default")),
        )
        self._reload_label_description_maps()
        self._set_status("Label descriptions updated", _C_SUCCESS)

    def _reload_label_description_maps(self):
        self._cortical_hover_label_map, self._subcortical_hover_label_map = load_hover_label_maps()
        self._bind_hover_label_overlays()
        self._refresh_hover_label_overlays()

    def _apply_config_change(self, new_cfg: 'AppConfig'):
        """Diff new vs old config and apply the minimal necessary update."""
        old = self._current_config
        if old is None:
            # First call from main(); just store and return — scan is started separately
            self._current_config = new_cfg
            return

        csv_changed   = new_cfg.output_csv != old.output_csv
        root_changed  = new_cfg.cases_root  != old.cases_root
        files_changed = new_cfg.file_names  != old.file_names
        opts_changed  = (new_cfg.save_generated_qsm      != old.save_generated_qsm or
                         new_cfg.show_seg_in_derived_views != old.show_seg_in_derived_views)

        # ── Case 1: nothing data-related changed ──────────────────────────
        if not csv_changed and not root_changed and not files_changed:
            self._act_save_gen.setChecked(new_cfg.save_generated_qsm)
            if new_cfg.show_seg_in_derived_views != self._show_seg_in_derived_views:
                self._act_show_seg_derived.setChecked(new_cfg.show_seg_in_derived_views)
            self._current_config = new_cfg
            self._set_status("Options updated", _C_SUCCESS)
            return

        # ── Case 2: only CSV path changed ─────────────────────────────────
        # Write current in-memory results to new path; no reload needed.
        if csv_changed and not root_changed and not files_changed:
            try:
                write_results(new_cfg.output_csv, self.results)
            except Exception as exc:
                QMessageBox.warning(self, "CSV write failed",
                                    f"Could not write to new CSV:\n{exc}")
                return
            self.output_csv = new_cfg.output_csv
            if opts_changed:
                self._act_save_gen.setChecked(new_cfg.save_generated_qsm)
                if new_cfg.show_seg_in_derived_views != self._show_seg_in_derived_views:
                    self._act_show_seg_derived.setChecked(new_cfg.show_seg_in_derived_views)
            self._current_config = new_cfg
            self._set_status(
                f"CSV path changed → {os.path.basename(new_cfg.output_csv)}", _C_SUCCESS)
            return

        # ── Case 3: data folder or file names changed → full reinit ───────
        # Auto-save current labels first
        if self.cases and not self.current_saved:
            self.save_current_case()

        # Stop all background threads
        self._cleanup_load_thread()
        self._cleanup_scan_thread()
        self._prefetch_queue = []
        self._reinit_gen += 1        # invalidates any in-flight prefetch threads
        self._loading = False
        self._pending_load_index = None

        # Clear case list and caches
        self.cases.clear()
        self._case_items.clear()
        self.case_list.clear()
        self._data_cache.clear()
        self.case_index = 0
        self.current_saved = True

        # Clear all napari layers to avoid shape-mismatch on next load
        for canvas in self._all_canvases():
            try:
                canvas.viewer.layers.clear()
            except Exception:
                pass
        for attr in ('raw_layer', 'seg_cortical_layer', 'seg_subcortical_layer',
                     'cortical_layer', 'cortical_seg_layer',
                     'subcortical_layer', 'subcortical_seg_layer'):
            setattr(self, attr, None)
        self.raw_data = self.cortical_data = self.subcortical_data = None
        self.cortical_roi_data = self.cortical_cube_data = self.cube_mask_data = None
        self._reoriented_affine = None
        self._zooms = None

        # Apply new config values
        self.output_csv  = new_cfg.output_csv
        self.file_names  = dict(new_cfg.file_names)
        self.results     = read_existing_results(new_cfg.output_csv)
        self._act_save_gen.setChecked(new_cfg.save_generated_qsm)
        if new_cfg.show_seg_in_derived_views != self._show_seg_in_derived_views:
            self._act_show_seg_derived.setChecked(new_cfg.show_seg_in_derived_views)
        self._current_config = new_cfg

        # Reset UI labels
        self.case_label.setText("-")
        self.status_label.setText("Status:  –")
        self.status_label.setStyleSheet(f"color:{_C_DIM}; font-size:{_BASE_PT}pt;")
        self._set_nav_enabled(False)

        # Kick off fresh scan (deferred so the dialog closes first)
        QTimer.singleShot(0, lambda: self.start_case_scan(new_cfg.cases_root))



    # ── status bar ────────────────────────────────────────────────────────────
    def _build_statusbar(self):
        sb = self.statusBar()
        self._status_msg = QLabel("Ready")
        self._status_msg.setStyleSheet(f"color:{_C_DIM}; font-size:9pt; padding:0 8px;")
        sb.addWidget(self._status_msg, 1)
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setFixedWidth(220)
        self._progress_bar.setVisible(False)
        sb.addPermanentWidget(self._progress_bar)
    def _set_status(self, msg: str, color: str = _C_DIM):
        self._status_msg.setStyleSheet(f"color:{color}; font-size:9pt; padding:0 8px;")
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
        # top row: Raw | Cortical — collapsible=False but NO minimum size constraint
        # so the divider can be dragged to any position
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
        # Connect splitter moves → debounced fit + keep top/bot aligned
        for spl in (self._v_splitter, self._top_splitter, self._bot_splitter,
                    self._main_splitter):
            spl.splitterMoved.connect(lambda *_: self._fit_timer.start())
        # Deferred size setup
        QTimer.singleShot(0, self._init_splitter_sizes)
        # QC Grading Criteria floating overlay (child of central widget)
        self._qc_overlay = _QCRulesOverlay(self.centralWidget())
    def _init_splitter_sizes(self):
        w = self._main_splitter.width()
        if w <= 10:
            return
        left_w = int(w * 0.72)
        self._main_splitter.setSizes([left_w, w - left_w])
        h = self._v_splitter.height()
        if h > 10:
            self._v_splitter.setSizes([h // 2, h // 2])
        # Set stretch factors on image canvases so the splitter divider can be
        # dragged freely to any position (not locked to equal halves)
        top_w = max(1, self._top_splitter.width())
        self._top_splitter.setSizes([top_w // 2, top_w - top_w // 2])
        self._top_splitter.setStretchFactor(0, 1)
        self._top_splitter.setStretchFactor(1, 1)
        bot_w = max(1, self._bot_splitter.width())
        self._bot_splitter.setSizes([bot_w // 2, bot_w - bot_w // 2])
        self._bot_splitter.setStretchFactor(0, 1)
        self._bot_splitter.setStretchFactor(1, 1)
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
        lay.setContentsMargins(_PANEL_MARGIN, _PANEL_MARGIN, _PANEL_MARGIN, _PANEL_MARGIN)
        lay.setSpacing(_PANEL_SPACING)

        # ── Case info ──────────────────────────────────────────────────────
        gb = QGroupBox("Case")
        vl = QVBoxLayout(gb); vl.setSpacing(4)
        self.case_label = QLabel("-")
        self.case_label.setStyleSheet(
            f"font-family: Consolas, Menlo, monospace; font-weight:700;"
            f" font-size:{'13' if _IS_WINDOWS else str(_BASE_PT+4)}pt; color:{_C_ACCENT};")
        self.status_label = QLabel("Status: unsaved")
        self.status_label.setStyleSheet(f"font-size:{_BASE_PT}pt;")
        self.source_label = QLabel("Sources: -")
        self.source_label.setStyleSheet(f"color:{_C_DIM}; font-size:{_SMALL_PT}pt;")
        self.orient_label = QLabel("Native: -")
        self.orient_label.setStyleSheet(f"color:{_C_DIM}; font-size:{_SMALL_PT}pt;")
        self.spacing_label = QLabel("Spacing: -")
        self.spacing_label.setStyleSheet(f"color:{_C_DIM}; font-size:{_SMALL_PT}pt;")
        for w in (self.case_label, self.status_label,
                  self.source_label, self.orient_label, self.spacing_label):
            vl.addWidget(w)
        lay.addWidget(gb)

        # ── Overlay toggles ───────────────────────────────────────────────
        ob = QGroupBox("Overlay")
        ol = QVBoxLayout(ob); ol.setSpacing(6)
        self.cort_seg_cb = QCheckBox("Cortical labels (SynthSeg ROIs)")
        self.cort_seg_cb.setChecked(DEFAULT_CORT_SEG_VISIBLE)
        self.sub_seg_cb  = QCheckBox("Subcortical labels")
        self.sub_seg_cb.setChecked(DEFAULT_SUB_SEG_VISIBLE)
        ol.addWidget(self.cort_seg_cb)
        ol.addWidget(self.sub_seg_cb)
        lay.addWidget(ob)

        # ── Hotkeys ───────────────────────────────────────────────────────
        hb = QGroupBox("Hotkeys")
        hl = QVBoxLayout(hb)
        hotkey_lbl = QLabel(
            "Wheel       scroll slices (respects Sync)\n"
            "Ctrl+Wheel  zoom current view\n"
            "Ctrl+L      Contrast & Overlay\n"
            "C           Toggle cortical overlay\n"
            "S           Toggle subcortical overlay\n"
            "N / P       Next / Prev case (auto-save)\n"
            "Ctrl+S      Save")
        hotkey_lbl.setStyleSheet(
            f"font-family: Consolas, Menlo, monospace; font-size:{_SMALL_PT}pt; color:{_C_DIM};")
        hotkey_lbl.setWordWrap(False)
        hl.addWidget(hotkey_lbl)
        lay.addWidget(hb)

        # ── QC Grading Criteria entry ─────────────────────────────────────
        qc_entry = QFrame()
        qc_entry.setObjectName("_qcRulesEntry")
        qc_entry.setStyleSheet(
            "#_qcRulesEntry {"
            f"  background: {_C_BG}; border: 1px solid {_C_BORDER};"
            "  border-radius: 4px;"
            "}"
            "#_qcRulesEntry:hover {"
            f"  border-color: {_C_ACCENT};"
            "}"
        )
        qc_entry.setCursor(Qt.PointingHandCursor)
        qce_lay = QHBoxLayout(qc_entry)
        qce_lay.setContentsMargins(
            4 if _IS_WINDOWS else 6,
            2 if _IS_WINDOWS else 4,
            4 if _IS_WINDOWS else 6,
            2 if _IS_WINDOWS else 4,
        )
        qce_lbl = QLabel("▶  QC Grading Criteria")
        qce_lbl.setStyleSheet(
            f"color: {_C_ACCENT}; font-size: {_SMALL_PT}pt; "
            "border: none; background: transparent;"
        )
        qce_lbl.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        qce_lay.addWidget(qce_lbl)
        hint = QLabel("hover to view")
        hint.setStyleSheet(
            f"color: {_C_DIM}; font-size: {_SMALL_PT - (1 if _IS_WINDOWS else 1)}pt; "
            "border: none; background: transparent;"
        )
        hint.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        qce_lay.addStretch(1)
        qce_lay.addWidget(hint)
        qc_entry.installEventFilter(self)
        self._qc_rules_entry = qc_entry
        lay.addWidget(qc_entry)

        lay.addStretch(1)

        self.cort_seg_cb.toggled.connect(
            lambda v: self._update_seg_visibility("cortical", v))
        self.sub_seg_cb.toggled.connect(
            lambda v: self._update_seg_visibility("subcortical", v))
        return panel
    # ── QC panel ──────────────────────────────────────────────────────────────
    # def _build_qc_panel(self) -> QWidget:
    #     panel = QWidget()
    #     panel.setStyleSheet(f"QWidget {{ background: {_C_PANEL}; }}")
    #     panel.setMinimumWidth(260)

    #     cl = QVBoxLayout(panel)
    #     if _IS_WINDOWS:
    #         cl.setContentsMargins(5, 12, 12, 12)
    #         cl.setSpacing(10)
    #     else:
    #         fm = panel.fontMetrics()
    #         _sp  = max(2, fm.height() // 3)
    #         _isp = max(1, fm.height() // 5)
    #         _mg  = max(4, fm.height() // 2)
    #         cl.setContentsMargins(_mg, _mg, _mg, _mg)
    #         cl.setSpacing(_sp)

    #     # ── Motion QC Scores ──────────────────────────────────────────────
    #     sg = QGroupBox("Motion QC Scores")
    #     if _IS_WINDOWS:
    #         sf = QFormLayout(sg); sf.setSpacing(8)
    #         self.cortex_combo = _make_combo(MOTION_SCORES, placeholder="")
    #         self.subcortex_combo = _make_combo(MOTION_SCORES, placeholder="")
    #         sf.addRow("Cortex:",    self.cortex_combo)
    #         sf.addRow("Subcortex:", self.subcortex_combo)
    #     else:
    #         sv = QVBoxLayout(sg)
    #         sv.setSpacing(_sp)
    #         cortex_row = QVBoxLayout(); cortex_row.setSpacing(_isp)
    #         cortex_row_lbl = QLabel("Cortex:")
    #         cortex_row_lbl.setStyleSheet(f"color:{_C_DIM}; font-size:{_SMALL_PT}pt;")
    #         self.cortex_combo = _make_combo(MOTION_SCORES, placeholder="")
    #         cortex_row.addWidget(cortex_row_lbl)
    #         cortex_row.addWidget(self.cortex_combo)
    #         sv.addLayout(cortex_row)
    #         subcortex_row = QVBoxLayout(); subcortex_row.setSpacing(_isp)
    #         subcortex_row_lbl = QLabel("Subcortex:")
    #         subcortex_row_lbl.setStyleSheet(f"color:{_C_DIM}; font-size:{_SMALL_PT}pt;")
    #         self.subcortex_combo = _make_combo(MOTION_SCORES, placeholder="")
    #         subcortex_row.addWidget(subcortex_row_lbl)
    #         subcortex_row.addWidget(self.subcortex_combo)
    #         sv.addLayout(subcortex_row)
    #     cl.addWidget(sg)

    #     # ── Segmentation Accuracy ─────────────────────────────────────────
    #     LABEL_ACCURACY = ["0 – Good", "1 – Bad"]
    #     lag = QGroupBox("Segmentation Accuracy  (optional)")
    #     if _IS_WINDOWS:
    #         laf = QFormLayout(lag); laf.setSpacing(8)
    #         self.cort_label_combo = _make_combo(LABEL_ACCURACY, placeholder="")
    #         self.sub_label_combo = _make_combo(LABEL_ACCURACY, placeholder="")
    #         laf.addRow("Cortical:",    self.cort_label_combo)
    #         laf.addRow("Subcortical:", self.sub_label_combo)
    #     else:
    #         lv2 = QVBoxLayout(lag); lv2.setSpacing(_sp)
    #         cort_la_row = QVBoxLayout(); cort_la_row.setSpacing(_isp)
    #         cort_la_lbl = QLabel("Cortical:")
    #         cort_la_lbl.setStyleSheet(f"color:{_C_DIM}; font-size:{_SMALL_PT}pt;")
    #         self.cort_label_combo = _make_combo(LABEL_ACCURACY, placeholder="")
    #         cort_la_row.addWidget(cort_la_lbl)
    #         cort_la_row.addWidget(self.cort_label_combo)
    #         lv2.addLayout(cort_la_row)
    #         sub_la_row = QVBoxLayout(); sub_la_row.setSpacing(_isp)
    #         sub_la_lbl = QLabel("Subcortical:")
    #         sub_la_lbl.setStyleSheet(f"color:{_C_DIM}; font-size:{_SMALL_PT}pt;")
    #         self.sub_label_combo = _make_combo(LABEL_ACCURACY, placeholder="")
    #         sub_la_row.addWidget(sub_la_lbl)
    #         sub_la_row.addWidget(self.sub_label_combo)
    #         lv2.addLayout(sub_la_row)
    #     cl.addWidget(lag)
    #     # Notes
    #     ng = QGroupBox("Notes")
    #     nl = QVBoxLayout(ng)
    #     self.notes_edit = QTextEdit()
    #     self.notes_edit.setPlaceholderText("Optional notes…")
    #     self.notes_edit.setMinimumHeight(_NOTES_MIN_H)
    #     self.notes_edit.setMaximumHeight(_NOTES_MAX_H)
    #     nl.addWidget(self.notes_edit)
    #     cl.addWidget(ng)
    #     # Review flag
    #     fg = QGroupBox("Review")
    #     fl = QVBoxLayout(fg)
    #     self.review_flag_cb = QCheckBox("Flag this case for later review")
    #     fl.addWidget(self.review_flag_cb)
    #     cl.addWidget(fg)
    #     # Save
    #     self._btn_save = QPushButton("💾   Save   (Ctrl+S)")
    #     self._btn_save.setStyleSheet(_SAVE_BTN_CSS)
    #     self._btn_save.clicked.connect(self.save_current_case)
    #     cl.addWidget(self._btn_save)
    #     # Navigation
    #     nav = QGroupBox("Navigation")
    #     nl2 = QHBoxLayout(nav); nl2.setSpacing(8)
    #     self.btn_prev = QPushButton("◀  Prev")
    #     self.btn_next = QPushButton("Next  ▶")
    #     for b in (self.btn_prev, self.btn_next):
    #         b.setStyleSheet(_NAV_BTN_CSS)
    #     self.btn_prev.clicked.connect(self.prev_case)
    #     self.btn_next.clicked.connect(self.next_case)
    #     nl2.addWidget(self.btn_prev)
    #     nl2.addWidget(self.btn_next)
    #     cl.addWidget(nav)
    #     # Case list
    #     lg = QGroupBox("Case List")
    #     ll2 = QVBoxLayout(lg)
    #     self.case_list = QListWidget()
    #     self._case_items: Dict[str, QListWidgetItem] = {}
    #     for c in self.cases:
    #         item = QListWidgetItem(c.case_id)
    #         item.setData(Qt.UserRole, c.case_id)
    #         self.case_list.addItem(item)
    #         self._case_items[c.case_id] = item
    #     self.case_list.setMinimumHeight(_CASELIST_MIN_H)
    #     self.case_list.itemClicked.connect(self.on_case_clicked)
    #     ll2.addWidget(self.case_list)
    #     cl.addWidget(lg, 1)
    #     self.cortex_combo.currentTextChanged.connect(self._mark_unsaved)
    #     self.subcortex_combo.currentTextChanged.connect(self._mark_unsaved)
    #     self.cortex_combo.currentTextChanged.connect(lambda _: self._clear_qc_error_if_filled(self.cortex_combo))
    #     self.subcortex_combo.currentTextChanged.connect(lambda _: self._clear_qc_error_if_filled(self.subcortex_combo))
    #     self.cort_label_combo.currentTextChanged.connect(self._mark_unsaved)
    #     self.sub_label_combo.currentTextChanged.connect(self._mark_unsaved)
    #     self.notes_edit.textChanged.connect(self._mark_unsaved)
    #     self.review_flag_cb.toggled.connect(self._mark_unsaved)
    #     self.cortex_combo.currentTextChanged.connect(lambda _: self._refresh_current_case_list_item())
    #     self.subcortex_combo.currentTextChanged.connect(lambda _: self._refresh_current_case_list_item())
    #     self.cort_label_combo.currentTextChanged.connect(lambda _: self._refresh_current_case_list_item())
    #     self.sub_label_combo.currentTextChanged.connect(lambda _: self._refresh_current_case_list_item())
    #     self.notes_edit.textChanged.connect(self._refresh_current_case_list_item)
    #     self.review_flag_cb.toggled.connect(lambda _: self._refresh_current_case_list_item())
    #     return panel
    def _build_qc_panel(self) -> QWidget:
        panel = QWidget()
        panel.setStyleSheet(f"QWidget {{ background: {_C_PANEL}; }}")
        panel.setMinimumWidth(260)

        # 强制写死极其紧凑的间距，不再依赖原生的多余 padding
        _sp  = 4   # 各个 GroupBox 之间的垂直间距
        _isp = 2   # Label 和下拉框之间的微小间距
        _mg  = 4   # 整个面板靠边缘的留白
        
        # 统一的紧凑型 GroupBox 内部边距 (左, 上, 右, 下)
        # 上边距(12)是专门为了不遮挡标题文字预留的，其余压紧到极限
        _gb_margins = (6, 12, 6, 6)

        cl = QVBoxLayout(panel)
        cl.setContentsMargins(_mg, _mg, _mg, _mg)
        cl.setSpacing(_sp)

        # ── Motion QC Scores ──────────────────────────────────────────────
        sg = QGroupBox("Motion QC Scores")
        sv = QVBoxLayout(sg)
        sv.setContentsMargins(*_gb_margins) # 注入强力紧凑边距
        sv.setSpacing(_sp)

        # Cortex row
        cortex_row = QVBoxLayout()
        cortex_row.setSpacing(_isp)
        cortex_row_lbl = QLabel("Cortex:")
        cortex_row_lbl.setStyleSheet(f"color:{_C_DIM}; font-size:{_SMALL_PT}pt;")
        self.cortex_combo = _make_combo(MOTION_SCORES, placeholder="")
        cortex_row.addWidget(cortex_row_lbl)
        cortex_row.addWidget(self.cortex_combo)
        sv.addLayout(cortex_row)

        # Subcortex row
        subcortex_row = QVBoxLayout()
        subcortex_row.setSpacing(_isp)
        subcortex_row_lbl = QLabel("Subcortex:")
        subcortex_row_lbl.setStyleSheet(f"color:{_C_DIM}; font-size:{_SMALL_PT}pt;")
        self.subcortex_combo = _make_combo(MOTION_SCORES, placeholder="")
        subcortex_row.addWidget(subcortex_row_lbl)
        subcortex_row.addWidget(self.subcortex_combo)
        sv.addLayout(subcortex_row)
        cl.addWidget(sg)

        # # ── Segmentation Accuracy ─────────────────────────────────────────
        if SEGMENTATION_TASK == True:
            LABEL_ACCURACY = ["0 – Good", "1 – Bad"]
            lag = QGroupBox("Segmentation Accuracy  (optional)")
            lv2 = QVBoxLayout(lag)
            lv2.setContentsMargins(*_gb_margins) # 注入强力紧凑边距
            lv2.setSpacing(_sp)

            cort_la_row = QVBoxLayout()
            cort_la_row.setSpacing(_isp)
            cort_la_lbl = QLabel("Cortical:")
            cort_la_lbl.setStyleSheet(f"color:{_C_DIM}; font-size:{_SMALL_PT}pt;")
            self.cort_label_combo = _make_combo(LABEL_ACCURACY, placeholder="")
            cort_la_row.addWidget(cort_la_lbl)
            cort_la_row.addWidget(self.cort_label_combo)
            lv2.addLayout(cort_la_row)

            sub_la_row = QVBoxLayout()
            sub_la_row.setSpacing(_isp)
            sub_la_lbl = QLabel("Subcortical:")
            sub_la_lbl.setStyleSheet(f"color:{_C_DIM}; font-size:{_SMALL_PT}pt;")
            self.sub_label_combo = _make_combo(LABEL_ACCURACY, placeholder="")
            sub_la_row.addWidget(sub_la_lbl)
            sub_la_row.addWidget(self.sub_label_combo)
            lv2.addLayout(sub_la_row)
            cl.addWidget(lag)
        
        # ── Notes ─────────────────────────────────────────────────────────
        ng = QGroupBox("Notes")
        nl = QVBoxLayout(ng)
        nl.setContentsMargins(*_gb_margins) # 注入强力紧凑边距
        self.notes_edit = QTextEdit()
        self.notes_edit.setPlaceholderText("Optional notes…")
        self.notes_edit.setMinimumHeight(_NOTES_MIN_H)
        self.notes_edit.setMaximumHeight(_NOTES_MAX_H)
        nl.addWidget(self.notes_edit)
        cl.addWidget(ng)
        
        # ── Review & Save (50/50 并排) ────────────────────────────────────
        rs_layout = QHBoxLayout()
        rs_layout.setSpacing(8)
        
        # 左侧 50%：独立的 Review 边框
        fg = QGroupBox("Review")
        fl = QVBoxLayout(fg)
        fl.setContentsMargins(*_gb_margins)
        self.review_flag_cb = QCheckBox("Flag for review")
        self.review_flag_cb.setToolTip("Flag this case for later review")
        fl.addWidget(self.review_flag_cb)
        rs_layout.addWidget(fg, 1)  
        
        # 右侧 50%：Save 按钮
        # 核心：套一层垂直布局，顶部向下推一个刚好等于标题预留高度的边距
        btn_layout = QVBoxLayout()
        btn_layout.setContentsMargins(0, _GB_MARGIN_TOP, 0, 0) 
        
        self._btn_save = QPushButton("💾   Save   (Ctrl+S)")
        self._btn_save.setStyleSheet(_SAVE_BTN_CSS)
        self._btn_save.clicked.connect(self.save_current_case)
        self._btn_save.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        
        btn_layout.addWidget(self._btn_save)
        rs_layout.addLayout(btn_layout, 1) # 将包含边距的布局加入到这一行中
        cl.addLayout(rs_layout)
        
        # ── Navigation ────────────────────────────────────────────────────
        nav = QGroupBox("Navigation")
        nl2 = QHBoxLayout(nav)
        nl2.setContentsMargins(*_gb_margins) # 注入强力紧凑边距
        nl2.setSpacing(6)
        self.btn_prev = QPushButton("◀  Prev")
        self.btn_next = QPushButton("Next  ▶")
        for b in (self.btn_prev, self.btn_next):
            b.setStyleSheet(_NAV_BTN_CSS)
        self.btn_prev.clicked.connect(self.prev_case)
        self.btn_next.clicked.connect(self.next_case)
        nl2.addWidget(self.btn_prev)
        nl2.addWidget(self.btn_next)
        cl.addWidget(nav)
        
        # ── Case list ─────────────────────────────────────────────────────
        lg = QGroupBox("Case List")
        ll2 = QVBoxLayout(lg)
        ll2.setContentsMargins(6, 12, 6, 6) # 注入强力紧凑边距
        self.case_list = QListWidget()
        self._case_items: Dict[str, QListWidgetItem] = {}
        for c in self.cases:
            item = QListWidgetItem(c.case_id)
            item.setData(Qt.UserRole, c.case_id)
            self.case_list.addItem(item)
            self._case_items[c.case_id] = item
        self.case_list.setMinimumHeight(_CASELIST_MIN_H)
        self.case_list.itemClicked.connect(self.on_case_clicked)
        ll2.addWidget(self.case_list)
        cl.addWidget(lg, 1)
        
        # --- Signal connections ---
        self.cortex_combo.currentTextChanged.connect(self._mark_unsaved)
        self.subcortex_combo.currentTextChanged.connect(self._mark_unsaved)
        self.cortex_combo.currentTextChanged.connect(lambda _: self._clear_qc_error_if_filled(self.cortex_combo))
        self.subcortex_combo.currentTextChanged.connect(lambda _: self._clear_qc_error_if_filled(self.subcortex_combo))
        if SEGMENTATION_TASK == True:
            self.cort_label_combo.currentTextChanged.connect(self._mark_unsaved)
            self.sub_label_combo.currentTextChanged.connect(self._mark_unsaved)
        self.notes_edit.textChanged.connect(self._mark_unsaved)
        self.review_flag_cb.toggled.connect(self._mark_unsaved)
        self.cortex_combo.currentTextChanged.connect(lambda _: self._refresh_current_case_list_item())
        self.subcortex_combo.currentTextChanged.connect(lambda _: self._refresh_current_case_list_item())
        if SEGMENTATION_TASK == True:
            self.cort_label_combo.currentTextChanged.connect(lambda _: self._refresh_current_case_list_item())
            self.sub_label_combo.currentTextChanged.connect(lambda _: self._refresh_current_case_list_item())
        self.notes_edit.textChanged.connect(self._refresh_current_case_list_item)
        self.review_flag_cb.toggled.connect(lambda _: self._refresh_current_case_list_item())
        
        return panel
    # ── wheel filters ─────────────────────────────────────────────────────────
    def _install_wheel_filters(self):
        for canvas in self._all_canvases():
            def make_cb(c):
                def _on_wheel(event):
                    self._on_canvas_wheel(c, event)
                return _on_wheel
            canvas.install_wheel_filter(make_cb(canvas))
    def _on_canvas_zoom_changed(self, source: ImageCanvas):
        data = None
        if source is self.raw_canvas:
            data = self.raw_data
        elif source is self.cortical_canvas:
            data = self.cortical_data
        elif source is self.subcortical_canvas:
            data = self.subcortical_data
        if data is not None:
            source.fit_to_shape(data.shape, self._dims_order, self._zooms, force=True)

    def _flush_wheel_scroll(self):
        pending = list(self._wheel_pending_steps.values())
        self._wheel_pending_steps.clear()
        if not pending or self._loading:
            return
        self._syncing = True
        try:
            for source, steps in pending:
                if steps == 0:
                    continue
                sa = self._scroll_axis
                if source.sync_enabled:
                    active = [c for c in self._all_canvases() if c.sync_enabled] or [source]
                    new_z = get_slice(source.viewer, sa) + steps
                    for c in active:
                        set_slice(c.viewer, new_z, sa)
                else:
                    new_z = get_slice(source.viewer, sa) + steps
                    set_slice(source.viewer, new_z, sa)
        finally:
            self._syncing = False

    def _on_canvas_wheel(self, source: ImageCanvas, event):
        if self._loading:
            return
        dy = event.angleDelta().y()
        if dy == 0:
            return
        delta = -1 if dy > 0 else 1
        if bool(event.modifiers() & Qt.ControlModifier):
            source.step_zoom_by(delta, apply=True)
            return
        key = id(source)
        if key in self._wheel_pending_steps:
            _, prev = self._wheel_pending_steps[key]
            self._wheel_pending_steps[key] = (source, prev + delta)
        else:
            self._wheel_pending_steps[key] = (source, delta)
        self._wheel_timer.start()
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
    def _refresh_source_label(self):
        mode = self.cortical_mode_combo.currentText() if hasattr(self, "cortical_mode_combo") else "ROI only"
        cort_ok = self._cortical_roi_source_ok if mode == "ROI only" else self._cortical_cube_source_ok
        self.source_label.setText(
            f"Cortical ({mode}): {'file' if cort_ok else 'gen'}   "
            f"Subcortical: {'file' if self._subcortical_source_ok else 'gen'}")

    def _on_cortical_display_mode_changed(self, *_):
        if self._loading or self.raw_data is None:
            return
        mode = self.cortical_mode_combo.currentText()
        data = self.cortical_roi_data if mode == "ROI only" else self.cortical_cube_data
        if data is None:
            return
        self.cortical_data = data
        current_z = get_slice(self.cortical_canvas.viewer, self._scroll_axis)
        current_limits = tuple(self.cortical_layer.contrast_limits) if self.cortical_layer is not None else (DEFAULT_LOW, DEFAULT_HIGH)
        self._update_image_layer('cortical_layer', self.cortical_canvas, data, name='cortical_qsm', scale=self._zooms, contrast_limits=current_limits)
        seg_data = getattr(self, '_loaded_cortical_seg', None)
        if seg_data is not None:
            self._update_label_layer('cortical_seg_layer', self.cortical_canvas, seg_data, name='cortical_labels', scale=self._zooms, opacity=self.contrast_dlg.seg_opacity_spin.value(), visible=self.cort_seg_cb.isChecked() and self._show_seg_in_derived_views)
        self._apply_orientation(force_mid=False)
        set_slice(self.cortical_canvas.viewer, current_z, self._scroll_axis)
        self._refresh_source_label()

    # ── orientation ───────────────────────────────────────────────────────────
    def _apply_orientation(self, force_mid: bool = False):
        orient_name   = self._current_orient_name()
        canonical_key = self._current_canonical_key()
        cfg = ORIENTATIONS.get(orient_name, ORIENTATIONS[DEFAULT_ORIENTATION])
        self._scroll_axis   = cfg["axis"]
        self._dims_order    = cfg["order"]
        self._canonical_key = canonical_key
        auto_fv, auto_fh = _AUTO_FLIP.get((canonical_key, orient_name), (False, False))
        fv, fh = auto_fv, auto_fh
        self._flip_v, self._flip_h = fv, fh
        if canonical_key == "Native" and self._reoriented_affine is not None:
            left, right, top, bottom = native_dir_labels(
                self._reoriented_affine, orient_name)
            if fh: left, right = right, left
            if fv: top, bottom = bottom, top
        else:
            left, right, top, bottom = _VIEW_DIR_LABELS.get(
                orient_name, ('?','?','?','?'))
            if fh:
                left, right = right, left
            if fv:
                top, bottom = bottom, top
        for canvas, data in self._canvas_data_pairs():
            if data is None: continue
            canvas.set_view(self._dims_order, self._scroll_axis, data.shape, fv, fh)
            canvas.lock_scroll_mode()
            canvas.set_direction_labels(left, right, top, bottom)
            if hasattr(self, '_act_show_view_directions'):
                canvas.set_direction_overlay_visible(self._act_show_view_directions.isChecked())
            canvas.fit_to_shape(data.shape, self._dims_order, self._zooms)
        if force_mid and self.raw_data is not None:
            mid = self.raw_data.shape[self._scroll_axis] // 2
            self._set_all_slices_force(mid)
    # ── fit / resize ──────────────────────────────────────────────────────────
    def _fit_all(self, force: bool = False):
        if self._loading and not force:
            return
        for canvas, data in self._canvas_data_pairs():
            if data is not None:
                canvas.fit_to_shape(data.shape, self._dims_order, self._zooms, force=force)
    def eventFilter(self, obj, event):
        """Handle hover on the QC Grading Criteria entry label."""
        entry   = getattr(self, '_qc_rules_entry', None)
        overlay = getattr(self, '_qc_overlay', None)
        if entry is not None and overlay is not None and obj is entry:
            et = event.type()
            if et == QEvent.Enter:
                overlay.show_centered()
            elif et == QEvent.Leave:
                overlay.schedule_hide()
        return super().eventFilter(obj, event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_timer.start()
        # Reposition QC overlay if it is currently visible
        ov = getattr(self, '_qc_overlay', None)
        if ov is not None and ov.isVisible():
            ov.show_centered()
    # ── shortcuts ─────────────────────────────────────────────────────────────
    def _bind_shortcuts(self):
        for c in self._all_canvases():
            c.viewer.bind_key("C")(lambda _: self.cort_seg_cb.toggle())
            c.viewer.bind_key("S")(lambda _: self.sub_seg_cb.toggle())
            c.viewer.bind_key("N")(lambda _: self.next_case())
            c.viewer.bind_key("P")(lambda _: self.prev_case())
            c.viewer.bind_key("Control-S")(lambda _: self.save_current_case())
    # ── misc ─────────────────────────────────────────────────────────────────
    def current_case(self):
        if not self.cases or self.case_index < 0 or self.case_index >= len(self.cases):
            return None
        return self.cases[self.case_index]
    def current_case_id(self):
        case = self.current_case()
        return case.case_id if case is not None else ""
    def _set_qc_field_error(self, widget: QComboBox, has_error: bool):
        widget.setStyleSheet(_QC_ERROR_STYLE if has_error else "")

    def _clear_qc_error_if_filled(self, widget: QComboBox):
        if widget.currentText().strip():
            self._set_qc_field_error(widget, False)

    def _validate_motion_scores_before_leave(self) -> bool:
        missing = []
        if not self.cortex_combo.currentText().strip():
            missing.append((self.cortex_combo, "Cortex"))
        if not self.subcortex_combo.currentText().strip():
            missing.append((self.subcortex_combo, "Subcortex"))
        self._set_qc_field_error(self.cortex_combo, any(w is self.cortex_combo for w, _ in missing))
        self._set_qc_field_error(self.subcortex_combo, any(w is self.subcortex_combo for w, _ in missing))
        if not missing:
            return True
        names = " and ".join(label for _, label in missing)
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("Missing Motion QC score")
        box.setText(f"Please fill in the Motion QC score for: {names}.")
        box.setInformativeText("The empty Motion QC field(s) have been highlighted in red. Click OK to continue to the next case anyway, or Cancel to stay on this case.")
        box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        box.setDefaultButton(QMessageBox.Cancel)
        return box.exec_() == QMessageBox.Ok

    def _mark_unsaved(self, *_):
        self.current_saved = False
        self.status_label.setText("Status:  ✏ unsaved")
        self.status_label.setStyleSheet(f"color:{_C_WARN}; font-size:{_BASE_PT}pt;")
    def _update_seg_visibility(self, which, checked):
        if which == "cortical":
            if self.seg_cortical_layer:
                self.seg_cortical_layer.visible = checked
            if self.cortical_seg_layer:
                self.cortical_seg_layer.visible = checked and self._show_seg_in_derived_views
        elif which == "subcortical":
            if self.seg_subcortical_layer:
                self.seg_subcortical_layer.visible = checked
            if self.subcortical_seg_layer:
                self.subcortical_seg_layer.visible = checked and self._show_seg_in_derived_views
    def _on_show_seg_derived_toggled(self, checked):
        self._show_seg_in_derived_views = bool(checked)
        if self.cortical_seg_layer is not None:
            self.cortical_seg_layer.visible = self.cort_seg_cb.isChecked() and self._show_seg_in_derived_views
        if self.subcortical_seg_layer is not None:
            self.subcortical_seg_layer.visible = self.sub_seg_cb.isChecked() and self._show_seg_in_derived_views
    def _update_seg_opacity(self, value):
        for lyr in (self.seg_cortical_layer, self.seg_subcortical_layer,
                    self.cortical_seg_layer, self.subcortical_seg_layer):
            if lyr is not None: lyr.opacity = float(value)
    def apply_contrast(self, level, window, apply_all=True):
        lo, hi = level - window / 2, level + window / 2
        layers = [self.raw_layer]
        if apply_all: layers += [self.cortical_layer, self.subcortical_layer]
        for lyr in layers:
            if lyr is not None: lyr.contrast_limits = (lo, hi)
    @staticmethod
    def _score_to_label(score):
        score = _normalize_saved_choice(score)
        if not score:
            return ""
        for label in MOTION_SCORES:
            if label[0] == score:
                return label
        return score if score in MOTION_SCORES else ""
    @staticmethod
    def _label_to_score(label): return label[0] if label else ""
    def _collect_record(self):
        def _la(combo):
            t = combo.currentText()
            return t[0] if t else ""
        return QCRecord(
            case_id=self.current_case_id(),
            cortex_score=self._label_to_score(self.cortex_combo.currentText()),
            subcortex_score=self._label_to_score(self.subcortex_combo.currentText()),
            cort_label_ok=_la(self.cort_label_combo) if SEGMENTATION_TASK == True else "",
            sub_label_ok=_la(self.sub_label_combo) if SEGMENTATION_TASK == True else "",
            notes=self.notes_edit.toPlainText().strip(),
            marked_for_review=("1" if self.review_flag_cb.isChecked() else ""))
    @staticmethod
    def _record_is_completed(rec: Optional[QCRecord]) -> bool:
        return bool(rec and rec.cortex_score and rec.subcortex_score)
    @staticmethod
    def _record_is_flagged(rec: Optional[QCRecord]) -> bool:
        return bool(rec and str(rec.marked_for_review).strip() in {"1", "true", "True"})
    def _case_item_display_text(self, case_id: str, rec: Optional[QCRecord]) -> str:
        # Ensure case_id is always a printable ASCII/UTF-8 string
        safe_id = case_id if case_id else "(no id)"
        suffix = ""
        if self._record_is_completed(rec):
            suffix += " \u2713"   # ✓  — U+2713, safe on all platforms
        if self._record_is_flagged(rec):
            suffix += " [!]"      # ASCII-safe flag indicator
        return f"{safe_id}{suffix}"
    def _refresh_case_list_item(self, case_id: str, rec: Optional[QCRecord] = None):
        item = getattr(self, "_case_items", {}).get(case_id)
        if item is None:
            return
        if rec is None:
            rec = self.results.get(case_id)
        completed = self._record_is_completed(rec)
        flagged = self._record_is_flagged(rec)
        item.setText(self._case_item_display_text(case_id, rec))
        fg = QColor(_C_TEXT)
        if flagged:
            fg = QColor("#ffcc66")
        elif completed:
            fg = QColor(_C_SUCCESS)
        item.setForeground(fg)
        font = item.font()
        font.setBold(completed or flagged)
        item.setFont(font)
    def _refresh_all_case_list_items(self):
        for c in self.cases:
            self._refresh_case_list_item(c.case_id)
    def _refresh_current_case_list_item(self):
        if not getattr(self, "cases", None):
            return
        self._refresh_case_list_item(self.current_case_id(), self._collect_record())
    # ── save / navigation ─────────────────────────────────────────────────────
    def save_current_case(self):
        if not self.cases:
            return
        rec = self._collect_record()
        self.results[rec.case_id] = rec
        write_results(self.output_csv, self.results)
        self._refresh_case_list_item(rec.case_id, rec)
        self.current_saved = True
        self.status_label.setText("Status:  ✔ saved")
        self.status_label.setStyleSheet(f"color:{_C_SUCCESS}; font-size:{_BASE_PT}pt;")
    def next_case(self):
        if not self.cases or self.case_index >= len(self.cases) - 1 or self._is_loading(): return
        if not self._validate_motion_scores_before_leave():
            return
        self.save_current_case()
        self.load_case(self.case_index + 1)
    def prev_case(self):
        if not self.cases or self.case_index <= 0 or self._is_loading(): return
        self.save_current_case()
        self.load_case(self.case_index - 1)
    def on_case_clicked(self, item):
        if self._is_loading() or not self.cases: return
        for i, c in enumerate(self.cases):
            if c.case_id == item.data(Qt.UserRole) and i != self.case_index:
                if not self._validate_motion_scores_before_leave():
                    self.case_list.blockSignals(True)
                    self.case_list.setCurrentRow(self.case_index)
                    self.case_list.blockSignals(False)
                    return
                self.save_current_case()
                self.load_case(i)
                return
    def _is_loading(self):
        return self._loading
    def _set_nav_enabled(self, enabled: bool):
        has_cases = bool(self.cases)
        self.btn_prev.setEnabled(enabled and has_cases and self.case_index > 0)
        self.btn_next.setEnabled(enabled and has_cases and self.case_index < len(self.cases) - 1)
        self._btn_save.setEnabled(enabled and has_cases)
        self.case_list.setEnabled(enabled and has_cases)
        # Enable/disable orientation menu items while loading
        for act in getattr(self, '_canonical_actions', {}).values():
            act.setEnabled(enabled and has_cases)
        for act in getattr(self, '_orient_actions', {}).values():
            act.setEnabled(enabled and has_cases)
    # ── load case (background worker + cache) ─────────────────────────────────
    def _progress(self, pct: int, msg: str):
        self._progress_bar.setValue(int(pct))
        self._set_status(msg, _C_ACCENT)

    def load_case(self, index: int):
        if index < 0 or index >= len(self.cases):
            return
        if self._loading:
            self._pending_load_index = index
            return
        self._loading = True
        self.case_index = index
        case = self.cases[index]
        canonical_key = self._current_canonical_key()
        cache_key = self._cache_key_for(case, canonical_key)
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(True)
        self._set_status(f"Loading {case.case_id}…", _C_ACCENT)
        self._set_nav_enabled(False)
        if cache_key in self._data_cache:
            data = self._data_cache[cache_key]
            self._data_cache.move_to_end(cache_key)
            self._progress(100, "Loaded from cache")
            self._apply_loaded_data(data)
            self._loading = False
            self._set_nav_enabled(True)
            self._progress_bar.setVisible(False)
            if self._pending_load_index is not None and self._pending_load_index != self.case_index:
                nxt = self._pending_load_index
                self._pending_load_index = None
                self.load_case(nxt)
            else:
                # Also prefetch after cache hits
                QTimer.singleShot(300, self._prefetch_upcoming_cases)
            return

        self._cleanup_load_thread()
        self._load_thread = QThread(self)
        self._load_worker = LoadWorker(
            case=case,
            canonical_key=canonical_key,
            save_generated=self._act_save_gen.isChecked(),
            subcortical_margin=SUBCORTICAL_MARGIN,
            cortical_dilation_iter=CORTICAL_DILATION_ITER,
            display_mode=self.cortical_mode_combo.currentText(),
            file_names=self.file_names,
        )
        self._load_worker.moveToThread(self._load_thread)
        self._load_thread.started.connect(self._load_worker.run)
        self._load_worker.progressed.connect(self._progress)
        self._load_worker.finished.connect(lambda data, key=cache_key: self._on_load_finished(key, data))
        self._load_worker.failed.connect(self._on_load_failed)
        self._load_worker.finished.connect(self._cleanup_load_thread)
        self._load_worker.failed.connect(self._cleanup_load_thread)
        self._load_thread.start()

    def _on_load_finished(self, cache_key, data: dict):
        self._remember_cache(cache_key, data)
        self._apply_loaded_data(data)
        self._loading = False
        self._set_nav_enabled(True)
        self._progress_bar.setVisible(False)
        if self._pending_load_index is not None and self._pending_load_index != self.case_index:
            nxt = self._pending_load_index
            self._pending_load_index = None
            QTimer.singleShot(0, lambda: self.load_case(nxt))
        else:
            self._pending_load_index = None
            # Prefetch next few cases in background
            QTimer.singleShot(500, self._prefetch_upcoming_cases)

    def _prefetch_upcoming_cases(self, count: int = 3):
        """
        Start background loads of the next `count` cases so they are cached
        when the user navigates to them.  Uses separate worker threads (one
        at a time) so we never interfere with an ongoing primary load.
        """
        if self._loading or not self.cases:
            return
        canonical_key = self._current_canonical_key()
        targets = []
        for offset in range(1, count + 1):
            idx = self.case_index + offset
            if idx < len(self.cases):
                case = self.cases[idx]
                key = self._cache_key_for(case, canonical_key)
                if key not in self._data_cache:
                    targets.append((idx, case, key))
        if not targets:
            return
        # Kick off prefetch for the first uncached target; the rest follow
        # automatically when the prefetch worker finishes.
        self._prefetch_queue = targets
        self._start_next_prefetch(canonical_key)

    def _start_next_prefetch(self, canonical_key: str):
        if not getattr(self, '_prefetch_queue', None) or self._loading:
            return
        idx, case, cache_key = self._prefetch_queue.pop(0)
        if cache_key in self._data_cache:
            QTimer.singleShot(100, lambda: self._start_next_prefetch(canonical_key))
            return
        # Capture generation at submission time
        gen_at_submit = self._reinit_gen
        worker = LoadWorker(
            case=case,
            canonical_key=canonical_key,
            save_generated=self._act_save_gen.isChecked(),
            subcortical_margin=SUBCORTICAL_MARGIN,
            cortical_dilation_iter=CORTICAL_DILATION_ITER,
            display_mode=self.cortical_mode_combo.currentText(),
            file_names=self.file_names,
        )
        thread = QThread(self)
        thread.setObjectName(f"prefetch_{idx}")
        worker.moveToThread(thread)
        thread.started.connect(worker.run)

        def _on_done(data, key=cache_key, t=thread, w=worker, gen=gen_at_submit):
            # Discard if a reinit happened while this thread was running
            if gen == self._reinit_gen:
                self._remember_cache(key, data)
            t.quit()
            t.finished.connect(t.deleteLater)
            w.deleteLater()
            if gen == self._reinit_gen and not self._loading:
                QTimer.singleShot(300, lambda: self._start_next_prefetch(canonical_key))

        def _on_fail(msg, t=thread, w=worker):
            t.quit()
            t.finished.connect(t.deleteLater)
            w.deleteLater()

        worker.finished.connect(_on_done)
        worker.failed.connect(_on_fail)
        thread.start()

    def _on_load_failed(self, message: str):
        self._loading = False
        self._progress_bar.setVisible(False)
        self._set_status(f"Error: {message}", _C_WARN)
        self._set_nav_enabled(True)
        self._pending_load_index = None

    def _apply_loaded_data(self, data: dict):
        raw = data['raw']
        cort = data['cort']
        sub = data['sub']
        self.raw_data = raw
        self.cortical_data = cort
        self.cortical_roi_data = data['cort_roi']
        self.cortical_cube_data = data['cort_cube']
        self.cube_mask_data = data['cube_mask']
        self.subcortical_data = sub
        self._reoriented_affine = data['reoriented_affine']
        self._zooms = data['zooms']
        opacity = self.contrast_dlg.seg_opacity_spin.value()
        sc = data['zooms']
        self._progress(99, 'Adding layers…')
        raw_limits = tuple(self.raw_layer.contrast_limits) if self.raw_layer is not None else (DEFAULT_LOW, DEFAULT_HIGH)
        cort_limits = tuple(self.cortical_layer.contrast_limits) if self.cortical_layer is not None else (DEFAULT_LOW, DEFAULT_HIGH)
        sub_limits = tuple(self.subcortical_layer.contrast_limits) if self.subcortical_layer is not None else (DEFAULT_LOW, DEFAULT_HIGH)
        self._loaded_cortical_seg = data['cortical_seg']
        self._loaded_subcortical_seg = data['subcortical_seg']
        self.raw_layer = self._update_image_layer('raw_layer', self.raw_canvas, raw, name='raw_qsm', scale=sc, contrast_limits=raw_limits)
        self.seg_cortical_layer = self._update_label_layer('seg_cortical_layer', self.raw_canvas, data['cortical_seg'], name='cortical_labels', scale=sc, opacity=opacity, visible=self.cort_seg_cb.isChecked())
        self.seg_subcortical_layer = self._update_label_layer('seg_subcortical_layer', self.raw_canvas, data['subcortical_seg'], name='subcortical_labels', scale=sc, opacity=opacity, visible=self.sub_seg_cb.isChecked())
        self.cortical_layer = self._update_image_layer('cortical_layer', self.cortical_canvas, cort, name='cortical_qsm', scale=sc, contrast_limits=cort_limits)
        self.cortical_seg_layer = self._update_label_layer('cortical_seg_layer', self.cortical_canvas, data['cortical_seg'], name='cortical_labels', scale=sc, opacity=opacity, visible=self.cort_seg_cb.isChecked() and self._show_seg_in_derived_views)
        self.subcortical_layer = self._update_image_layer('subcortical_layer', self.subcortical_canvas, sub, name='subcortical_qsm', scale=sc, contrast_limits=sub_limits)
        self.subcortical_seg_layer = self._update_label_layer('subcortical_seg_layer', self.subcortical_canvas, data['subcortical_seg'], name='subcortical_labels', scale=sc, opacity=opacity, visible=self.sub_seg_cb.isChecked() and self._show_seg_in_derived_views)
        self._bind_hover_label_overlays()
        for key, path in data.get('generated_paths', {}).items():
            setattr(self.cases[self.case_index], key, path)
        self._apply_orientation(force_mid=True)
        self._fit_all(force=True)
        case = self.cases[self.case_index]
        rec = self.results.get(case.case_id, QCRecord(case_id=case.case_id))
        if SEGMENTATION_TASK == True:
            for w in (self.cortex_combo, self.subcortex_combo,
                    self.cort_label_combo, self.sub_label_combo, self.notes_edit, self.review_flag_cb):
                w.blockSignals(True)
        else:
            for w in (self.cortex_combo, self.subcortex_combo, self.notes_edit, self.review_flag_cb):
                w.blockSignals(True)
        self.cortex_combo.setCurrentText(self._score_to_label(rec.cortex_score))
        self.subcortex_combo.setCurrentText(self._score_to_label(rec.subcortex_score))
        self._set_qc_field_error(self.cortex_combo, False)
        self._set_qc_field_error(self.subcortex_combo, False)
        def _restore_label_combo(combo, val):
            for i in range(combo.count()):
                if combo.itemText(i).startswith(val):
                    combo.setCurrentIndex(i)
                    return
            combo.setCurrentIndex(0)
        if SEGMENTATION_TASK == True:
            _restore_label_combo(self.cort_label_combo, rec.cort_label_ok)
            _restore_label_combo(self.sub_label_combo, rec.sub_label_ok)
        self.notes_edit.setPlainText(rec.notes)
        self.review_flag_cb.setChecked(bool(rec.marked_for_review))
        if SEGMENTATION_TASK == True:
            for w in (self.cortex_combo, self.subcortex_combo,
                        self.cort_label_combo, self.sub_label_combo, self.notes_edit, self.review_flag_cb):
                    w.blockSignals(False)
            else:
                for w in (self.cortex_combo, self.subcortex_combo, self.notes_edit, self.review_flag_cb):
                    w.blockSignals(False)
        saved = case.case_id in self.results
        self.case_label.setText(case.case_id)
        if saved:
            self.status_label.setText("Status:  ✔ saved")
            self.status_label.setStyleSheet(f"color:{_C_SUCCESS}; font-size:{_BASE_PT}pt;")
        else:
            self.status_label.setText("Status:  ✏ unsaved")
            self.status_label.setStyleSheet(f"color:{_C_WARN}; font-size:{_BASE_PT}pt;")
        self._cortical_roi_source_ok = bool(data['cort_roi_ok'])
        self._cortical_cube_source_ok = bool(data['cort_cube_ok'])
        self._subcortical_source_ok = bool(data['sub_ok'])
        self._refresh_source_label()
        nat_str = " → ".join(data['native_axcodes'])
        ck = self._current_canonical_key()
        reor_str = f"  (→ {ck})" if ck != "Native" else ""
        self.orient_label.setText(f"Native: {nat_str}{reor_str}")
        z = data['zooms']
        self.spacing_label.setText(f"Spacing: {z[0]:.3f} × {z[1]:.3f} × {z[2]:.3f} mm")
        self.current_saved = saved
        self._refresh_all_case_list_items()
        self._refresh_case_list_item(case.case_id, rec)
        self.case_list.setCurrentRow(self.case_index)
        self.contrast_dlg.reset()
        self._set_status(f"Loaded  {case.case_id}", _C_SUCCESS)


    def _cleanup_scan_thread(self):
        thread = self._scan_thread
        worker = self._scan_worker
        self._scan_worker = None
        self._scan_thread = None
        if thread is not None:
            thread.quit()
            thread.wait(1000)
            thread.deleteLater()
        if worker is not None:
            worker.deleteLater()

    def start_case_scan(self, root: str):
        self._scan_completed = False
        self._cleanup_scan_thread()
        self._scan_thread = QThread(self)
        self._scan_worker = CaseScanWorker(root=root, file_names=self.file_names)
        self._scan_worker.moveToThread(self._scan_thread)
        self._scan_thread.started.connect(self._scan_worker.run)
        self._scan_worker.case_found.connect(self._on_case_found)
        self._scan_worker.finished.connect(self._on_case_scan_finished)
        self._scan_worker.failed.connect(self._on_case_scan_failed)
        self._scan_worker.finished.connect(lambda *_: self._cleanup_scan_thread())
        self._scan_worker.failed.connect(lambda *_: self._cleanup_scan_thread())
        self._scan_thread.start()
        self._set_status("Scanning case list…", _C_ACCENT)
        # Record the live config so _apply_config_change can diff it later
        if self._current_config is None:
            self._current_config = AppConfig(
                cases_root=root,
                output_csv=self.output_csv,
                file_names=dict(self.file_names),
                save_generated_qsm=self._act_save_gen.isChecked(),
                show_seg_in_derived_views=self._show_seg_in_derived_views,
            )

    def _append_case_list_item(self, case: CasePaths):
        """Insert case into the list in sorted order by case_id."""
        if case.case_id in self._case_items:
            return
        # Find sorted insertion position
        case_ids = [self.cases[i].case_id for i in range(len(self.cases))]
        insert_pos = bisect.bisect_left(case_ids, case.case_id)
        self.cases.insert(insert_pos, case)
        item = QListWidgetItem(case.case_id)
        item.setData(Qt.UserRole, case.case_id)
        self.case_list.insertItem(insert_pos, item)
        self._case_items[case.case_id] = item
        self._refresh_case_list_item(case.case_id)
        # Update case_index if newly inserted before current position
        if insert_pos <= self.case_index and self.case_index > 0:
            self.case_index += 1

    def _on_case_found(self, case: CasePaths):
        if case.case_id in self._case_items:
            return
        self._append_case_list_item(case)
        found = len(self.cases)
        self._set_status(f"Found {found} case{'s' if found != 1 else ''}… scanning continues", _C_ACCENT)
        if found == 1 and not self._is_loading():
            self.case_index = 0
            self.case_list.setCurrentRow(0)
            self.load_case(0)
        else:
            self._set_nav_enabled(not self._is_loading())

    def _on_case_scan_finished(self, total: int):
        self._scan_completed = True
        if total <= 0:
            self._set_status("No valid cases found", _C_WARN)
            QMessageBox.critical(
                self,
                "No valid cases found",
                "No valid cases were found under the selected data folder.\nPlease check the folder path and the six file names in the setup page.",
            )
            self._set_nav_enabled(False)
            return
        self._set_status(f"Case list ready ({total} cases)", _C_SUCCESS)
        self._set_nav_enabled(not self._is_loading())

    def _on_case_scan_failed(self, message: str):
        self._scan_completed = True
        self._set_status(f"Case scan failed: {message}", _C_WARN)
        QMessageBox.critical(self, "Case scan failed", message)
        self._set_nav_enabled(False)

    def closeEvent(self, event):
        try:
            self._wheel_timer.stop()
            self._fit_timer.stop()
            self._cleanup_load_thread()
            self._cleanup_scan_thread()
        finally:
            super().closeEvent(event)

# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # High-DPI support — must be set BEFORE QApplication is created.
    # On Windows with 125 %/150 % display scaling these prevent clipping
    # of menu checkmarks, text, and other fixed-size UI elements.
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps,   True)
    # OpenGL setup — must come BEFORE QApplication is created
    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
    if not _IS_MACOS:
        # UseDesktopOpenGL is a Windows/Linux hint; on macOS it can cause issues
        QApplication.setAttribute(Qt.AA_UseDesktopOpenGL)

    app = QApplication.instance() or QApplication(sys.argv)

    # Platform-appropriate base font
    if _IS_MACOS:
        app.setFont(QFont("Helvetica Neue", _BASE_PT))
    else:
        app.setFont(QFont("Segoe UI", _BASE_PT))
    app.setStyleSheet(_GLOBAL_CSS)
    app.setWindowIcon(_make_app_icon())
    setup = StartupConfigDialog(
        defaults_root=CASES_ROOT,
        defaults_output_csv=OUTPUT_CSV,
        defaults_file_names=FILE_NAMES,
    )
    if setup.exec_() != QDialog.Accepted or setup.config is None:
        return
    config = setup.config
    win = ReviewerMainWindow(
        cases=[],
        output_csv=config.output_csv,
        file_names=config.file_names,
        save_generated_qsm=config.save_generated_qsm,
        show_seg_in_derived_views=config.show_seg_in_derived_views,
    )
    win.show()
    QTimer.singleShot(0, lambda: win.start_case_scan(config.cases_root))
    sys.exit(app.exec_())
if __name__ == "__main__":
    main()
