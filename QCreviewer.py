"""
QSM QC Reviewer v11
=====================
1. Added background case loading with QThread to keep the UI responsive during NIfTI reading, mask generation, and derived QSM creation.
2. Added a recent-case in-memory cache so switching back to recently opened cases is much faster.
3. Optimized mouse wheel slice navigation with lightweight event batching/throttling for smoother scrolling.
4. Kept Ctrl + mouse wheel zoom behavior while separating it cleanly from normal slice scrolling.
5. Reduced unnecessary Napari layer destruction/recreation by reusing layers and updating data/visibility when possible.
6. Reduced redundant fit/camera updates to improve responsiveness during case loading, splitter resizing, and view changes.
7. Improved performance of cortical mode switching by avoiding full view rebuilds where possible.
8. Preserved all current reviewer features, including expanded cortical QSM, segmentation overlays, case flags, zoom presets/custom zoom, and startup settings.
9. Fixed a potential ROI variable inconsistency in the expanded cortical QSM generation path.
"""
import os
import sys
from dataclasses import dataclass, asdict, fields
from typing import Callable, Dict, List, Optional, Tuple
from collections import OrderedDict
import nibabel as nib
import nibabel.orientations as nibo
import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation
from qtpy.QtCore import QObject, QEvent, QTimer, Qt, QSize, Signal, QThread
from qtpy.QtGui import QFont, QColor, QPainter, QPainterPath, QPen, QPixmap, QIcon
from qtpy.QtWidgets import (
    QAction, QApplication, QCheckBox, QComboBox,
    QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame,
    QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
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
    "cortical_qsm_cube": "QSM_TOTAL_mcpc3Ds_chi_SFCR_Avg_wGDC_cortical_expanded.nii.gz",
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
QMainWindow, QWidget {{
    background: {_C_BG}; color: {_C_TEXT};
    font-family: Segoe UI, Arial, sans-serif; font-size: 10pt;
}}
QToolBar {{
    background: {_C_HEADER}; border-bottom: 1px solid {_C_BORDER};
    spacing: 8px; padding: 4px 10px;
}}
QToolBar QLabel  {{ color: {_C_DIM}; font-size: 10pt; }}
QToolBar QComboBox {{
    background: {_C_PANEL}; color: {_C_TEXT}; border: 1px solid {_C_BORDER};
    border-radius: 4px; padding: 3px 10px; font-size: 10pt; min-width: 140px;
}}
QToolBar QComboBox::drop-down {{ border: none; width: 20px; }}
QToolBar QCheckBox {{ color: {_C_TEXT}; font-size: 10pt; }}
QSplitter::handle           {{ background: {_C_BORDER}; }}
QSplitter::handle:horizontal {{ width: 5px; }}
QSplitter::handle:vertical   {{ height: 5px; }}
QGroupBox {{
    border: 1px solid {_C_BORDER}; border-radius: 5px;
    margin-top: 22px; padding-top: 8px;
    font-size: 10pt; font-weight: 600; color: {_C_ACCENT};
}}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; }}
QLabel           {{ font-size: 10pt; }}
QComboBox {{
    background: {_C_PANEL}; color: {_C_TEXT}; border: 1px solid {_C_BORDER};
    border-radius: 4px; padding: 4px 10px; font-size: 10pt;
}}
QComboBox QAbstractItemView {{
    background: {_C_PANEL}; color: {_C_TEXT};
    selection-background-color: {_C_ACCENT}; font-size: 10pt;
}}
QListWidget {{
    background: {_C_PANEL}; color: {_C_TEXT}; border: 1px solid {_C_BORDER};
    border-radius: 4px; font-size: 10pt;
}}
QListWidget::item           {{ padding: 3px 6px; }}
QListWidget::item:selected  {{ background: {_C_ACCENT}; color: white; }}
QListWidget::item:hover     {{ background: {_C_HEADER}; }}
QTextEdit {{
    background: {_C_PANEL}; color: {_C_TEXT}; border: 1px solid {_C_BORDER};
    border-radius: 4px; font-size: 10pt; padding: 5px;
}}
QPushButton {{
    background: {_C_PANEL}; color: {_C_TEXT}; border: 1px solid {_C_BORDER};
    border-radius: 4px; padding: 6px 16px; font-size: 10pt;
}}
QPushButton:hover   {{ background: {_C_HEADER}; border-color: {_C_ACCENT}; }}
QPushButton:pressed {{ background: {_C_ACCENT}; color: white; }}
QPushButton:disabled {{ color: {_C_DIM}; border-color: {_C_BORDER}; }}
QCheckBox           {{ color: {_C_TEXT}; font-size: 10pt; spacing: 6px; }}
QDoubleSpinBox {{
    background: {_C_PANEL}; color: {_C_TEXT}; border: 1px solid {_C_BORDER};
    border-radius: 4px; padding: 3px 8px; font-size: 10pt;
}}
QScrollArea        {{ border: none; background: transparent; }}
QScrollBar:vertical {{
    background: {_C_BG}; width: 8px; border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background: {_C_BORDER}; border-radius: 4px; min-height: 24px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}
QStatusBar         {{ background: {_C_HEADER}; color: {_C_DIM}; font-size: 9pt; }}
QProgressBar {{
    background: {_C_PANEL}; border: 1px solid {_C_BORDER}; border-radius: 3px;
    text-align: center; color: {_C_TEXT}; font-size: 9pt; max-height: 16px;
}}
QProgressBar::chunk {{ background: {_C_ACCENT}; border-radius: 2px; }}
QMenuBar            {{ background: {_C_HEADER}; color: {_C_TEXT}; font-size: 10pt; }}
QMenuBar::item:selected {{ background: {_C_ACCENT}; }}
QMenu               {{ background: {_C_PANEL}; color: {_C_TEXT};
                       border: 1px solid {_C_BORDER}; font-size: 10pt; }}
QMenu::item:selected {{ background: {_C_ACCENT}; }}
"""
_SAVE_BTN_CSS = f"""
QPushButton {{
    background: {_C_ACCENT}; color: white; border: none;
    border-radius: 4px; padding: 9px 18px; font-size: 11pt; font-weight: 600;
}}
QPushButton:hover   {{ background: #5aabff; }}
QPushButton:pressed {{ background: #3a8eef; }}
QPushButton:disabled {{ background: {_C_BORDER}; color: {_C_DIM}; }}
"""
_NAV_BTN_CSS = f"""
QPushButton {{
    background: {_C_PANEL}; color: {_C_TEXT}; border: 1px solid {_C_BORDER};
    border-radius: 4px; padding: 7px 20px; font-size: 10pt;
}}
QPushButton:hover   {{ background: {_C_HEADER}; border-color: {_C_ACCENT}; color: {_C_ACCENT}; }}
QPushButton:disabled {{ color: {_C_DIM}; border-color: {_C_BORDER}; }}
"""
_QC_ERROR_STYLE = f"""
QComboBox {{
    background: {_C_PANEL}; color: {_C_TEXT};
    border: 2px solid #d65c5c; border-radius: 4px; padding: 4px 10px; font-size: 10pt;
}}
QComboBox QAbstractItemView {{
    background: {_C_PANEL}; color: {_C_TEXT};
    selection-background-color: {_C_ACCENT}; font-size: 10pt;
}}
"""
ZOOM_PRESETS = [25, 50, 75, 100, 125, 150, 200, 300]
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
class StartupConfigDialog(QDialog):
    def __init__(self, parent=None, defaults_root: str = CASES_ROOT,
                 defaults_output_csv: str = OUTPUT_CSV,
                 defaults_file_names: Optional[Dict[str, str]] = None):
        super().__init__(parent)
        self.setWindowTitle("QSM QC Reviewer - Path Setup")
        self.setWindowIcon(_make_app_icon())
        self.setMinimumWidth(760)
        self.setModal(True)
        self.setStyleSheet(_GLOBAL_CSS)
        self._defaults_file_names = dict(defaults_file_names or FILE_NAMES)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(12)
        title_row = QHBoxLayout()
        logo = QLabel()
        logo.setPixmap(_make_app_icon().pixmap(28, 28))
        title = QLabel("Launch Settings")
        title.setStyleSheet(f"font-size: 15pt; font-weight: 700; color: {_C_ACCENT};")
        subtitle = QLabel("Set the data folder, CSV file name, display options, and the six file names before entering the labeling UI.")
        subtitle.setStyleSheet(f"color: {_C_DIM};")
        title_col = QVBoxLayout()
        title_col.addWidget(title)
        title_col.addWidget(subtitle)
        title_row.addLayout(title_col, 1)
        lay.addLayout(title_row)
        box = QGroupBox("Paths")
        form = QFormLayout(box)
        form.setSpacing(10)
        self.root_edit = QLineEdit(defaults_root)
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse_root)
        root_row = QHBoxLayout()
        root_row.addWidget(self.root_edit, 1)
        root_row.addWidget(browse_btn)
        form.addRow("Data folder:", self._wrap_layout(root_row))
        csv_default_name = os.path.basename(defaults_output_csv) if defaults_output_csv else "qc_results.csv"
        self.csv_edit = QLineEdit(csv_default_name)
        form.addRow("CSV file name:", self.csv_edit)
        lay.addWidget(box)
        files_box = QGroupBox("Data file names")
        files_grid = QGridLayout(files_box)
        files_grid.setHorizontalSpacing(10)
        files_grid.setVerticalSpacing(8)
        labels = [
            ("raw_qsm", "Raw QSM"),
            ("segmentation", "Segmentation"),
            ("subcortical_label", "Subcortical label"),
            ("cortical_qsm", "Cortical QSM"),
            ("cortical_qsm_cube", "Cortical QSM (expanded)"),
            ("subcortical_qsm", "Subcortical QSM"),
        ]
        self.file_edits: Dict[str, QLineEdit] = {}
        for row, (key, label_text) in enumerate(labels):
            lbl = QLabel(label_text + ":")
            edit = QLineEdit(self._defaults_file_names.get(key, ""))
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
            cortical_qsm_cube=paths["cortical_qsm_cube"]
                if os.path.exists(paths["cortical_qsm_cube"]) else None,
            subcortical_qsm=paths["subcortical_qsm"]
                if os.path.exists(paths["subcortical_qsm"]) else None,
        ))
    return cases
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
_DIR_CSS  = (f"font-size:9pt; color:{_C_DIM}; background:{_C_DIR_BAR};"
             " padding:2px 8px; letter-spacing:1px;")
_SYNC_CSS = (f"QCheckBox {{ color:#aabbcc; font-size:9pt; }}"
             f"QCheckBox::indicator {{ width:13px; height:13px; }}")
class ImageCanvas(QWidget):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._zoom_mode = "fit"   # "fit" or float multiplier relative to fit
        self._updating_zoom_ui = False
        # title bar
        hdr = QWidget()
        hdr.setStyleSheet(f"background:{_C_HEADER};")
        hdr.setFixedHeight(32)
        hl = QHBoxLayout(hdr)
        hl.setContentsMargins(8, 0, 8, 0); hl.setSpacing(6)
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet(f"font-weight:600; font-size:10pt; color:{_C_TEXT};")
        self.title_right_layout = QHBoxLayout()
        self.title_right_layout.setContentsMargins(0, 0, 0, 0)
        self.title_right_layout.setSpacing(6)
        zoom_lbl = QLabel("Zoom")
        zoom_lbl.setStyleSheet(f"color:{_C_DIM}; font-size:9pt;")
        self.zoom_combo = QComboBox()
        self.zoom_combo.setEditable(True)
        self.zoom_combo.setInsertPolicy(QComboBox.NoInsert)
        self.zoom_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.zoom_combo.setMinimumWidth(132)
        self.zoom_combo.setMaximumWidth(168)
        self.zoom_combo.addItem("Fit window")
        for pct in ZOOM_PRESETS:
            self.zoom_combo.addItem(f"{pct}%")
        self.zoom_combo.setCurrentText("Fit window")
        self.zoom_combo.lineEdit().editingFinished.connect(self._on_zoom_combo_edited)
        self.zoom_combo.currentTextChanged.connect(self._on_zoom_combo_changed)
        self.sync_cb = QCheckBox("🔗 Sync")
        self.sync_cb.setChecked(True)
        self.sync_cb.setStyleSheet(_SYNC_CSS)
        self.sync_cb.setToolTip("Uncheck to scroll this view independently")
        hl.addWidget(self.title_label)
        hl.addLayout(self.title_right_layout)
        hl.addStretch(1)
        hl.addWidget(zoom_lbl)
        hl.addWidget(self.zoom_combo)
        hl.addWidget(self.sync_cb)
        # direction bar
        dir_bar = QWidget()
        dir_bar.setStyleSheet(f"background:{_C_DIR_BAR};")
        dir_bar.setFixedHeight(22)
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
        self._last_fit_signature = None
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
    def set_title_right_widget(self, widget: Optional[QWidget]):
        while self.title_right_layout.count():
            item = self.title_right_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
        if widget is not None:
            self.title_right_layout.addWidget(widget)

    @property
    def sync_enabled(self): return self.sync_cb.isChecked()
    @staticmethod
    def _parse_zoom_text(text: str):
        s = (text or "").strip()
        if not s:
            return None
        s_low = s.lower()
        if s_low in {"fit", "fit window", "fitwindow", "window", "auto"}:
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
            return "Fit window"
        try:
            pct = float(mode) * 100.0
            if abs(pct - round(pct)) < 1e-6:
                return f"{int(round(pct))}%"
            return f"{pct:.1f}%"
        except Exception:
            return "Fit window"
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
        self._lbl_left.setText(f"← {left}")
        self._lbl_right.setText(f"{right} →")
        self._lbl_mid.setText(f"↑ {top}   ↓ {bottom}")
    def install_wheel_filter(self, on_wheel):
        target = self._canvas_native or self.qt_viewer
        self._wheel_filter = WheelScrollFilter(on_wheel, parent=self)
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
        except Exception:
            self.viewer.reset_view()
# ─────────────────────────────────────────────────────────────────────────────
# Main window
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
        self.resize(1900, 1020)
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
        self._cache_limit = 3
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
        # Layers
        self.raw_layer = self.seg_cortical_layer = None
        self.seg_subcortical_layer = self.cortical_layer = self.subcortical_layer = None
        self.cortical_seg_layer = self.subcortical_seg_layer = None
        # Canvases
        self.raw_canvas         = ImageCanvas("Raw QSM + Segmentation")
        self.cortical_canvas    = ImageCanvas("Cortical QSM")
        self.subcortical_canvas = ImageCanvas("Subcortical QSM")
        self.cortical_mode_combo = QComboBox()
        self.cortical_mode_combo.addItems(["ROI only", "All regions outside subcortical"])
        self.cortical_mode_combo.setCurrentText("All regions outside subcortical")
        self.cortical_mode_combo.setMinimumWidth(220)
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
        self.load_case(0)
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
    # ── menu ─────────────────────────────────────────────────────────────────
    def _build_menu(self):
        vm = self.menuBar().addMenu("View")
        act = QAction("Contrast && Overlay…", self)
        act.setShortcut("Ctrl+L")
        act.triggered.connect(self.contrast_dlg.show)
        vm.addAction(act)
        vm.addSeparator()
        self._act_show_seg_derived = QAction(
            "Show segmentation in cortical/subcortical QSM", self,
            checkable=True, checked=self._show_seg_in_derived_views)
        self._act_show_seg_derived.toggled.connect(self._on_show_seg_derived_toggled)
        vm.addAction(self._act_show_seg_derived)
        tm = self.menuBar().addMenu("Tools")
        self._act_save_gen = QAction(
            "Save generated QSM files", self,
            checkable=True, checked=self._initial_save_generated_qsm)
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
        # connections
        self.canonical_combo.currentTextChanged.connect(
            lambda _: self.load_case(self.case_index))
        self.orient_combo.currentTextChanged.connect(
            lambda _: self._apply_orientation())
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
        # Connect splitter moves → debounced fit + keep top/bot aligned
        for spl in (self._v_splitter, self._top_splitter, self._bot_splitter,
                    self._main_splitter):
            spl.splitterMoved.connect(lambda *_: self._fit_timer.start())
        # Deferred size setup
        QTimer.singleShot(0, self._init_splitter_sizes)
    def _init_splitter_sizes(self):
        w = self._main_splitter.width()
        if w <= 10:
            return
        left_w = int(w * 0.70)
        self._main_splitter.setSizes([left_w, w - left_w])
        h = self._v_splitter.height()
        if h > 10:
            self._v_splitter.setSizes([h // 2, h // 2])
        top_w = max(1, self._top_splitter.width())
        bot_w = max(1, self._bot_splitter.width())
        self._top_splitter.setSizes([top_w // 2, top_w - (top_w // 2)])
        self._bot_splitter.setSizes([bot_w // 2, bot_w - (bot_w // 2)])
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
            f" font-size:13pt; color:{_C_ACCENT};")
        self.status_label = QLabel("Status: unsaved")
        self.status_label.setStyleSheet(f"font-size:10pt;")
        self.source_label = QLabel("Sources: -")
        self.source_label.setStyleSheet(f"color:{_C_DIM}; font-size:9pt;")
        self.orient_label = QLabel("Native: -")
        self.orient_label.setStyleSheet(f"color:{_C_DIM}; font-size:9pt;")
        self.spacing_label = QLabel("Spacing: -")
        self.spacing_label.setStyleSheet(f"color:{_C_DIM}; font-size:9pt;")
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
            "Wheel    scroll slices (respects Sync)\n"
            "Ctrl+Wheel zoom current view\n"
            "Ctrl+L   Contrast & Overlay\n"
            "C        Toggle cortical overlay\n"
            "S        Toggle subcortical overlay\n"
            "N / P    Next / Prev case (auto-save)\n"
            "Ctrl+S   Save")
        hotkey_lbl.setStyleSheet(
            f"font-family: Consolas, monospace; font-size:9pt; color:{_C_DIM};")
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
        panel.setMinimumWidth(260)
        cl = QVBoxLayout(panel)
        cl.setContentsMargins(12, 12, 12, 12)
        cl.setSpacing(10)
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
        cl.addWidget(sg)
        # Label accuracy
        LABEL_ACCURACY = ["0 – Good", "1 – Bad"]
        lag = QGroupBox("Segmentation Accuracy  (optional)")
        laf = QFormLayout(lag); laf.setSpacing(8)
        self.cort_label_combo = QComboBox()
        self.sub_label_combo  = QComboBox()
        for cb in (self.cort_label_combo, self.sub_label_combo):
            cb.addItem("")
            cb.addItems(LABEL_ACCURACY)
        laf.addRow("Cortical:",    self.cort_label_combo)
        laf.addRow("Subcortical:", self.sub_label_combo)
        cl.addWidget(lag)
        # Notes
        ng = QGroupBox("Notes")
        nl = QVBoxLayout(ng)
        self.notes_edit = QTextEdit()
        self.notes_edit.setPlaceholderText("Optional notes…")
        self.notes_edit.setMinimumHeight(80)
        self.notes_edit.setMaximumHeight(140)
        nl.addWidget(self.notes_edit)
        cl.addWidget(ng)
        # Review flag
        fg = QGroupBox("Review")
        fl = QVBoxLayout(fg)
        self.review_flag_cb = QCheckBox("Flag this case for later review")
        fl.addWidget(self.review_flag_cb)
        cl.addWidget(fg)
        # Save
        self._btn_save = QPushButton("💾   Save   (Ctrl+S)")
        self._btn_save.setStyleSheet(_SAVE_BTN_CSS)
        self._btn_save.clicked.connect(self.save_current_case)
        cl.addWidget(self._btn_save)
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
        cl.addWidget(nav)
        # Case list
        lg = QGroupBox("Case List")
        ll2 = QVBoxLayout(lg)
        self.case_list = QListWidget()
        self._case_items: Dict[str, QListWidgetItem] = {}
        for c in self.cases:
            item = QListWidgetItem(c.case_id)
            item.setData(Qt.UserRole, c.case_id)
            self.case_list.addItem(item)
            self._case_items[c.case_id] = item
        self.case_list.setMinimumHeight(180)
        self.case_list.itemClicked.connect(self.on_case_clicked)
        ll2.addWidget(self.case_list)
        cl.addWidget(lg, 1)
        self.cortex_combo.currentTextChanged.connect(self._mark_unsaved)
        self.subcortex_combo.currentTextChanged.connect(self._mark_unsaved)
        self.cortex_combo.currentTextChanged.connect(lambda _: self._clear_qc_error_if_filled(self.cortex_combo))
        self.subcortex_combo.currentTextChanged.connect(lambda _: self._clear_qc_error_if_filled(self.subcortex_combo))
        self.cort_label_combo.currentTextChanged.connect(self._mark_unsaved)
        self.sub_label_combo.currentTextChanged.connect(self._mark_unsaved)
        self.notes_edit.textChanged.connect(self._mark_unsaved)
        self.review_flag_cb.toggled.connect(self._mark_unsaved)
        self.cortex_combo.currentTextChanged.connect(lambda _: self._refresh_current_case_list_item())
        self.subcortex_combo.currentTextChanged.connect(lambda _: self._refresh_current_case_list_item())
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
        orient_name   = self.orient_combo.currentText()
        canonical_key = self.canonical_combo.currentText()
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
        for canvas, data in self._canvas_data_pairs():
            if data is None: continue
            canvas.set_view(self._dims_order, self._scroll_axis, data.shape, fv, fh)
            canvas.lock_scroll_mode()
            canvas.set_direction_labels(left, right, top, bottom)
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
        self.status_label.setStyleSheet(f"color:{_C_WARN}; font-size:10pt;")
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
            cort_label_ok=_la(self.cort_label_combo),
            sub_label_ok=_la(self.sub_label_combo),
            notes=self.notes_edit.toPlainText().strip(),
            marked_for_review=("1" if self.review_flag_cb.isChecked() else ""))
    @staticmethod
    def _record_is_completed(rec: Optional[QCRecord]) -> bool:
        return bool(rec and rec.cortex_score and rec.subcortex_score)
    @staticmethod
    def _record_is_flagged(rec: Optional[QCRecord]) -> bool:
        return bool(rec and str(rec.marked_for_review).strip() in {"1", "true", "True"})
    def _case_item_display_text(self, case_id: str, rec: Optional[QCRecord]) -> str:
        suffix = ""
        if self._record_is_completed(rec):
            suffix += " ✓"
        if self._record_is_flagged(rec):
            suffix += " ⚠️"
        return f"{case_id}{suffix}"
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
        rec = self._collect_record()
        self.results[rec.case_id] = rec
        write_results(self.output_csv, self.results)
        self._refresh_case_list_item(rec.case_id, rec)
        self.current_saved = True
        self.status_label.setText("Status:  ✔ saved")
        self.status_label.setStyleSheet(f"color:{_C_SUCCESS}; font-size:10pt;")
    def next_case(self):
        if self.case_index >= len(self.cases) - 1 or self._is_loading(): return
        if not self._validate_motion_scores_before_leave():
            return
        self.save_current_case()
        self.load_case(self.case_index + 1)
    def prev_case(self):
        if self.case_index <= 0 or self._is_loading(): return
        self.save_current_case()
        self.load_case(self.case_index - 1)
    def on_case_clicked(self, item):
        if self._is_loading(): return
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
        self.btn_prev.setEnabled(enabled and self.case_index > 0)
        self.btn_next.setEnabled(enabled and self.case_index < len(self.cases) - 1)
        self._btn_save.setEnabled(enabled)
        self.case_list.setEnabled(enabled)
        self.canonical_combo.setEnabled(enabled)
        self.orient_combo.setEnabled(enabled)
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
        canonical_key = self.canonical_combo.currentText() if hasattr(self, 'canonical_combo') else DEFAULT_CANONICAL
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
        for key, path in data.get('generated_paths', {}).items():
            setattr(self.cases[self.case_index], key, path)
        self._apply_orientation(force_mid=True)
        self._fit_all(force=True)
        case = self.cases[self.case_index]
        rec = self.results.get(case.case_id, QCRecord(case_id=case.case_id))
        for w in (self.cortex_combo, self.subcortex_combo,
                  self.cort_label_combo, self.sub_label_combo, self.notes_edit, self.review_flag_cb):
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
        _restore_label_combo(self.cort_label_combo, rec.cort_label_ok)
        _restore_label_combo(self.sub_label_combo, rec.sub_label_ok)
        self.notes_edit.setPlainText(rec.notes)
        self.review_flag_cb.setChecked(bool(rec.marked_for_review))
        for w in (self.cortex_combo, self.subcortex_combo,
                  self.cort_label_combo, self.sub_label_combo, self.notes_edit, self.review_flag_cb):
            w.blockSignals(False)
        saved = case.case_id in self.results
        self.case_label.setText(case.case_id)
        if saved:
            self.status_label.setText("Status:  ✔ saved")
            self.status_label.setStyleSheet(f"color:{_C_SUCCESS}; font-size:10pt;")
        else:
            self.status_label.setText("Status:  ✏ unsaved")
            self.status_label.setStyleSheet(f"color:{_C_WARN}; font-size:10pt;")
        self._cortical_roi_source_ok = bool(data['cort_roi_ok'])
        self._cortical_cube_source_ok = bool(data['cort_cube_ok'])
        self._subcortical_source_ok = bool(data['sub_ok'])
        self._refresh_source_label()
        nat_str = " → ".join(data['native_axcodes'])
        ck = self.canonical_combo.currentText()
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

    def closeEvent(self, event):
        try:
            self._wheel_timer.stop()
            self._fit_timer.stop()
            self._cleanup_load_thread()
        finally:
            super().closeEvent(event)

# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # Must come BEFORE QApplication is created
    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
    QApplication.setAttribute(Qt.AA_UseDesktopOpenGL)
    app = QApplication.instance() or QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
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
    cases = find_cases(config.cases_root, config.file_names)
    if not cases:
        QMessageBox.critical(
            None,
            "No valid cases found",
            "No valid cases were found under the selected data folder.\n"
            "Please check the folder path and the six file names in the setup page.",
        )
        return
    win = ReviewerMainWindow(
        cases=cases,
        output_csv=config.output_csv,
        file_names=config.file_names,
        save_generated_qsm=config.save_generated_qsm,
        show_seg_in_derived_views=config.show_seg_in_derived_views,
    )
    win.show()
    sys.exit(app.exec_())
if __name__ == "__main__":
    main()
