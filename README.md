# QSM QC Reviewer

A desktop application for **scan-level quality control (QC)** of QSM data, designed for rapid visual review of raw QSM, cortical QSM, and subcortical QSM in a single interface.

This app is built for workflows where reviewers need to:
- inspect multiple QSM-derived views side by side,
- scroll slices quickly,
- overlay segmentation/labels,
- assign motion QC scores,
- record notes,
- flag cases for later review,
- and save structured QC results to CSV.

It also supports on-the-fly generation of derived QSM views and packaging as a standalone Windows executable for distribution to other reviewers.

---

## Main Features

### 1. Three-view QC layout
The reviewer UI displays three coordinated image views:

- **Raw QSM + Segmentation**
- **Cortical QSM**
- **Subcortical QSM**

These are arranged in a multi-panel layout with adjustable splitters so panel sizes can be resized during review.

### 2. Path setup page before entering the reviewer
When the app starts, it opens a **Path Setup** window before the main labeling UI.

You can configure:

- **Data folder**
- **CSV file name**
- **Input file names** for all required NIfTI files
- Whether to **save generated QSM files**
- Whether to **show segmentation in cortical/subcortical QSM views**

This allows the app to be reused on different datasets or directory structures without editing the code.

### 3. Case loading and case list navigation
The app scans the configured data folder for cases and builds a **case list**.

You can:
- move to the **next** or **previous** case,
- click any **case ID** in the case list to jump directly,
- reload existing saved annotations from CSV,
- and visually track review status in the list.

### 4. QC scoring and notes
The right-side QC panel includes structured fields for:

#### Motion QC Scores
- **Cortex**
- **Subcortex**

#### Segmentation Accuracy
- **Cortical**
- **Subcortical**

#### Notes
- Free-text notes for any additional observations

Annotations can be saved to CSV and reloaded automatically when reopening the app.

### 5. Missing-score warning before leaving a case
If you click **Next** or jump to another case while any **Motion QC Score** field is still empty, the app:

- shows a warning dialog,
- highlights the missing QC field(s) in red,
- allows you to **continue anyway** or **cancel**.

If you cancel and then fill in the missing score, the red highlight is cleared automatically.

### 6. Review flagging
A case can be marked for later review using a dedicated **flag** control.

Flagged cases:
- are shown with a **warning marker** in the case list,
- can be unflagged later,
- and are recorded in the output CSV using a dedicated column.

The case list also marks completed cases with a check mark when the required motion QC fields are filled.

### 7. Overlay controls
The app supports segmentation and label overlays for multiple views.

You can toggle overlays such as:
- cortical labels,
- subcortical labels,
- and segmentation overlays in derived QSM views.

There is also an option to enable segmentation overlays not only on the raw/initial QSM view, but also on:
- **Cortical QSM**
- **Subcortical QSM**

### 8. Cortical QSM display modes
The cortical view supports multiple display modes:

- **ROI only**
- **Expanded / All regions outside subcortical**

The mode selector is shown directly in the **Cortical QSM panel header**.

By default, the cortical panel uses the **expanded** version.

### 9. Generated derived QSM files
The app supports automatic generation of derived QSM images, including:

- **Cortical QSM**
- **Cortical QSM (expanded)**
- **Subcortical QSM**

The expanded cortical QSM combines:
- the **ROI-based cortical mask**, and
- the **outside-subcortical-cube mask**

so both relevant cortical ROI regions and non-subcortical outer regions remain visible.

If **Save generated QSM files** is enabled, the generated files are saved locally for reuse.

### 10. Zoom controls per view
Each image view has its own **Zoom** control located in the panel header.

Supported zoom options include:
- Fit window
- 25%
- 50%
- 75%
- 100%
- 125%
- 150%
- 200%
- 300%

The zoom control also accepts **custom input**, such as:
- `180%`
- `180`
- `1.8`

Each panel maintains its own zoom state independently.

### 11. Mouse and keyboard interaction
The app supports efficient image navigation:

- **Mouse wheel**: scroll slices
- **Ctrl + mouse wheel**: change zoom level for the active panel only
- independent zoom and slice navigation per view
- optional synchronized slice behavior where supported by the panel controls

### 12. Existing CSV restoration
If the configured CSV already contains saved labels when the app is launched, the reviewer state is restored automatically for matching cases, including:

- Motion QC scores
- Segmentation Accuracy
- Notes
- Review flag

### 13. Performance optimizations
The app includes several performance-oriented improvements for large review sessions:

- background case loading
- recent-case caching
- smoother wheel-based slice navigation
- reduced unnecessary redraws / layer rebuilds
- improved responsiveness when switching cases

---

## Expected Input Files

The app is configured around a set of NIfTI files per case.

Default file keys:

```python
FILE_NAMES = {
    "raw_qsm":           "QSM_TOTAL_mcpc3Ds_chi_SFCR+0_Avg_wGDC.nii.gz",
    "segmentation":      "T1_SynthSeg_relabeled_corrected_to_SWI.nii.gz",
    "subcortical_label": "QSM_TOTAL_mcpc3Ds_chi_SFCR_Avg_wGDC_labels_a2.nii.gz",
    "cortical_qsm":      "QSM_TOTAL_mcpc3Ds_chi_SFCR_Avg_wGDC_cortical_dilated.nii.gz",
    "cortical_qsm_cube": "QSM_TOTAL_mcpc3Ds_chi_SFCR_Avg_wGDC_cortical_expanded.nii.gz",
    "subcortical_qsm":   "QSM_TOTAL_mcpc3Ds_chi_SFCR_Avg_wGDC_subcortical_expanded.nii.gz",
}
```

These names can be changed in the Path Setup page before entering the reviewer UI.

---

## Output CSV

The CSV stores the review results for each case, including fields such as:

- case ID
- cortical motion QC
- subcortical motion QC
- cortical segmentation accuracy
- subcortical segmentation accuracy
- notes
- review flag

If a CSV already exists, the app loads existing rows and restores the saved state for corresponding cases.

---

## Typical Review Workflow

1. Launch the app.
2. Configure paths and file names in the **Path Setup** page.
3. Enter the main reviewer UI.
4. Review:
   - Raw QSM + segmentation
   - Cortical QSM
   - Subcortical QSM
5. Adjust zoom, overlays, and panel sizes as needed.
6. Assign motion QC and segmentation accuracy scores.
7. Add notes if needed.
8. Flag cases that require later re-review.
9. Save the case.
10. Move to the next case or jump through the case list.

---

## Build From Source

### Requirements
A typical Windows development environment will need:

- Python 3.10 recommended
- PyQt5
- napari
- numpy
- scipy
- nibabel
- pandas
- other dependencies used by the application

If you are using a conda environment, activate it first.

### Install dependencies
Example:

```bash
pip install pyqt5 napari numpy scipy nibabel pandas pyinstaller
```

Depending on your environment, you may also need additional napari/vispy-related packages already used by your current working build.

### Run from source
From the project directory:

```bash
python QCreviewer.py
```

Replace `QCreviewer.py` with your actual entry file name if different.

---

## Build a Windows Executable (.exe)

This project is intended to be packaged with **PyInstaller**.

### Included packaging files
The repository can include:

- `QSM_QC_Reviewer.spec`
- `build_qsm_qc_reviewer.bat`
- `app.ico`
- `logo.png`

### Build using the batch script
On Windows, from the project folder:

```powershell
.\build_qsm_qc_reviewer.bat
```

This will generate a packaged application under:

```text
dist\QCreviewer\
```

or a similarly named output folder, depending on your PyInstaller configuration.

### Important notes
- The packaged app should be distributed as the **entire dist application folder**, not just the `.exe`.
- Do **not** commit `build/` or `dist/` to GitHub.
- Use a `.gitignore` entry such as:

```gitignore
build/
dist/
__pycache__/
*.pyc
```

---

## Release Download Instructions

The recommended distribution method is through **GitHub Releases**.

### For maintainers
1. Build the Windows app with PyInstaller.
2. Compress the entire packaged app folder inside `dist/`, for example:

```text
QCreviewer-win64-v1.0.0.zip
```

3. Go to the repository on GitHub.
4. Open **Releases**.
5. Click **Draft a new release**.
6. Create or select a version tag, for example `v1.0.0`.
7. Add a title, such as:

```text
QCreviewer v1.0.0
```

8. Upload the zip file as a release asset.
9. Publish the release.

### For users
1. Open the repository’s **Releases** page.
2. Download the latest Windows zip package.
3. Extract the zip file.
4. Open the extracted folder.
5. Run the executable (`QCreviewer.exe` or the packaged app executable name).

### Distribution recommendation
Always distribute the **zipped packaged folder** from the Release page rather than a single exe copied out by itself.

---

## Repository Structure Example

```text
QCreviewer/
├─ QCreviewer.py
├─ QSM_QC_Reviewer.spec
├─ build_qsm_qc_reviewer.bat
├─ app.ico
├─ logo.png
├─ README.md
└─ .gitignore
```

Locally generated packaging artifacts:

```text
build/
dist/
```

should remain local and should not be pushed to the repository.

---

## Notes

- This app is designed for **QSM QC review workflows** where reviewers need both structured scoring and rapid visual inspection.
- If distributing to other reviewers, test the packaged build on another Windows machine before release.
- When sharing builds, prefer **GitHub Releases assets** instead of storing large binaries in the repository itself.
