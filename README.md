# ieeg-wm: iEEG Preprocessing and Decoding for Working-Memory Research

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![MNE](https://img.shields.io/badge/MNE-1.6+-orange.svg)](https://mne.tools)

Python implementation supporting the analyses presented in the thesis:

> **"Preprocessing and decoding neural traces of serial biases in working memory
> from intracranial electrophysiological recordings in humans"**  
> — Alberto Hurtado · [LinkedIn](https://www.linkedin.com/in/alberto-hurtado-morell/)

---

## Table of Contents

1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [Installation](#installation)
4. [Quickstart](#quickstart)
5. [Notebooks](#notebooks)
6. [API Reference](#api-reference)
7. [Citation](#citation)
8. [References](#references)
9. [License](#license)

---

## Overview

This repository provides complete, reproducible tools for intracranial EEG (iEEG/sEEG) research:

- Bipolar re-referencing of stereo-EEG (sEEG) recordings
- Zero-phase FIR bandpass, highpass, and lowpass filtering
- Automated artifact detection using Z-score (Staresina et al., 2015) and MAD-based robust methods
- Interactive epoch and channel rejection with a Matplotlib GUI
- BIDS-compliant data export via `mne-bids`
- Multitaper power spectral density estimation
- Circular–linear correlation between neural signals and stimulus orientations
- Permutation-based significance testing at the channel level with anatomical region mapping
- K-fold cross-validated SVR decoding of stimulus orientation from high-frequency spectral features
- Glass-brain visualization of significant anatomical regions

---

## Repository Structure

```
ieeg-working-memory/
├── ieeg_wm/                       # Importable Python package
│   ├── __init__.py                # Public API re-exports
│   ├── preprocessing.py           # Bipolar referencing, filtering, artifact detection, GUI rejection
│   └── analysis.py                # PSD, circular correlation, decoding, visualization
├── notebooks/                     # Numbered workflow notebooks
│   ├── 01_preprocessing.ipynb     # End-to-end preprocessing walkthrough
│   ├── 02_circular_correlation_svm.ipynb
│   ├── 03_forward_encoding.ipynb
│   └── 04_prepare_bids.ipynb
├── tests/                         # pytest unit tests
├── environment.yml
├── requirements.txt
├── pyproject.toml
└── LICENSE
```

---

## Installation

### 1. Clone and create the environment

```bash
git clone https://github.com/albertohurtado/ieeg-working-memory.git
cd ieeg-working-memory
conda env create -f environment.yml
conda activate ieeg-wm
```

### 2. Install the package in editable mode

```bash
pip install -e .
```

### 3. Install the decoding_toolbox

Required for `circ_corr_fun` and `decode_with_psd_permutation`. This package
is not available on PyPI — clone it manually:

```bash
git clone https://github.com/alepebel/decoding_toolbox.git
```

Then add its `Helper_funcs/` directory to your Python path before running the
relevant notebooks or scripts:

```python
import sys
sys.path.insert(0, "/path/to/decoding_toolbox/Helper_funcs")
```

### 4. Optional: autoreject

Required for `plot_rejection_with_rejectlog` and the RejectLog visualization:

```bash
pip install autoreject
```

### 5. Optional: deep-learning dependencies

Required for the forward encoding notebook:

```bash
pip install "ieeg-wm[deep-learning]"
```

---

## Quickstart

```python
import mne
from ieeg_wm.preprocessing import bipolar_reference_seeg, artifact_detection_tuned

raw = mne.io.read_raw_brainvision("my_recording.vhdr", preload=True)
raw_bip = bipolar_reference_seeg(raw)

data = raw_bip.get_data()
data_clean = artifact_detection_tuned(
    data,
    amp_thr=6.0,
    grad_thr=6.0,
    hpf_thr=6.0,
    min_art_sec=0.005,
    pad_samps=25,
    fill_gap_sec=0.02,
)
```

---

## Notebooks

| Notebook | Description |
|---|---|
| `01_preprocessing.ipynb` | End-to-end preprocessing: loading, bipolar referencing, filtering, artifact detection, epoch rejection |
| `02_circular_correlation_svm.ipynb` | Circular–linear correlation analysis and SVR decoding of stimulus orientation |
| `03_forward_encoding.ipynb` | Forward encoding model for working-memory representations |
| `04_prepare_bids.ipynb` | Converting raw iEEG recordings to BIDS format |

---

## API Reference

### `ieeg_wm.preprocessing`

| Function | Description |
|---|---|
| `bipolar_reference_seeg(raw)` | Convert monopolar sEEG to bipolar montage |
| `eegfilt(data, srate, ...)` | Zero-phase FIR bandpass / highpass / lowpass filter |
| `artifact_detection(data, ...)` | Z-score-based artifact detector |
| `artifact_detection_tuned(data, ...)` | MAD-based robust artifact detector (recommended) |
| `convert_to_mne(data_clean, raw_template)` | Wrap cleaned array in MNE RawArray |
| `calculate_metric(data_in, metric)` | Per-epoch/channel summary statistics |
| `reject_visual_mne(epochs, metric)` | Interactive epoch/channel rejection GUI |
| `plot_rejection_with_rejectlog(...)` | Visualize rejections via Autoreject RejectLog |
| `plot_mne_native_with_bads(...)` | MNE native epoch plot with bad channels highlighted |

### `ieeg_wm.analysis`

| Function | Description |
|---|---|
| `circ_corr_fun(Y, X)` | Trial-wise linear–circular correlation |
| `get_freq_spec(data, fs, ...)` | Multitaper PSD estimation |
| `norm_freq(psd, f, fband)` | Log-ratio high-frequency band normalization |
| `compute_null_psd(epochs, angles_rad, ...)` | True and permuted PSD–angle correlations |
| `channel_significance_by_window_psd(...)` | Permutation significance per channel and time window |
| `channel_significance_by_window_psd_preT(...)` | Same but using pre-stimulus metadata column |
| `plot_period_regions(...)` | Glass-brain visualization of active anatomical regions |
| `top_regions(df_locs, top_n)` | Most frequent significant anatomical regions |
| `plot_period_heatmap(...)` | Continuous-intensity heatmap overlay on glass-brain |
| `decode_with_psd_permutation(epochs, ...)` | K-fold SVR decoding with permutation p-value |
| `shrinkage_gamma(X, ...)` | Ledoit–Wolf covariance shrinkage coefficient |

---

## Citation

If you use this code, please cite the associated thesis:

```bibtex
@bachelorsthesis{hurtado2025ieeg,
  author  = {Hurtado, Alberto},
  title   = {Preprocessing and decoding neural traces of serial biases in
             working memory from intracranial electrophysiological recordings
             in humans},
  school  = {[University of Barcelona]},
  year    = {2025},
  url     = {https://www.linkedin.com/in/alberto-hurtado-morell/}
}
```

---

## References

Staresina, B. P., Bergmann, T. O., Bonefond, M., van der Meij, R., Jensen, O.,
Deuker, L., Elger, C. E., Axmacher, N., & Fell, J. (2015). Hierarchical nesting
of slow oscillations, spindles and ripples in the human hippocampus during sleep.
*Nature Neuroscience*, 18(11), 1679–1686. https://doi.org/10.1038/nn.4119

---

## License

MIT License — see [LICENSE](LICENSE) for details.
