"""ieeg_wm — Preprocessing and decoding tools for iEEG working-memory research.

Author: Alberto Hurtado (https://www.linkedin.com/in/alberto-hurtado-morell/)
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ieeg-wm")
except PackageNotFoundError:
    __version__ = "0.1.0"

from .analysis import (
    channel_significance_by_window_psd,
    channel_significance_by_window_psd_preT,
    circ_corr_fun,
    compute_null_psd,
    decode_with_psd_permutation,
    get_freq_spec,
    norm_freq,
    plot_period_heatmap,
    plot_period_regions,
    shrinkage_gamma,
    top_regions,
)
from .preprocessing import (
    artifact_detection,
    artifact_detection_tuned,
    bipolar_reference_seeg,
    calculate_metric,
    convert_to_mne,
    eegfilt,
    padding,
    plot_mne_native_with_bads,
    plot_rejection_with_rejectlog,
    reject_visual_mne,
    remove_small_segments,
)

__all__ = [
    # preprocessing
    "bipolar_reference_seeg",
    "padding",
    "remove_small_segments",
    "eegfilt",
    "artifact_detection",
    "artifact_detection_tuned",
    "convert_to_mne",
    "calculate_metric",
    "reject_visual_mne",
    "plot_rejection_with_rejectlog",
    "plot_mne_native_with_bads",
    # analysis
    "circ_corr_fun",
    "get_freq_spec",
    "norm_freq",
    "compute_null_psd",
    "channel_significance_by_window_psd",
    "channel_significance_by_window_psd_preT",
    "plot_period_regions",
    "top_regions",
    "plot_period_heatmap",
    "decode_with_psd_permutation",
    "shrinkage_gamma",
]
