"""iEEG preprocessing utilities: bipolar referencing, FIR filtering,
artifact detection (Z-score and MAD-based), and interactive epoch rejection.
"""

from __future__ import annotations

import traceback
import warnings

import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np
import scipy.signal as sp_signal
import scipy.stats as sp_stats
from matplotlib.widgets import Button, CheckButtons, RadioButtons, RectangleSelector
from scipy.ndimage import label
from scipy.signal import filtfilt, firwin, hilbert
from scipy.stats import median_abs_deviation as mad

try:
    from autoreject import RejectLog
except ImportError:
    RejectLog = None


def bipolar_reference_seeg(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Apply bipolar referencing to sEEG recordings.

    Groups channels by electrode prefix, sorts contacts numerically, and
    creates virtual bipolar channels by subtracting adjacent pairs.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        MNE Raw object with channel names encoding electrode and contact
        number (e.g., ``"A1"``, ``"A2"``).

    Returns
    -------
    mne.io.BaseRaw
        Raw object with bipolar virtual channels named
        ``"contact_i-contact_{i+1}"`` and original reference channels dropped.
    """
    ch_names = raw.ch_names
    electrode_groups = {}

    for ch in ch_names:
        prefix = ''.join(filter(str.isalpha, ch))
        if prefix not in electrode_groups:
            electrode_groups[prefix] = []
        electrode_groups[prefix].append(ch)

    for prefix in electrode_groups:
        try:
            electrode_groups[prefix].sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
        except ValueError:
            print(f"Skipping non-numeric channel: {electrode_groups[prefix]}")

    anodes, cathodes, bipolar_names = [], [], []
    for prefix, contacts in electrode_groups.items():
        contacts = [c for c in contacts if any(char.isdigit() for char in c)]
        for i in range(len(contacts) - 1):
            anodes.append(contacts[i])
            cathodes.append(contacts[i + 1])
            bipolar_names.append(f"{contacts[i]}-{contacts[i+1]}")

    raw_bipolar = mne.set_bipolar_reference(raw, anodes, cathodes, ch_name=bipolar_names, drop_refs=True)
    return raw_bipolar


def padding(data: np.ndarray, padding_value: int) -> np.ndarray:
    """Extend binary artifact markers around each detected cluster.

    Parameters
    ----------
    data : np.ndarray
        One-dimensional binary array where ``1`` denotes an artifact sample.
        Modified in-place.
    padding_value : int
        Number of samples to extend on each side of every artifact cluster.

    Returns
    -------
    np.ndarray
        Modified ``data`` with padded artifact regions set to ``1``.
    """
    labeled_array, num_features = label(data)

    if num_features > 0:
        for i in range(1, num_features + 1):
            clust = np.where(labeled_array == i)[0]

            if clust[0] - padding_value >= 0:
                data[clust[0] - padding_value: clust[0]] = 1

            if clust[-1] + padding_value < len(data):
                data[clust[-1]: clust[-1] + padding_value] = 1

    return data


def remove_small_segments(data: np.ndarray, min_seg_length: int) -> np.ndarray:
    """Fill short clean gaps between artifact clusters.

    Identifies non-artifact runs (zeros) shorter than *min_seg_length* and
    converts them to artifacts, bridging adjacent artifact clusters.

    Parameters
    ----------
    data : np.ndarray
        One-dimensional binary array where ``1`` denotes artifact samples.
        Modified in-place.
    min_seg_length : int
        Minimum number of clean (zero) samples to preserve as non-artifact.
        Runs shorter than this are set to ``1``.

    Returns
    -------
    np.ndarray
        Modified ``data`` with short zero-runs set to ``1``.
    """
    mks = ~data.astype(bool)
    labeled_array, num_features = label(mks)

    if num_features > 0:
        segment_sizes = np.bincount(labeled_array.flat)[1:]
        small_segments = np.where(segment_sizes < min_seg_length)[0] + 1

        for seg in small_segments:
            data[labeled_array == seg] = 1

    return data


def eegfilt(
    data: np.ndarray,
    srate: float,
    locutoff: float | None = None,
    hicutoff: float | None = None,
    epochframes: int = 0,
    filtorder: int | None = None,
    revfilt: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Design and apply a zero-phase FIR filter to multi-channel data.

    Applies a bandpass, high-pass, or low-pass FIR filter using
    ``scipy.signal.filtfilt`` (forward-backward to eliminate phase
    distortion). Data can be filtered in equal-length segments to manage
    edge effects.

    Parameters
    ----------
    data : np.ndarray
        Shape ``(n_channels, n_times)`` — raw signal array.
    srate : float
        Sampling frequency in Hz.
    locutoff : float or None, optional
        Low-cutoff frequency in Hz. ``None`` omits a high-pass stage.
    hicutoff : float or None, optional
        High-cutoff frequency in Hz. ``None`` omits a low-pass stage.
    epochframes : int, optional
        Segment length in samples. ``0`` (default) processes the entire
        recording as one chunk.
    filtorder : int or None, optional
        FIR filter order. If ``None``, computed as
        ``max(3 * int(srate / cutoff), 15)``.
    revfilt : int, optional
        If ``1``, inverts the filter (notch instead of bandpass, etc.).
        Default is ``0``.

    Returns
    -------
    smoothdata : np.ndarray
        Filtered signal of the same shape as *data*.
    filtwts : np.ndarray
        One-dimensional array of FIR filter coefficients.

    Raises
    ------
    ValueError
        If cutoff frequencies are out of range, or if *epochframes* does
        not evenly divide the signal or is too short for the filter order.
    """
    chans, frames = data.shape
    nyq = srate * 0.5
    minfac = 3
    min_filtorder = 15
    trans = 0.15

    if locutoff is not None and hicutoff is not None and locutoff > hicutoff:
        raise ValueError("locutoff must be <= hicutoff.")
    if (locutoff is not None and locutoff < 0) or (hicutoff is not None and hicutoff < 0):
        raise ValueError("Cutoff frequencies must be non-negative.")
    if (locutoff is not None and locutoff >= nyq) or (hicutoff is not None and hicutoff >= nyq):
        raise ValueError("Cutoff frequencies must be less than Nyquist.")

    if filtorder is None:
        if locutoff is not None:
            filtorder = minfac * int(srate / locutoff)
        elif hicutoff is not None:
            filtorder = minfac * int(srate / hicutoff)
        filtorder = max(filtorder, min_filtorder)

    if epochframes == 0:
        epochframes = frames
    epochs = frames // epochframes
    if epochs * epochframes != frames:
        raise ValueError("epochframes does not evenly divide frames.")
    if filtorder * 3 > epochframes:
        raise ValueError("epochframes must be at least 3 times the filtorder.")

    if locutoff is not None and hicutoff is not None:
        bands = [0, locutoff * (1 - trans), locutoff, hicutoff, hicutoff * (1 + trans), nyq]
        desired = [0, 0, 1, 1, 0, 0]
        cutoff = [bands[2], bands[3]]
    elif locutoff is not None:
        bands = [0, locutoff * (1 - trans), locutoff, nyq]
        desired = [0, 0, 1, 1]
        cutoff = [bands[2]]
    elif hicutoff is not None:
        bands = [0, hicutoff, hicutoff * (1 + trans), nyq]
        desired = [1, 1, 0, 0]
        cutoff = [bands[1]]
    else:
        raise ValueError("You must provide a non-zero low or high cutoff frequency.")

    if revfilt:
        desired = [1 - d for d in desired]

    normalized_cutoff = [c / nyq for c in cutoff]
    filtwts = firwin(filtorder + 1, normalized_cutoff, pass_zero=desired[0] == 1)

    smoothdata = np.zeros_like(data)
    for e in range(epochs):
        for c in range(chans):
            segment = data[c, e * epochframes:(e + 1) * epochframes]
            smoothdata[c, e * epochframes:(e + 1) * epochframes] = filtfilt(filtwts, 1, segment)

    return smoothdata, filtwts


def artifact_detection(
    data: np.ndarray,
    std_thres: float,
    std_thres2: float,
    padding_value: int,
    min_seg_length: int,
) -> np.ndarray:
    """Detect artifacts using Z-score thresholding and replace with NaN.

    For each channel, flags samples where amplitude, gradient, or
    high-frequency envelope Z-score exceeds *std_thres*, or where amplitude
    Z-score exceeds *std_thres2* jointly with gradient or envelope Z-score.
    Applies padding and gap-filling post-processing.

    Parameters
    ----------
    data : np.ndarray
        Shape ``(n_channels, n_times)`` — raw iEEG signal.
    std_thres : float
        Primary Z-score threshold applied to individual metrics.
    std_thres2 : float
        Secondary (stricter) Z-score threshold requiring joint conditions.
    padding_value : int
        Samples to extend each detected artifact cluster on both sides.
    min_seg_length : int
        Minimum clean segment length to preserve; shorter gaps are bridged.

    Returns
    -------
    np.ndarray
        Array of the same shape as *data* with artifact samples replaced by
        ``np.nan``.
    """
    data_clean = np.copy(data)

    for chani in range(data.shape[0]):
        channel_data = data[chani, :]

        z_score_amp = (channel_data - np.mean(channel_data)) / np.std(channel_data)
        grad = np.diff(channel_data, append=np.nan)
        z_score_grad = (grad - np.nanmean(grad)) / np.nanstd(grad)

        if channel_data.ndim == 1:
            channel_data = channel_data[np.newaxis, :]

        hpf_data, _ = eegfilt(channel_data, 500, None, 249)
        hpf_data = np.abs(hilbert(hpf_data))
        z_score_hpf_d = (hpf_data - np.mean(hpf_data)) / np.std(hpf_data)

        markers = np.zeros_like(channel_data)
        condition1 = (z_score_amp > std_thres) | (z_score_grad > std_thres) | (z_score_hpf_d > std_thres)
        condition2 = (z_score_amp > std_thres2) & ((z_score_grad > std_thres2) | (z_score_hpf_d > std_thres2))
        markers[condition1 | condition2] = 1

        new_trace = padding(markers, padding_value)
        markers = remove_small_segments(new_trace, min_seg_length)

        data_clean[chani, markers.flatten() == 1] = np.nan

    return data_clean


def artifact_detection_tuned(
    data: np.ndarray,
    amp_thr: float,
    grad_thr: float,
    hpf_thr: float,
    min_art_sec: float,
    pad_samps: int,
    fill_gap_sec: float,
    overshoot_fac: float = 1.3,
    sfreq: float = 500.0,
) -> np.ndarray:
    """Detect artifacts using MAD-based thresholding and replace them with NaN.

    For each channel, three metrics are computed: raw amplitude, first
    derivative, and high-frequency (>240 Hz) analytic envelope. Each metric
    is normalised by its median and MAD. A sample is flagged if any metric
    exceeds its threshold. Short clusters are pruned unless they contain a
    spike exceeding ``amp_thr * overshoot_fac``. Surviving clusters are
    padded, and clean gaps shorter than *fill_gap_sec* are bridged.

    Follows the artifact detection approach of Staresina et al. (2015,
    Nature Neuroscience) with modifications for robustness against
    non-Gaussian iEEG noise.

    Parameters
    ----------
    data : np.ndarray
        Shape ``(n_channels, n_samples)`` — raw iEEG in microvolts.
    amp_thr : float
        Amplitude threshold in MAD units. Samples where
        ``(x - median(x)) / MAD(x) > amp_thr`` are flagged.
    grad_thr : float
        First-derivative threshold in MAD units.
    hpf_thr : float
        High-frequency envelope threshold in MAD units. The envelope is
        computed as the absolute value of the Hilbert transform applied to
        a high-pass-filtered (>240 Hz) copy of the signal.
    min_art_sec : float
        Minimum artifact cluster duration in seconds. Clusters shorter than
        ``int(min_art_sec * sfreq)`` samples are discarded unless their peak
        MAD score exceeds ``amp_thr * overshoot_fac``.
    pad_samps : int
        Number of samples to extend each surviving artifact cluster on both
        sides, capturing gradual onset and offset transients.
    fill_gap_sec : float
        Maximum duration in seconds of a clean gap between two artifact
        clusters that is bridged to produce a contiguous block.
    overshoot_fac : float, optional
        Multiplier applied to *amp_thr* for the "giant spike" criterion. Any
        cluster whose peak MAD score exceeds ``amp_thr * overshoot_fac`` is
        retained regardless of duration. Default is ``1.3``.
    sfreq : float, optional
        Sampling frequency in Hz. Default is ``500.0``.

    Returns
    -------
    np.ndarray
        Array of the same shape as *data* with artifact samples replaced by
        ``np.nan``; clean samples are unchanged.

    Notes
    -----
    Internally calls :func:`eegfilt`, :func:`padding`, and
    :func:`remove_small_segments`.

    References
    ----------
    Staresina, B. P., et al. (2015). Hierarchical nesting of slow oscillations,
    spindles and ripples in the human hippocampus during sleep. *Nature
    Neuroscience*, 18(11), 1679–1686. https://doi.org/10.1038/nn.4119

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> data = rng.standard_normal((4, 5000))
    >>> data[0, 1000:1005] = 500.0          # inject a spike
    >>> cleaned = artifact_detection_tuned(
    ...     data, amp_thr=5.0, grad_thr=5.0, hpf_thr=5.0,
    ...     min_art_sec=0.005, pad_samps=25, fill_gap_sec=0.02,
    ... )
    >>> np.isnan(cleaned[0, 1002])
    True
    """
    nchan, nsamp = data.shape
    out = data.copy()
    min_art = int(min_art_sec * sfreq)
    fill_gap = int(fill_gap_sec * sfreq)

    for ch in range(nchan):
        x = data[ch]

        med_x = np.median(x)
        mad_x = mad(x, scale='normal')
        z_amp = (x - med_x) / mad_x

        grad = np.diff(x, append=x[-1])
        med_g = np.median(grad)
        mad_g = mad(grad, scale='normal')
        z_grad = (grad - med_g) / mad_g

        hpf, _ = eegfilt(x[np.newaxis, :], sfreq, None, 240)
        env = np.abs(hilbert(hpf)).flatten()
        med_h = np.median(env)
        mad_h = mad(env, scale='normal')
        z_hpf = (env - med_h) / mad_h

        mask0 = (
            (z_amp > amp_thr)
            | (z_grad > grad_thr)
            | (z_hpf > hpf_thr)
        ).astype(int)

        labeled, nseg = label(mask0)
        for seg_id in range(1, nseg + 1):
            idx = np.where(labeled == seg_id)[0]
            max_z = max(z_amp[idx].max(), z_grad[idx].max(), z_hpf[idx].max())

            if idx.size < min_art and max_z < amp_thr * overshoot_fac:
                mask0[idx] = 0

        mask1 = padding(mask0, pad_samps)
        mask2 = remove_small_segments(mask1, fill_gap)
        out[ch, mask2 == 1] = np.nan

    return out


def convert_to_mne(
    data_clean: np.ndarray,
    raw_template: mne.io.BaseRaw,
) -> mne.io.RawArray:
    """Wrap a cleaned NumPy array in an MNE RawArray.

    Channel names and sampling frequency are copied from *raw_template*.
    Channels with ``"trig"`` in their name (case-insensitive) are typed as
    ``"stim"``; all others are typed as ``"seeg"``.

    Parameters
    ----------
    data_clean : np.ndarray
        Shape ``(n_channels, n_times)`` — cleaned iEEG data.
    raw_template : mne.io.BaseRaw
        Template Raw object supplying ``ch_names`` and ``sfreq``.

    Returns
    -------
    mne.io.RawArray
        New RawArray containing *data_clean* with info from *raw_template*.
    """
    ch_names = raw_template.ch_names
    sfreq = raw_template.info['sfreq']

    ch_types = ['stim' if 'trig' in ch.lower() else 'seeg' for ch in ch_names]

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw_clean = mne.io.RawArray(data_clean, info)

    return raw_clean


def calculate_metric(
    data_in: np.ndarray,
    metric: str = 'var',
) -> tuple[np.ndarray | None, bool]:
    """Compute a per-epoch, per-channel summary statistic.

    Handles NaN values and SciPy version differences for ``mad`` and
    ``kurtosis`` computations.

    Parameters
    ----------
    data_in : np.ndarray
        Shape ``(n_epochs, n_channels, n_times)``.
    metric : str, optional
        Statistic to compute. One of ``'var'``, ``'std'``, ``'mad'``,
        ``'1/var'``, ``'min'``, ``'max'``, ``'maxabs'``, ``'range'``
        (alias ``'ptp'``), or ``'kurtosis'``. Default is ``'var'``.

    Returns
    -------
    metric_data : np.ndarray or None
        Shape ``(n_channels, n_epochs)`` with computed values. ``None`` if
        *metric* is not recognised.
    found_nans : bool
        ``True`` if *data_in* contained any ``np.nan`` values.
    """
    found_nans = False

    with np.errstate(invalid='ignore'):
        if np.isnan(data_in).any():
            found_nans = True

    n_epochs, n_channels, n_times = data_in.shape
    metric_data = np.full((n_channels, n_epochs), np.nan)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for i_chan in range(n_channels):
            chan_data = data_in[:, i_chan, :]

            if metric == 'var':
                metric_data[i_chan, :] = np.nanvar(chan_data, axis=1)
            elif metric == 'std':
                metric_data[i_chan, :] = np.nanstd(chan_data, axis=1)
            elif metric == 'mad':
                try:
                    metric_data[i_chan, :] = sp_stats.median_abs_deviation(
                        chan_data, axis=1, scale='normal', nan_policy='omit'
                    )
                except TypeError:
                    warnings.warn(
                        "SciPy version < 1.7.0 detected. "
                        "MAD calculation cannot ignore NaNs natively via 'nan_policy'. "
                        "Manually skipping NaNs. Result may be NaN if all data in an epoch is NaN.",
                        RuntimeWarning, stacklevel=2
                    )
                    for i_epoch in range(n_epochs):
                        epoch_data = chan_data[i_epoch, :]
                        if np.all(np.isnan(epoch_data)):
                            metric_data[i_chan, i_epoch] = np.nan
                            continue
                        metric_data[i_chan, i_epoch] = sp_stats.median_abs_deviation(
                            epoch_data[~np.isnan(epoch_data)], scale='normal'
                        )
            elif metric == '1/var':
                variances = np.nanvar(chan_data, axis=1)
                variances[variances == 0] = np.finfo(float).eps
                metric_data[i_chan, :] = 1.0 / variances
            elif metric == 'min':
                metric_data[i_chan, :] = np.nanmin(chan_data, axis=1)
            elif metric == 'max':
                metric_data[i_chan, :] = np.nanmax(chan_data, axis=1)
            elif metric == 'maxabs':
                metric_data[i_chan, :] = np.nanmax(np.abs(chan_data), axis=1)
            elif metric in ['range', 'ptp']:
                max_vals = np.nanmax(chan_data, axis=1)
                min_vals = np.nanmin(chan_data, axis=1)
                metric_data[i_chan, :] = max_vals - min_vals
            elif metric == 'kurtosis':
                try:
                    metric_data[i_chan, :] = sp_stats.kurtosis(
                        chan_data, axis=1, nan_policy='omit'
                    )
                except TypeError:
                    warnings.warn(
                        "Kurtosis calculation cannot ignore NaNs natively via 'nan_policy'. "
                        "Manually skipping NaNs. Result may be NaN if all data in an epoch is NaN.",
                        RuntimeWarning, stacklevel=2
                    )
                    for i_epoch in range(n_epochs):
                        epoch_data = chan_data[i_epoch, :]
                        if np.all(np.isnan(epoch_data)):
                            metric_data[i_chan, i_epoch] = np.nan
                            continue
                        metric_data[i_chan, i_epoch] = sp_stats.kurtosis(
                            epoch_data[~np.isnan(epoch_data)]
                        )
            else:
                warnings.warn(f"Metric '{metric}' not implemented.", UserWarning, stacklevel=2)
                return None, found_nans

    return metric_data, found_nans


def reject_visual_mne(
    epochs: mne.BaseEpochs,
    metric: str = 'var',
    sfreq: float | None = None,
) -> tuple[list[int], list[str]]:
    """Interactive Matplotlib GUI for epoch and channel rejection.

    Launches a blocking figure with a metric heatmap (channels × trials),
    per-channel and per-trial scatter summaries, an optional Welch power
    spectrum, and radio-button metric switching. Left-click or drag-rectangle
    to toggle individual trials/channels as rejected. Click
    "Quit & Return Rejected" to close the figure and return selections.

    Parameters
    ----------
    epochs : mne.BaseEpochs
        Segmented data. EEG, MEG, ECoG, and sEEG channel types are shown.
    metric : str, optional
        Summary statistic for the heatmap. One of ``'var'``, ``'std'``,
        ``'mad'``, ``'maxabs'``, ``'ptp'``, ``'kurtosis'``. Default ``'var'``.
    sfreq : float or None, optional
        Sampling frequency in Hz. If ``None``, read from
        ``epochs.info['sfreq']``.

    Returns
    -------
    rejected_trials : list of int
        Zero-based indices of epochs marked as bad.
    rejected_channels : list of str
        Names of channels marked as bad.

    Notes
    -----
    Calls ``plt.show(block=True)`` and blocks until the figure is closed.
    Requires an interactive Matplotlib backend (e.g., Qt5Agg, TkAgg, or
    ``%matplotlib widget`` in Jupyter).

    This function is intentionally kept as a single unit to preserve tightly
    coupled widget state management.
    """
    if not isinstance(epochs, mne.BaseEpochs):
        raise TypeError("Input must be an MNE Epochs object.")

    data = epochs.get_data(picks=['eeg', 'meg', 'ecog', 'seeg'])
    if data is None or data.size == 0:
        warnings.warn("No data found for EEG, MEG, ECoG, or SEEG channels. Returning empty selections.", UserWarning)
        return [], []

    n_epochs, n_channels, n_times = data.shape
    ch_names = epochs.copy().pick_types(eeg=True, meg=True, ecog=True, seeg=True).ch_names

    if sfreq is None:
        sfreq = epochs.info['sfreq']

    current_metric = metric
    rejected_trials = set()
    rejected_channels_idx = set()
    bad_channel_names = set()
    calculate_spectrum = True
    _nan_warning_issued = False

    plot_elements = {}
    selectors = {}

    def _calculate_metric_and_warn(data_to_calc, metric_name):
        nonlocal _nan_warning_issued
        metric_res, found_nans = calculate_metric(data_to_calc, metric_name)
        if found_nans and not _nan_warning_issued:
            warnings.warn(
                "Input data contains NaNs. Metrics will be calculated ignoring NaNs. "
                "Resulting metrics may be NaN if all data points for a specific "
                "channel/epoch calculation are NaN.",
                RuntimeWarning, stacklevel=3
            )
            _nan_warning_issued = True
        return metric_res

    metric_data = _calculate_metric_and_warn(data, current_metric)
    if metric_data is None:
        warnings.warn(
            f"Initial metric '{current_metric}' is invalid or calculation failed. Defaulting to 'var'.",
            UserWarning
        )
        current_metric = 'var'
        metric_data = _calculate_metric_and_warn(data, current_metric)
        if metric_data is None:
            raise ValueError("Could not calculate the default 'var' metric. Check input data.")

    if np.all(np.isnan(metric_data)):
        warnings.warn(
            "Initial metric calculation resulted in all NaNs. "
            "Check input data or the chosen metric. The visualization might be uninformative.",
            RuntimeWarning, stacklevel=2
        )

    fig = plt.figure(figsize=(12, 8.5))
    gs = fig.add_gridspec(
        4, 3, width_ratios=[3, 2, 1.5], height_ratios=[2, 2, 0.7, 0.4],
        hspace=0.4, wspace=0.3
    )
    ax_summary = fig.add_subplot(gs[0, 0])
    ax_chan_summary = fig.add_subplot(gs[0, 1], sharey=ax_summary)
    ax_trial_summary = fig.add_subplot(gs[1, 0], sharex=ax_summary)
    ax_spectrum = fig.add_subplot(gs[1, 1])
    ax_controls = fig.add_subplot(gs[0:2, 2])
    ax_metric_radio = fig.add_subplot(gs[2, 0])
    ax_spec_check = fig.add_subplot(gs[2, 1])
    ax_quit_button = fig.add_subplot(gs[3, 2])

    plt.setp(ax_chan_summary.get_yticklabels(), visible=False)
    plt.setp(ax_trial_summary.get_xticklabels(), visible=False)
    ax_controls.axis('off')

    def update_spectrum_plot():
        ax_spectrum.clear()
        if calculate_spectrum:
            good_trials_idx = sorted(list(set(range(n_epochs)) - rejected_trials))
            good_channels_idx = sorted(list(set(range(n_channels)) - rejected_channels_idx))

            if not good_trials_idx or not good_channels_idx:
                ax_spectrum.text(0.5, 0.5, 'No good data selected\nfor spectrum',
                                 ha='center', va='center', transform=ax_spectrum.transAxes)
                ax_spectrum.set_title('Avg Spectrum')
            else:
                good_data = data[np.ix_(good_trials_idx, good_channels_idx, np.arange(n_times))]

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    avg_data_for_spectrum = np.nanmean(good_data, axis=(0, 1))

                if np.all(np.isnan(avg_data_for_spectrum)):
                    warnings.warn(
                        "Average time series for spectrum calculation is all NaNs. Skipping Welch calculation.",
                        RuntimeWarning, stacklevel=2
                    )
                    ax_spectrum.text(0.5, 0.5, 'Cannot compute spectrum\n(average data is all NaN)',
                                     ha='center', va='center', transform=ax_spectrum.transAxes)
                    ax_spectrum.set_title('Avg Spectrum (Error)')
                elif np.any(np.isnan(avg_data_for_spectrum)):
                    warnings.warn(
                        "Average time series for spectrum calculation contains some NaNs. "
                        "This might affect Welch calculation or indicate an issue.",
                        RuntimeWarning, stacklevel=2
                    )
                    ax_spectrum.text(0.5, 0.5, 'Cannot compute spectrum\n(NaNs in avg data)',
                                     ha='center', va='center', transform=ax_spectrum.transAxes)
                    ax_spectrum.set_title('Avg Spectrum (Error)')
                else:
                    freqs, psd = sp_signal.welch(avg_data_for_spectrum, fs=sfreq, nperseg=min(n_times, 256))
                    ax_spectrum.semilogy(freqs, psd)
                    ax_spectrum.set_xlabel('Frequency (Hz)')
                    ax_spectrum.set_ylabel('PSD (Power/Hz)')
                    ax_spectrum.set_title('Avg Spectrum (Good Data)')
                    ax_spectrum.grid(True, linestyle=':')
                    ax_spectrum.set_xlim([freqs[0], sfreq / 2])
        else:
            ax_spectrum.text(0.5, 0.5, 'Spectrum calculation\ndisabled',
                             ha='center', va='center', transform=ax_spectrum.transAxes)
            ax_spectrum.set_title('Avg Spectrum (Disabled)')
            ax_spectrum.tick_params(axis='both', which='both', left=False, bottom=False,
                                    labelleft=False, labelbottom=False)
        fig.canvas.draw_idle()

    def update_plots():
        nonlocal metric_data
        metric_data = _calculate_metric_and_warn(data, current_metric)
        if metric_data is None:
            warnings.warn(f"Failed to calculate metric '{current_metric}'. Plots will not update.", UserWarning)
            return

        if np.all(np.isnan(metric_data)):
            warnings.warn(
                f"Metric '{current_metric}' resulted in all NaNs for the current data. "
                "The visualization might be uninformative.",
                RuntimeWarning, stacklevel=2
            )

        good_trials_idx = sorted(list(set(range(n_epochs)) - rejected_trials))
        good_channels_idx_for_summary = sorted(list(set(range(n_channels)) - rejected_channels_idx))

        ax_summary.clear()

        current_cmap = matplotlib.colormaps['viridis'].copy()
        current_cmap.set_bad(color='grey', alpha=0.5)
        im = ax_summary.imshow(metric_data, aspect='auto', cmap=current_cmap,
                               origin='lower', interpolation='nearest')
        plot_elements['summary_im'] = im
        ax_summary.set_title(f'Summary Metric ({current_metric})')
        ax_summary.set_xlabel('Trial Number')
        ax_summary.set_ylabel('Channel Number')

        for r_chan_idx in rejected_channels_idx:
            ax_summary.axhline(r_chan_idx, color='white', linestyle='--', alpha=0.7, lw=0.8)
        for r_trial_idx in rejected_trials:
            ax_summary.axvline(r_trial_idx, color='white', linestyle='--', alpha=0.7, lw=0.8)

        ax_chan_summary.clear()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            chan_means = np.nanmean(metric_data[:, good_trials_idx], axis=1) if good_trials_idx else np.full(n_channels, np.nan)

        chan_indices_all = np.arange(n_channels)

        good_ch_mask = np.ones(n_channels, dtype=bool)
        if list(rejected_channels_idx):
            good_ch_mask[list(rejected_channels_idx)] = False

        plot_elements['chan_scatter_good'] = ax_chan_summary.scatter(
            chan_means[good_ch_mask], chan_indices_all[good_ch_mask],
            c='blue', marker='.', label='Good Ch'
        )
        if np.any(~good_ch_mask):
            plot_elements['chan_scatter_bad'] = ax_chan_summary.scatter(
                chan_means[~good_ch_mask], chan_indices_all[~good_ch_mask],
                c='red', marker='x', label='Bad Ch'
            )
        ax_chan_summary.set_xlabel(f'Mean {current_metric} (good trials)')
        ax_chan_summary.tick_params(axis='y', which='both', left=False, labelleft=False)
        ax_chan_summary.grid(True, axis='x', linestyle=':')
        ax_chan_summary.set_ylim(ax_summary.get_ylim())

        ax_trial_summary.clear()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            trial_means = np.nanmean(metric_data[good_channels_idx_for_summary, :], axis=0) if good_channels_idx_for_summary else np.full(n_epochs, np.nan)

        trial_indices_all = np.arange(n_epochs)

        good_tr_mask = np.ones(n_epochs, dtype=bool)
        if list(rejected_trials):
            good_tr_mask[list(rejected_trials)] = False

        plot_elements['trial_scatter_good'] = ax_trial_summary.scatter(
            trial_indices_all[good_tr_mask], trial_means[good_tr_mask],
            c='blue', marker='.', label='Good Tr'
        )
        if np.any(~good_tr_mask):
            plot_elements['trial_scatter_bad'] = ax_trial_summary.scatter(
                trial_indices_all[~good_tr_mask], trial_means[~good_tr_mask],
                c='red', marker='x', label='Bad Tr'
            )
        ax_trial_summary.set_ylabel(f'Mean {current_metric} (good channels)')
        ax_trial_summary.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        ax_trial_summary.grid(True, axis='y', linestyle=':')
        ax_trial_summary.set_xlim(ax_summary.get_xlim())

        update_spectrum_plot()
        update_control_text()
        fig.canvas.draw_idle()

    def update_control_text():
        ax_controls.clear()
        ax_controls.axis('off')

        rejected_ch_names_list = sorted([ch_names[i] for i in rejected_channels_idx])
        if len(rejected_ch_names_list) > 10:
            ch_text_list = rejected_ch_names_list[:5] + ['...'] + rejected_ch_names_list[-5:]
        else:
            ch_text_list = rejected_ch_names_list
        ch_text = ", ".join(map(str, ch_text_list)) if ch_text_list else "None"

        rejected_trials_list = sorted(list(rejected_trials))
        if len(rejected_trials_list) > 10:
            tr_text_list = [str(t) for t in rejected_trials_list[:5]] + ['...'] + [str(t) for t in rejected_trials_list[-5:]]
        else:
            tr_text_list = [str(t) for t in rejected_trials_list]
        tr_text = ", ".join(tr_text_list) if tr_text_list else "None"

        info_text = (
            f"Metric: {current_metric}\n\n"
            f"Interaction:\n- Click point or drag box on\n  side plots to toggle selection.\n\n"
            f"Rejected Trials: {len(rejected_trials)}/{n_epochs}\n  {tr_text}\n\n"
            f"Rejected Channels: {len(rejected_channels_idx)}/{n_channels}\n  {ch_text}"
        )
        ax_controls.text(0.05, 0.95, info_text, ha='left', va='top', wrap=True, fontsize=9)
        fig.canvas.draw_idle()

    def on_click(event):
        if event.button != 1:
            return

        toggled = False
        target_chan_idx = -1
        target_trial_idx = -1

        if event.inaxes == ax_chan_summary:
            min_dist_sq = 25

            scatter_good = plot_elements.get('chan_scatter_good')
            if scatter_good:
                offsets = scatter_good.get_offsets()
                if offsets.size > 0:
                    valid_offsets = offsets[~np.isnan(offsets).any(axis=1)]
                    if valid_offsets.size > 0:
                        display_coords = ax_chan_summary.transData.transform(valid_offsets)
                        click_display = (event.x, event.y)
                        distances_sq = np.sum((display_coords - click_display) ** 2, axis=1)
                        if distances_sq.size > 0:
                            min_idx_local = np.argmin(distances_sq)
                            if distances_sq[min_idx_local] < min_dist_sq:
                                target_chan_idx = int(round(valid_offsets[min_idx_local, 1]))

            scatter_bad = plot_elements.get('chan_scatter_bad')
            if scatter_bad:
                offsets = scatter_bad.get_offsets()
                if offsets.size > 0:
                    valid_offsets = offsets[~np.isnan(offsets).any(axis=1)]
                    if valid_offsets.size > 0:
                        display_coords = ax_chan_summary.transData.transform(valid_offsets)
                        click_display = (event.x, event.y)
                        distances_sq = np.sum((display_coords - click_display) ** 2, axis=1)
                        if distances_sq.size > 0:
                            min_idx_local = np.argmin(distances_sq)
                            if distances_sq[min_idx_local] < min_dist_sq:
                                current_best_dist = distances_sq[min_idx_local]
                                if target_chan_idx == -1 or \
                                        (target_chan_idx != -1 and current_best_dist < np.min(distances_sq)):
                                    target_chan_idx = int(round(valid_offsets[min_idx_local, 1]))

            if 0 <= target_chan_idx < n_channels:
                chan_name = ch_names[target_chan_idx]
                if target_chan_idx in rejected_channels_idx:
                    rejected_channels_idx.remove(target_chan_idx)
                    if chan_name in bad_channel_names:
                        bad_channel_names.remove(chan_name)
                else:
                    rejected_channels_idx.add(target_chan_idx)
                    bad_channel_names.add(chan_name)
                toggled = True

        elif event.inaxes == ax_trial_summary:
            min_dist_sq = 25
            scatter_good = plot_elements.get('trial_scatter_good')
            if scatter_good:
                offsets = scatter_good.get_offsets()
                if offsets.size > 0:
                    valid_offsets = offsets[~np.isnan(offsets).any(axis=1)]
                    if valid_offsets.size > 0:
                        display_coords = ax_trial_summary.transData.transform(valid_offsets)
                        click_display = (event.x, event.y)
                        distances_sq = np.sum((display_coords - click_display) ** 2, axis=1)
                        if distances_sq.size > 0:
                            min_idx_local = np.argmin(distances_sq)
                            if distances_sq[min_idx_local] < min_dist_sq:
                                target_trial_idx = int(round(valid_offsets[min_idx_local, 0]))

            scatter_bad = plot_elements.get('trial_scatter_bad')
            if scatter_bad:
                offsets = scatter_bad.get_offsets()
                if offsets.size > 0:
                    valid_offsets = offsets[~np.isnan(offsets).any(axis=1)]
                    if valid_offsets.size > 0:
                        display_coords = ax_trial_summary.transData.transform(valid_offsets)
                        click_display = (event.x, event.y)
                        distances_sq = np.sum((display_coords - click_display) ** 2, axis=1)
                        if distances_sq.size > 0:
                            min_idx_local = np.argmin(distances_sq)
                            if distances_sq[min_idx_local] < min_dist_sq:
                                current_best_dist = distances_sq[min_idx_local]
                                if target_trial_idx == -1 or \
                                        (target_trial_idx != -1 and current_best_dist < np.min(distances_sq)):
                                    target_trial_idx = int(round(valid_offsets[min_idx_local, 0]))

            if 0 <= target_trial_idx < n_epochs:
                if target_trial_idx in rejected_trials:
                    rejected_trials.remove(target_trial_idx)
                else:
                    rejected_trials.add(target_trial_idx)
                toggled = True

        if toggled:
            update_plots()

    def onselect_trials(eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        selected_min_x, selected_max_x = min(x1, x2), max(x1, x2)
        selected_min_y, selected_max_y = min(y1, y2), max(y1, y2)

        toggled_trials_in_selection = set()
        for key in ['trial_scatter_good', 'trial_scatter_bad']:
            scatter_plot = plot_elements.get(key)
            if scatter_plot:
                offsets = scatter_plot.get_offsets()
                if offsets.size > 0:
                    valid_offsets = offsets[~np.isnan(offsets).any(axis=1)]
                    if valid_offsets.size > 0:
                        mask = (
                            (valid_offsets[:, 0] >= selected_min_x)
                            & (valid_offsets[:, 0] <= selected_max_x)
                            & (valid_offsets[:, 1] >= selected_min_y)
                            & (valid_offsets[:, 1] <= selected_max_y)
                        )
                        toggled_trials_in_selection.update(valid_offsets[mask, 0].astype(int))
        if toggled_trials_in_selection:
            for idx in toggled_trials_in_selection:
                if idx in rejected_trials:
                    rejected_trials.remove(idx)
                else:
                    rejected_trials.add(idx)
            update_plots()

    def onselect_channels(eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        selected_min_x, selected_max_x = min(x1, x2), max(x1, x2)
        selected_min_y, selected_max_y = min(y1, y2), max(y1, y2)

        toggled_channels_idx_in_selection = set()
        for key in ['chan_scatter_good', 'chan_scatter_bad']:
            scatter_plot = plot_elements.get(key)
            if scatter_plot:
                offsets = scatter_plot.get_offsets()
                if offsets.size > 0:
                    valid_offsets = offsets[~np.isnan(offsets).any(axis=1)]
                    if valid_offsets.size > 0:
                        mask = (
                            (valid_offsets[:, 0] >= selected_min_x)
                            & (valid_offsets[:, 0] <= selected_max_x)
                            & (valid_offsets[:, 1] >= selected_min_y)
                            & (valid_offsets[:, 1] <= selected_max_y)
                        )
                        toggled_channels_idx_in_selection.update(valid_offsets[mask, 1].astype(int))

        if toggled_channels_idx_in_selection:
            for idx in toggled_channels_idx_in_selection:
                if 0 <= idx < n_channels:
                    chan_name = ch_names[idx]
                    if idx in rejected_channels_idx:
                        rejected_channels_idx.remove(idx)
                        if chan_name in bad_channel_names:
                            bad_channel_names.remove(chan_name)
                    else:
                        rejected_channels_idx.add(idx)
                        bad_channel_names.add(chan_name)
            update_plots()

    def on_metric_select(label):
        """Handle selection of a new metric from the radio buttons."""
        nonlocal current_metric
        if label != current_metric:
            temp_metric_data = _calculate_metric_and_warn(data, label)
            if temp_metric_data is not None:
                current_metric = label
                update_plots()
            else:
                warnings.warn(f"Cannot switch to invalid or problematic metric: {label}", UserWarning)

    def on_spectrum_toggle(label):
        nonlocal calculate_spectrum
        new_state = plot_elements['spec_check'].get_status()[0]
        if new_state != calculate_spectrum:
            calculate_spectrum = new_state
            update_spectrum_plot()

    def quit_callback(event):
        plt.close(fig)

    fig.canvas.mpl_connect('button_press_event', on_click)

    valid_metrics = ['var', 'std', 'mad', 'maxabs', 'ptp', 'kurtosis']

    active_metric_idx = valid_metrics.index(current_metric) if current_metric in valid_metrics else valid_metrics.index('var')
    if current_metric not in valid_metrics:
        warnings.warn(f"Initial metric '{current_metric}' is not in valid_metrics. Defaulting to 'var'.", UserWarning)
        current_metric = 'var'

    ax_metric_radio.set_title('Select Metric:', fontsize=12)
    radio = RadioButtons(ax_metric_radio, valid_metrics, active=active_metric_idx)
    for label_widget in radio.labels:
        label_widget.set_fontsize(12)
    radio.on_clicked(on_metric_select)
    plot_elements['metric_radio'] = radio

    spec_check = CheckButtons(ax_spec_check, ['Plot Spectrum'], [calculate_spectrum])
    spec_check.on_clicked(on_spectrum_toggle)
    plot_elements['spec_check'] = spec_check

    quit_button = Button(ax_quit_button, 'Quit & Return Rejected')
    quit_button.on_clicked(quit_callback)
    plot_elements['quit_button'] = quit_button

    selector_props = dict(facecolor='grey', edgecolor='black', alpha=0.3, fill=True)
    selectors['trial'] = RectangleSelector(
        ax_trial_summary, onselect_trials, useblit=True, button=[1],
        minspanx=5, minspany=5, spancoords='pixels', interactive=True, props=selector_props
    )
    selectors['channel'] = RectangleSelector(
        ax_chan_summary, onselect_channels, useblit=True, button=[1],
        minspanx=5, minspany=5, spancoords='pixels', interactive=True, props=selector_props
    )

    def prevent_selector_on_widget_click(event):
        widget_axes = [
            plot_elements['metric_radio'].ax,
            plot_elements['spec_check'].ax,
            plot_elements['quit_button'].ax,
        ]
        if any(ax.contains(event)[0] for ax in widget_axes if ax.contains(event) is not None):
            if 'trial' in selectors and selectors['trial'].active:
                selectors['trial'].set_active(False)
            if 'channel' in selectors and selectors['channel'].active:
                selectors['channel'].set_active(False)

    update_plots()

    plt.show(block=True)

    final_rejected_trials_indices = sorted(list(rejected_trials))
    final_rejected_channel_names_list = sorted(list(bad_channel_names))

    print(f"Interactive session ended.")
    print(
        f"Returning {len(final_rejected_trials_indices)} rejected trial indices and "
        f"{len(final_rejected_channel_names_list)} rejected channel names."
    )
    return final_rejected_trials_indices, final_rejected_channel_names_list


def plot_rejection_with_rejectlog(
    epochs_original: mne.BaseEpochs,
    rejected_trial_indices: list[int],
    globally_rejected_channel_names: list[str],
    log_channel_names: list[str],
) -> None:
    """Visualize epoch and channel rejections using an Autoreject RejectLog.

    Constructs a ``RejectLog`` from user-selected rejections and renders both
    epoch traces with bad segments highlighted and a horizontal rejection
    summary. Requires the ``autoreject`` package.

    Parameters
    ----------
    epochs_original : mne.BaseEpochs
        Original MNE Epochs object.
    rejected_trial_indices : list of int
        Zero-based indices of rejected epochs.
    globally_rejected_channel_names : list of str
        Channel names rejected across all epochs.
    log_channel_names : list of str
        Channel names defining the columns of the RejectLog matrix.

    Returns
    -------
    None
        Displays figures; returns nothing.
    """
    if RejectLog is None:
        print("Autoreject library is not installed. Skipping RejectLog visualization.")
        print("Please install it if you want this feature: pip install autoreject")
        return

    print("\n--- Visualizing with Autoreject RejectLog ---")
    n_total_epochs = len(epochs_original)
    if n_total_epochs == 0:
        print("No epochs in `epochs_original`. Skipping RejectLog visualization.")
        return

    bad_epochs_bool_array = np.zeros(n_total_epochs, dtype=bool)
    if rejected_trial_indices:
        valid_indices = [idx for idx in rejected_trial_indices if 0 <= idx < n_total_epochs]
        if valid_indices:
            bad_epochs_bool_array[valid_indices] = True
        else:
            print("No valid rejected trial indices provided for RejectLog.")

    labels_int_array = np.zeros((n_total_epochs, len(log_channel_names)), dtype=int)

    for epoch_idx in range(n_total_epochs):
        if bad_epochs_bool_array[epoch_idx]:
            for ch_name_global_bad in globally_rejected_channel_names:
                if ch_name_global_bad in log_channel_names:
                    try:
                        ch_idx_in_log = log_channel_names.index(ch_name_global_bad)
                        labels_int_array[epoch_idx, ch_idx_in_log] = 1
                    except ValueError:
                        pass

    try:
        reject_log_viz = RejectLog(
            bad_epochs=bad_epochs_bool_array,
            labels=labels_int_array,
            ch_names=log_channel_names
        )
        print("Successfully created RejectLog object.")

        epochs_for_log_traces = epochs_original.copy().pick(log_channel_names, verbose=False)

        if not epochs_for_log_traces.ch_names:
            print(
                f"Warning: None of the 'log_channel_names' ({log_channel_names}) "
                f"were found in 'epochs_original'. Skipping RejectLog's plot_epochs."
            )
        else:
            print("Plotting epoch traces using reject_log_viz.plot_epochs()...")
            reject_log_viz.plot_epochs(
                epochs=epochs_for_log_traces,
                scalings=dict(eeg=60e-6, meg=20e-12, ecog=60e-6, seeg=60e-6),
                title="Epoch Traces (Bad Selections Highlighted by RejectLog)"
            )

        print("Plotting channel status summary using reject_log_viz.plot()...")
        fig_summary = reject_log_viz.plot(
            orientation='horizontal',
            show=False
        )
        if fig_summary:
            fig_summary.suptitle("Channel-Epoch Rejection Summary (from RejectLog)")

        plt.show()

    except Exception as e:
        print(f"An error occurred during RejectLog processing or plotting: {e}")
        traceback.print_exc()


def plot_mne_native_with_bads(
    epochs_original: mne.BaseEpochs,
    globally_rejected_channel_names: list[str],
    rejected_trial_indices_for_context: list[int] | None = None,
    num_epochs_to_show: int = 5,
) -> None:
    """Display epochs using MNE's native plotter with bad channels highlighted.

    Parameters
    ----------
    epochs_original : mne.BaseEpochs
        Original MNE Epochs object.
    globally_rejected_channel_names : list of str
        Channel names to mark as bad in ``info['bads']``.
    rejected_trial_indices_for_context : list of int or None, optional
        Epoch indices to prioritize in the display subset. Default is ``None``.
    num_epochs_to_show : int, optional
        Maximum number of epochs to render. Default is ``5``.

    Returns
    -------
    None
    """
    print("\n--- MNE Native Plotting Demonstration with info['bads'] ---")
    if not globally_rejected_channel_names:
        print("No globally rejected channel names provided. Plotting without highlighting specific bad channels.")

    epochs_for_mne_plot = epochs_original.copy()

    original_bads = list(epochs_for_mne_plot.info['bads'])
    newly_added_bads = []
    for ch_name in globally_rejected_channel_names:
        if ch_name in epochs_for_mne_plot.ch_names:
            if ch_name not in epochs_for_mne_plot.info['bads']:
                epochs_for_mne_plot.info['bads'].append(ch_name)
                newly_added_bads.append(ch_name)
        else:
            print(
                f"Warning: Channel '{ch_name}' specified in globally_rejected_channel_names "
                "not found in epoch channel names. Cannot mark as bad."
            )

    if newly_added_bads:
        print(f"Channels added to info['bads'] for this plot: {newly_added_bads}")
    elif not globally_rejected_channel_names:
        pass
    else:
        print("No new channels were added to info['bads'] for MNE native plotting (either already bad or not found).")

    indices_to_plot = []
    if rejected_trial_indices_for_context:
        indices_to_plot.extend(
            idx for idx in rejected_trial_indices_for_context if 0 <= idx < len(epochs_original)
        )

    current_epoch_idx = 0
    while len(set(indices_to_plot)) < num_epochs_to_show and current_epoch_idx < len(epochs_original):
        if current_epoch_idx not in indices_to_plot:
            indices_to_plot.append(current_epoch_idx)
        current_epoch_idx += 1

    indices_to_plot = sorted(list(set(idx for idx in indices_to_plot if 0 <= idx < len(epochs_original))))
    indices_to_plot = indices_to_plot[:num_epochs_to_show]

    if not indices_to_plot and len(epochs_original) > 0:
        indices_to_plot = list(range(min(len(epochs_original), num_epochs_to_show)))

    if not indices_to_plot:
        print("No epochs available or selected for MNE native plotting.")
        epochs_for_mne_plot.info['bads'] = original_bads
        return

    epochs_subset_for_mne_plot = epochs_for_mne_plot[indices_to_plot]

    print(
        f"Plotting MNE epochs (indices: {indices_to_plot}) "
        f"with info['bads']: {epochs_subset_for_mne_plot.info['bads']}"
    )

    epochs_subset_for_mne_plot.plot(
        bad_color='salmon',
        title=f"MNE Native Plot (Epochs: {indices_to_plot}, info['bads'] marked)",
        n_epochs=len(epochs_subset_for_mne_plot),
        show=True,
        block=True,
    )

    epochs_for_mne_plot.info['bads'] = original_bads
