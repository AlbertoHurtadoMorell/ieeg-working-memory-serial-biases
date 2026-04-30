"""iEEG spectral analysis, circular-linear correlation, permutation testing,
decoding, and anatomical visualisation utilities.
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import mne
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import plotting
from nitime import algorithms as tsa
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR

try:
    from base_stats import corr_linear_circular
    from classifier_funcs import AngularRegression
    _DECODING_TOOLBOX_AVAILABLE = True
except ImportError:
    _DECODING_TOOLBOX_AVAILABLE = False
    corr_linear_circular = None
    AngularRegression = None


def circ_corr_fun(Y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Compute trial-wise linear-circular correlation for multi-channel signals.

    Parameters
    ----------
    Y : np.ndarray
        Shape ``(n_trials, n_channels, n_timepoints)`` — recorded signals.
    X : np.ndarray
        Shape ``(n_trials,)`` — stimulus orientations in radians.

    Returns
    -------
    np.ndarray
        Shape ``(n_channels, n_timepoints)`` — coefficient of determination
        for each channel and timepoint.

    Notes
    -----
    Requires ``corr_linear_circular`` from the ``decoding_toolbox``.
    Install from: https://github.com/alepebel/decoding_toolbox
    Then add its ``Helper_funcs/`` directory to ``sys.path``.
    """
    if not _DECODING_TOOLBOX_AVAILABLE:
        raise ImportError(
            "decoding_toolbox is required for circ_corr_fun. "
            "Install from: https://github.com/alepebel/decoding_toolbox\n"
            "Then add its Helper_funcs/ directory to sys.path."
        )
    n_trial, n_chan, n_time = Y.shape
    Yreshape = Y.reshape([-1, n_chan * n_time])
    _, R2, _ = corr_linear_circular(Yreshape, X)
    R2 = R2.reshape([n_chan, n_time])
    return R2


def get_freq_spec(
    data: np.ndarray,
    fs: float,
    batch_size: int = 100,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Estimate multitaper power spectral density in batches.

    Parameters
    ----------
    data : np.ndarray
        Shape ``(n_trials, n_channels, n_timepoints)``.
    fs : float
        Sampling frequency in Hz.
    batch_size : int, optional
        Number of trials per processing batch to limit memory usage.
        Default is ``100``.

    Returns
    -------
    psd : np.ndarray
        Shape ``(n_trials, n_channels, n_frequencies)``.
    f : np.ndarray
        Frequency values in Hz, shape ``(n_frequencies,)``.
    nu : int
        Degrees of freedom of the multitaper estimate.
    """
    ntrials = data.shape[0]
    psd_mt_list = []

    for i in np.arange(0, ntrials, batch_size):
        data_slice = data[i:i + batch_size, :, :]
        print(f'Processing batch starting at trial {i}')
        f, psd_sl, nu = tsa.multi_taper_psd(
            data_slice, Fs=fs, adaptive=False, jackknife=False
        )
        psd_mt_list.append(psd_sl)

    psd = np.concatenate(psd_mt_list, axis=0)
    return psd, f, nu


def norm_freq(
    psd: np.ndarray,
    f: np.ndarray,
    fband: tuple[float, float],
) -> np.ndarray:
    """Compute log-ratio of high-frequency band power to total power.

    Parameters
    ----------
    psd : np.ndarray
        Shape ``(n_trials, n_channels, n_frequencies)``.
    f : np.ndarray
        Frequency values in Hz, shape ``(n_frequencies,)``.
    fband : tuple of float
        ``(low, high)`` in Hz defining the high-frequency band of interest.

    Returns
    -------
    np.ndarray
        Shape ``(n_trials, n_channels)`` — log10-transformed band-power
        ratio (band power / total power, 5–125 Hz).
    """
    low, high = fband
    idx_HG = np.logical_and(f >= low, f <= high)
    idx_tot = np.logical_and(f >= 5, f <= 125)

    HG_power = np.sum(psd[:, :, idx_HG], axis=-1)
    total_power = np.sum(psd[:, :, idx_tot], axis=-1)

    epsilon = 1e-10
    return np.log10((HG_power + epsilon) / (total_power + epsilon))


def compute_null_psd(
    epochs: mne.BaseEpochs,
    angles_rad: np.ndarray,
    freq_band: tuple[float, float],
    get_freq_spec,
    norm_freq,
    nperm: int = 200,
    batch_size: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute observed and permutation-null PSD-angle correlations.

    Parameters
    ----------
    epochs : mne.BaseEpochs
        Segmented data, shape ``(n_trials, n_channels, n_timepoints)``.
    angles_rad : np.ndarray
        Trial-wise stimulus orientations in radians, shape ``(n_trials,)``.
    freq_band : tuple of float
        ``(low, high)`` in Hz for the normalization band.
    get_freq_spec : callable
        Function returning ``(psd, f, nu)`` for a data array.
    norm_freq : callable
        Function returning log-ratio band-power values.
    nperm : int, optional
        Number of label permutations for the null distribution. Default ``200``.
    batch_size : int, optional
        Trials per batch for PSD estimation. Default ``100``.

    Returns
    -------
    R_true : np.ndarray
        Shape ``(n_channels,)`` — observed correlation per channel.
    R_null : np.ndarray
        Shape ``(nperm, n_channels)`` — correlations under permuted labels.
    """
    X_time = epochs.get_data()

    psd_mt, freqs, nu = get_freq_spec(X_time, epochs.info['sfreq'], batch_size=batch_size)

    X_rel = norm_freq(psd_mt, freqs, fband=freq_band)

    X_corr = X_rel[..., np.newaxis]

    R2_true = circ_corr_fun(X_corr, angles_rad)
    R_true = R2_true[:, 0]

    nch = len(epochs.ch_names)
    R_null = np.zeros((nperm, nch), dtype=float)
    for i in range(nperm):
        y_perm = np.random.permutation(angles_rad)
        R2p = circ_corr_fun(X_corr, y_perm)
        R_null[i, :] = R2p[:, 0]

    return R_true, R_null


def channel_significance_by_window_psd(
    epochs_dict: dict,
    electrode_df: pd.DataFrame,
    time_windows: dict,
    freq_band: tuple[float, float] = (2, 12),
    alpha: float = 0.05,
    nperm: int = 200,
    batch_size: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Test channel PSD-angle correlations across time windows.

    For each participant and time window, computes the permutation-based
    significance of PSD–stimulus-angle circular correlation, then annotates
    significant channels with anatomical region labels.

    Parameters
    ----------
    epochs_dict : dict
        Maps participant ID to MNE Epochs with a ``T_Angle`` metadata column.
    electrode_df : pd.DataFrame
        Must contain columns ``'participant'``, ``'Bipolar_Label'``,
        ``'FSLabel'``.
    time_windows : dict
        Maps window label (str) to ``(tmin, tmax)`` in seconds.
    freq_band : tuple of float, optional
        ``(low, high)`` in Hz. Default is ``(2, 12)``.
    alpha : float, optional
        Significance level. Default is ``0.05``.
    nperm : int, optional
        Number of permutations. Default is ``200``.
    batch_size : int, optional
        Trials per PSD batch. Default is ``100``.

    Returns
    -------
    signif_df : pd.DataFrame
        Columns: ``'participant'``, ``'channel'``, ``'window'``, ``'pval'``.
    merged : pd.DataFrame
        *signif_df* joined with anatomical region labels (column ``'region'``).
    region_counts : pd.DataFrame
        Counts of significant channels per region, sorted descending.
    """
    records = []
    for subj, ep in epochs_dict.items():
        angles = np.deg2rad(ep.metadata['T_Angle'].values)
        for win_name, (tmin, tmax) in time_windows.items():
            ep_win = ep.copy().crop(tmin, tmax)

            R_true, R_null = compute_null_psd(
                ep_win, angles,
                freq_band=freq_band,
                get_freq_spec=get_freq_spec,
                norm_freq=norm_freq,
                nperm=nperm,
                batch_size=batch_size,
            )

            pvals = np.mean(R_true[np.newaxis, :] < R_null, axis=0)

            for idx, ch in enumerate(ep.ch_names):
                if pvals[idx] < alpha:
                    records.append({
                        'participant': subj,
                        'channel': ch,
                        'window': win_name,
                        'pval': float(pvals[idx]),
                    })

    signif_df = pd.DataFrame.from_records(records)

    merged = signif_df.merge(
        electrode_df[['participant', 'Bipolar_Label', 'FSLabel']],
        left_on=['participant', 'channel'],
        right_on=['participant', 'Bipolar_Label'],
        how='left',
    )
    merged['region'] = (
        merged['FSLabel']
        .str.replace(r'^Mixed:', '', regex=True)
        .str.split(r'\|')
    ).explode('region')

    bad = merged['region'].str.contains('Unknown|White-Matter|WM', na=False)
    merged = merged.loc[~bad, ['participant', 'channel', 'window', 'pval', 'region']]

    region_counts = (
        merged.groupby('region')
        .size()
        .reset_index(name='count')
        .sort_values('count', ascending=False)
        .reset_index(drop=True)
    )

    return signif_df, merged, region_counts


def plot_period_regions(
    top_regions_df: pd.DataFrame,
    period_name: str,
    label_names_dict: dict,
    atlas_data: np.ndarray,
    img,
    cmap: str = 'Reds',
) -> None:
    """Visualize top anatomical regions as a binary mask on a glass-brain.

    Parameters
    ----------
    top_regions_df : pd.DataFrame
        Must contain a ``'Count'`` column with region label name strings.
    period_name : str
        Title label describing the epoch or time period.
    label_names_dict : dict
        Maps integer label IDs to anatomical label strings.
    atlas_data : np.ndarray
        Volumetric array containing integer atlas label IDs, same shape as
        the image data of *img*.
    img : nibabel image
        Nibabel image object providing affine and header for the output image.
    cmap : str, optional
        Matplotlib colormap name. Default is ``'Reds'``.

    Returns
    -------
    None
    """
    label_ids = []
    for region in top_regions_df['Count']:
        for label_id, label_name in label_names_dict.items():
            if region in label_name:
                label_ids.append(label_id)

    roi_data = (np.isin(atlas_data, label_ids)).astype(np.uint8)

    roi_img = nib.MGHImage(roi_data, img.affine, header=img.header)

    fig = plt.figure(figsize=(6, 4))
    plotting.plot_glass_brain(
        roi_img,
        title=f'Active regions: {period_name}',
        cmap=cmap,
        colorbar=True,
        threshold=0.5,
        display_mode='lyrz',
        figure=fig,
    )
    plt.show()


def top_regions(df_locs: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Return the most frequent anatomical regions from a significance table.

    Parameters
    ----------
    df_locs : pd.DataFrame
        Must contain a ``'region'`` column with anatomical label strings.
    top_n : int, optional
        Number of top regions to return. Default is ``5``.

    Returns
    -------
    pd.DataFrame
        Columns ``'Region'`` and ``'Count'`` for the *top_n* most frequent
        labels, excluding ``'Unknown'``.
    """
    filtered = df_locs[df_locs['region'] != 'Unknown']
    region_counts = filtered['region'].value_counts().nlargest(top_n)
    return region_counts.reset_index().rename(columns={'index': 'Region', 'region': 'Count'})


def plot_period_heatmap(
    top_regions_df: pd.DataFrame,
    period_name: str,
    label_names_dict: dict,
    atlas_data: np.ndarray,
    img,
) -> None:
    """Render a continuous-intensity heatmap of region counts on a glass-brain.

    Parameters
    ----------
    top_regions_df : pd.DataFrame
        Must contain columns ``'Count'`` (region name strings) and
        ``'count'`` (integer occurrence counts).
    period_name : str
        Title label describing the epoch or time period.
    label_names_dict : dict
        Maps integer label IDs to anatomical label strings.
    atlas_data : np.ndarray
        Volumetric array containing integer atlas label IDs.
    img : nibabel image
        Nibabel image providing affine and header.

    Returns
    -------
    None
    """
    roi_data = np.zeros_like(atlas_data, dtype=np.float32)

    for _, row in top_regions_df.iterrows():
        region_name = row['Count']
        region_count = row['count']

        for label_id, label_name in label_names_dict.items():
            if region_name in label_name:
                roi_data[atlas_data == label_id] = region_count

    roi_img = nib.MGHImage(roi_data, img.affine, header=img.header)

    fig = plt.figure(figsize=(8, 5))
    plotting.plot_glass_brain(
        roi_img,
        cmap='viridis',
        colorbar=True,
        vmin=0,
        vmax=roi_data.max(),
        display_mode='lyrz',
        figure=fig,
    )
    plt.show()


def channel_significance_by_window_psd_preT(
    epochs_dict: dict,
    electrode_df: pd.DataFrame,
    time_windows: dict,
    freq_band: tuple[float, float] = (2, 12),
    alpha: float = 0.05,
    nperm: int = 200,
    batch_size: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Test channel PSD correlations using pre-stimulus orientation metadata.

    Identical to :func:`channel_significance_by_window_psd` but reads
    stimulus angle from the ``'preT'`` metadata column, filtering trials
    where ``preT`` is ``NaN``.

    Parameters
    ----------
    epochs_dict : dict
        Maps participant ID to MNE Epochs with a ``preT`` metadata column.
    electrode_df : pd.DataFrame
        Must contain ``'participant'``, ``'Bipolar_Label'``, ``'FSLabel'``.
    time_windows : dict
        Maps window label to ``(tmin, tmax)`` in seconds.
    freq_band : tuple of float, optional
        Default is ``(2, 12)``.
    alpha : float, optional
        Default is ``0.05``.
    nperm : int, optional
        Default is ``200``.
    batch_size : int, optional
        Default is ``100``.

    Returns
    -------
    signif_df : pd.DataFrame
    merged : pd.DataFrame
    region_counts : pd.DataFrame
    """
    records = []
    for subj, ep in epochs_dict.items():
        for win_name, (tmin, tmax) in time_windows.items():
            ep_win = ep.copy().crop(tmin, tmax)

            angles = ep_win.metadata['preT'].values
            valid_trials = ~np.isnan(angles)
            angles_valid = np.deg2rad(angles[valid_trials])
            ep_win_valid = ep_win[valid_trials]

            if len(angles_valid) == 0:
                continue

            R_true, R_null = compute_null_psd(
                ep_win_valid, angles_valid,
                freq_band=freq_band,
                get_freq_spec=get_freq_spec,
                norm_freq=norm_freq,
                nperm=nperm,
                batch_size=batch_size,
            )

            pvals = np.mean(R_true[np.newaxis, :] < R_null, axis=0)

            for idx, ch in enumerate(ep_win_valid.ch_names):
                if pvals[idx] < alpha:
                    records.append({
                        'participant': subj,
                        'channel': ch,
                        'window': win_name,
                        'pval': float(pvals[idx]),
                    })

    signif_df = pd.DataFrame.from_records(records)

    merged = signif_df.merge(
        electrode_df[['participant', 'Bipolar_Label', 'FSLabel']],
        left_on=['participant', 'channel'],
        right_on=['participant', 'Bipolar_Label'],
        how='left',
    )
    merged['region'] = (
        merged['FSLabel']
        .str.replace(r'^Mixed:', '', regex=True)
        .str.split(r'\|')
    ).explode('region')

    bad = merged['region'].str.contains('Unknown|White-Matter|WM', na=False)
    merged = merged.loc[~bad, ['participant', 'channel', 'window', 'pval', 'region']]

    region_counts = (
        merged.groupby('region')
        .size()
        .reset_index(name='count')
        .sort_values('count', ascending=False)
        .reset_index(drop=True)
    )

    return signif_df, merged, region_counts


def decode_with_psd_permutation(
    epochs: mne.BaseEpochs,
    angles_deg: np.ndarray,
    freq_band: tuple[float, float] = (70, 150),
    nfold: int = 5,
    nperm: int = 200,
    random_state: int = 42,
) -> tuple[float, np.ndarray, float]:
    """Decode stimulus orientation from PSD features with permutation testing.

    Extracts multitaper PSD from the epochs, normalises to the specified
    frequency band, and fits a K-fold cross-validated SVR pipeline. Generates
    a permutation null distribution to assess decoding significance.

    Parameters
    ----------
    epochs : mne.BaseEpochs
        Segmented data providing raw time series for PSD extraction.
    angles_deg : np.ndarray
        Shape ``(n_trials,)`` — stimulus orientations in degrees.
    freq_band : tuple of float, optional
        ``(low, high)`` in Hz for the normalization band. Default ``(70, 150)``.
    nfold : int, optional
        Number of cross-validation folds. Default is ``5``.
    nperm : int, optional
        Number of label permutations for significance testing. Default ``200``.
    random_state : int, optional
        Random seed for reproducible splits and permutations. Default ``42``.

    Returns
    -------
    real_error : float
        Mean absolute angular decoding error in degrees for true labels.
    perm_errors : np.ndarray
        Shape ``(nperm,)`` — decoding errors under permuted labels.
    p_value : float
        Proportion of permuted errors less than or equal to *real_error*.

    Notes
    -----
    Requires ``AngularRegression`` from the ``decoding_toolbox``.
    Install from: https://github.com/alepebel/decoding_toolbox
    """
    if not _DECODING_TOOLBOX_AVAILABLE:
        raise ImportError(
            "decoding_toolbox is required for decode_with_psd_permutation. "
            "Install from: https://github.com/alepebel/decoding_toolbox\n"
            "Then add its Helper_funcs/ directory to sys.path."
        )

    X_raw = epochs.get_data()
    psd_mt, freqs, _ = get_freq_spec(X_raw, epochs.info['sfreq'])
    X_rel = norm_freq(psd_mt, freqs, fband=freq_band)

    angles_rad = np.deg2rad(angles_deg)

    kf = KFold(n_splits=nfold, shuffle=True, random_state=random_state)

    def one_decoding(X, y_rad):
        errs = []
        for train_ix, test_ix in kf.split(X):
            scaler = StandardScaler().fit(X[train_ix])
            X_train = scaler.transform(X[train_ix])
            X_test = scaler.transform(X[test_ix])

            clf = make_pipeline(AngularRegression(clf=LinearSVR()))
            clf.fit(X_train, y_rad[train_ix])
            preds_rad = clf.predict(X_test)

            err_rad = np.angle(np.exp(1j * (preds_rad - y_rad[test_ix])))
            errs.append(np.mean(np.abs(np.rad2deg(err_rad))))
        return np.mean(errs)

    real_error = one_decoding(X_rel, angles_rad)

    perm_errors = np.zeros(nperm)
    rng = np.random.RandomState(random_state)
    for i in range(nperm):
        y_perm = rng.permutation(angles_rad)
        perm_errors[i] = one_decoding(X_rel, y_perm)

    p_value = np.mean(perm_errors <= real_error)

    return real_error, perm_errors, p_value


def shrinkage_gamma(
    X: np.ndarray,
    mem_eff: bool = False,
    feedback: bool = True,
) -> float:
    """Compute the Ledoit-Wolf optimal covariance shrinkage coefficient.

    Estimates the shrinkage intensity *gamma* to regularise the sample
    covariance matrix of *X* toward a scaled identity (diagonal target).

    Parameters
    ----------
    X : np.ndarray
        Shape ``(n_features, n_samples)`` — data matrix whose covariance is
        to be regularised.
    mem_eff : bool, optional
        If ``False`` (default), uses a full 3-D outer-product computation
        (faster but higher memory). If ``True``, computes variances and
        cross-products iteratively to reduce peak memory usage.
    feedback : bool, optional
        If ``True`` and ``mem_eff=True``, prints progress every 1000
        off-diagonal elements. Default is ``True``.

    Returns
    -------
    float
        Shrinkage coefficient *gamma* in ``[0, 1]``. A value of ``0`` means
        no regularisation; ``1`` means full shrinkage to the diagonal.
    """
    num_f, num_n = X.shape

    if not mem_eff:
        m = np.mean(X, axis=1, keepdims=True)
        S = np.cov(X, rowvar=True)
        nu = np.trace(S) / num_f

        z = np.zeros((num_f, num_f, num_n))
        for n in range(num_n):
            x_centered = X[:, n:n + 1] - m
            z[:, :, n] = x_centered @ x_centered.T

        numerator = (num_n / ((num_n - 1) ** 2)) * np.sum(np.var(z, axis=2))
        denominator = np.sum((S - nu * np.eye(num_f)) ** 2)
        gamma = numerator / denominator

    else:
        X = X - np.mean(X, axis=1, keepdims=True)

        sum_var_diag = 0
        diag_s = np.zeros(num_f)

        for i_f in range(num_f):
            s = X[i_f, :] ** 2
            diag_s[i_f] = np.sum(s) / (num_n - 1)
            s_centered = s - np.mean(s)
            sum_var_diag += np.sum(s_centered ** 2) / (num_n - 1)

        nu = np.mean(diag_s)
        diag_s = diag_s - nu
        sum_s_diag = np.sum(diag_s ** 2)

        sum_s = 0
        sum_var = 0

        total_elements = (num_f - 1) * num_f // 2
        if feedback:
            print(f"Processing {total_elements} off-diagonal elements...")

        counter = 0
        for i_f1 in range(1, num_f):
            for i_f2 in range(i_f1):
                s = X[i_f1, :] * X[i_f2, :]
                sum_s += (np.sum(s) / (num_n - 1)) ** 2
                s_centered = s - np.mean(s)
                sum_var += np.sum(s_centered ** 2) / (num_n - 1)

                counter += 1
                if feedback and counter % 1000 == 0:
                    p_done = counter / total_elements
                    print(f"Progress: {p_done * 100:.2f}% complete")

        sum_s = sum_s * 2 + sum_s_diag
        sum_var = sum_var * 2 + sum_var_diag
        gamma = (num_n / ((num_n - 1) ** 2)) * sum_var / sum_s

    return gamma
