from pathlib import Path
import numpy as np
import antropy
from scipy.signal import hilbert, welch
from scipy.stats import mode

FS = 128
FEATURE_NAMES = (
    "mean", "std", "coefficient_variation", "median", "mode", "max", "min",
    "q1", "q3", "iqr", "kurtosis", "skewness", "dfa", "hjorth_activity",
    "hjorth_mobility", "hjorth_complexity", "permutation_entropy",
    "approximate_entropy", "spectral_entropy", "higuchi_fd", "total_psd",
    "centroid_psd", "determinism", "trapping_time", "diagonal_line_entropy",
    "average_diagonal_line_length", "recurrence_rate", "sef25", "sef50",
    "sef75", "hurst_exponent", "svd_entropy", "petrosian_fd", "katz_fd",
    "relative_band_power", "band_amplitude")


def _rqa(x, threshold, m=3, tau=1):
    embedded = np.column_stack([x[i:len(x) - (m - 1) * tau + i:tau]
                                 for i in range(0, m * tau, tau)])
    recurrence = np.max(np.abs(embedded[:, None] - embedded[None, :]), axis=2) <= threshold
    np.fill_diagonal(recurrence, False)
    diagonal_lengths = []
    for offset in range(1, len(x)):
        diagonal = np.diagonal(recurrence, offset)
        if len(diagonal) < 2:
            continue
        padded = np.r_[False, diagonal, False]
        starts = np.flatnonzero(~padded[:-1] & padded[1:])
        ends = np.flatnonzero(padded[:-1] & ~padded[1:])
        diagonal_lengths.extend((ends - starts).tolist())
    lengths = np.asarray([n for n in diagonal_lengths if n >= 2])
    if not len(lengths):
        return (0., 0., 0., 0., float(recurrence.mean()))
    return (float(lengths.sum() / max(recurrence.sum(), 1)),
            float(np.mean(lengths >= 2)),
            float(-np.sum((lengths / lengths.sum()) *
                          np.log(lengths / lengths.sum()))),
            float(lengths.mean()), float(recurrence.mean()))


def _hurst(x):
    if np.std(x) == 0:
        return 0.
    y = np.cumsum(x - x.mean())
    r = y.max() - y.min()
    s = np.std(x)
    return float(np.log((r / (s or 1)) + 1e-12) / np.log(len(x)))


def _feature_vector(x, band_limits=None):
    x = np.ascontiguousarray(x, dtype=np.float64)
    std = np.std(x)
    q1, q3 = np.percentile(x, (25, 75))
    f, pxx = welch(x, fs=FS, nperseg=128)
    total = np.trapezoid(pxx, f)
    cdf = np.cumsum(pxx)
    sef = lambda q: f[np.searchsorted(cdf, q * cdf[-1])] if cdf[-1] else 0.
    activity, mobility = antropy.hjorth_params(x)
    derivative_mobility = np.sqrt(np.var(np.diff(x, 2)) /
                                  (np.var(np.diff(x)) or 1e-12))
    complexity = derivative_mobility / (mobility or 1e-12)
    rqa = _rqa(x, 0.1 * std)
    band_power = 0. if band_limits is None else (
        np.trapezoid(pxx[(f >= band_limits[0]) & (f <= band_limits[1])],
                     f[(f >= band_limits[0]) & (f <= band_limits[1])])
        / (total or 1.))
    values = [
        x.mean(), std, std / (abs(x.mean()) or 1e-12), np.median(x),
        mode(x, keepdims=False).mode, x.max(), x.min(), q1, q3, q3 - q1,
        ((x - x.mean()) ** 4).mean() / (std ** 4 or 1e-12) - 3,
        ((x - x.mean()) ** 3).mean() / (std ** 3 or 1e-12),
        antropy.detrended_fluctuation(x), activity, mobility, complexity,
        antropy.perm_entropy(x),
        antropy.app_entropy(x), antropy.spectral_entropy(x, FS, method="fft",
                                                         nperseg=128),
        antropy.higuchi_fd(x), total, np.trapezoid(f * pxx, f) / (total or 1.),
        *rqa, sef(.25), sef(.5), sef(.75), _hurst(x),
        antropy.svd_entropy(x, normalize=True), antropy.petrosian_fd(x),
        antropy.katz_fd(x), band_power,
        float(np.mean(np.abs(hilbert(x)))),
    ]
    return np.nan_to_num(values, nan=0., posinf=0., neginf=0.).astype(np.float32)


def extract_features(layers, cache_path=None):
    """Return (N, 14, 214), caching exactly the requested layout."""
    if cache_path and Path(cache_path).exists():
        return np.load(cache_path)
    limits = (None, (0.5, 4), (4, 8), (8, 13), (13, 30), (30, 40))
    output = np.empty((len(layers), 14, 214), dtype=np.float32)
    for i, instance in enumerate(layers):
        columns = []
        for layer, band in zip(instance.transpose(2, 0, 1), limits):
            vectors = [_feature_vector(channel, band) for channel in layer]
            columns.append(np.stack(vectors, axis=0))
        output[i] = np.concatenate((columns[0][:, :34], *columns[1:]), axis=1)
        if (i + 1) % 25 == 0 or i + 1 == len(layers):
            print(f"[features] {i + 1}/{len(layers)} instances", flush=True)
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, output)
    return output
