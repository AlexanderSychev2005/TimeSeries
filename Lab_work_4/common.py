"""Shared signal processing and time-series analysis utilities for ECG data."""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.signal import periodogram
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss

RS_MIN_CHUNK = 8
DTW_RADIUS = 100
DATA_DIR = Path(__file__).parent / "data"
PN_DIR = "mitdb"
NON_BEAT_SYMBOLS = {"+", "~", "|", "x", "[", "!", "]"}


# Data I/O


def load_record(record: str) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Load ECG signal and annotations from local cache or fetch via PhysioNet."""
    signal_path = DATA_DIR / f"ecg_{record}_signal.csv"
    annotations_path = DATA_DIR / f"ecg_{record}_annotations.csv"
    if signal_path.exists() and annotations_path.exists():
        signal_df = pd.read_csv(signal_path)
        annotations_df = pd.read_csv(annotations_path)
        fs = round(1 / (signal_df["time_s"].iloc[1] - signal_df["time_s"].iloc[0]))
        return signal_df, annotations_df, fs

    wf_record = wfdb.rdrecord(record, pn_dir=PN_DIR)
    annotation = wfdb.rdann(record, "atr", pn_dir=PN_DIR)
    signal_df = pd.DataFrame(wf_record.p_signal, columns=wf_record.sig_name)
    signal_df.insert(0, "time_s", np.arange(len(signal_df)) / wf_record.fs)
    annotations_df = pd.DataFrame(
        {
            "sample": annotation.sample,
            "time_s": annotation.sample / annotation.fs,
            "symbol": annotation.symbol,
            "aux_note": annotation.aux_note,
        }
    )
    DATA_DIR.mkdir(exist_ok=True)
    signal_df.to_csv(signal_path, index=False)
    annotations_df.to_csv(annotations_path, index=False)
    return signal_df, annotations_df, wf_record.fs


def beat_samples_and_symbols(
    annotations_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract sample indices and annotation symbols for genuine heartbeats."""
    is_beat = ~annotations_df["symbol"].isin(NON_BEAT_SYMBOLS)
    beats = annotations_df[is_beat]
    return beats["sample"].to_numpy(), beats["symbol"].to_numpy()


def heart_rate_series(r_peak_samples: np.ndarray, fs: int) -> np.ndarray:
    """Compute instantaneous heart rate (BPM) from R-peak locations."""
    rr = np.diff(r_peak_samples) / fs
    return 60 / rr


# Time-Series Decomposition


def dominant_period(
    x: np.ndarray, fs: float, band: tuple[float, float] = (0.5, 3.0)
) -> int:
    """Find the period (in samples) of the peak spectral frequency within `band` Hz."""
    freqs, power = periodogram(x, fs=fs)
    in_band = (freqs >= band[0]) & (freqs <= band[1])
    peak_freq = freqs[in_band][np.argmax(power[in_band])]
    return max(round(fs / peak_freq), 2)


def decompose(
    x: np.ndarray, period: int
) -> tuple[pd.Series, pd.Series, pd.Series, float]:
    """Fit robust STL decomposition and return (trend, seasonal, resid, trend_r2)."""
    result = STL(x, period=period, robust=True).fit()
    ss_res = np.sum((x - result.trend) ** 2)
    ss_tot = np.sum((x - x.mean()) ** 2)
    r2_trend = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
    return result.trend, result.seasonal, result.resid, r2_trend


# Statistical Properties


def hurst_exponent(x: np.ndarray, min_chunk: int = RS_MIN_CHUNK) -> float:
    """Estimate the Hurst exponent using rescaled range (R/S) analysis."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    sizes = np.unique(
        np.logspace(np.log10(min_chunk), np.log10(n // 2), num=20).astype(int)
    )

    log_sizes, log_rs = [], []
    for size in sizes:
        n_chunks = n // size
        if n_chunks < 1:
            continue
        rs_values = []
        for i in range(n_chunks):
            chunk = x[i * size : (i + 1) * size]
            deviation = np.cumsum(chunk - chunk.mean())
            r = deviation.max() - deviation.min()
            s = chunk.std()
            if s > 0:
                rs_values.append(r / s)
        if rs_values:
            log_sizes.append(np.log(size))
            log_rs.append(np.log(np.mean(rs_values)))

    slope, _ = np.polyfit(log_sizes, log_rs, 1)
    return float(slope)


def properties(x: np.ndarray, label: str) -> dict:
    """Calculate summary statistics, stationarity tests (ADF, KPSS), and Hurst exponent."""
    mean, var, std = (
        float(np.mean(x)),
        float(np.var(x, ddof=1)),
        float(np.std(x, ddof=1)),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adf_stat, adf_p, *_ = adfuller(x)
        kpss_stat, kpss_p, *_ = kpss(x, nlags="auto")

    hurst = hurst_exponent(x)

    print(f"Properties: {label}")
    print(f"mean={mean:.4f}, var={var:.4f}, std={std:.4f}")
    print(
        f"ADF: stat={adf_stat:.3f}, p={adf_p:.4f} -> {'stationary' if adf_p < 0.05 else 'non-stationary'}"
    )
    print(
        f"KPSS: stat={kpss_stat:.3f}, p={kpss_p:.4f} -> {'non-stationary' if kpss_p < 0.05 else 'stationary'}"
    )
    print(f"Hurst exponent H={hurst:.4f}")

    return {
        "mean": mean,
        "var": var,
        "std": std,
        "adf_stat": adf_stat,
        "adf_p": adf_p,
        "kpss_stat": kpss_stat,
        "kpss_p": kpss_p,
        "hurst": hurst,
    }


# Similarity


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Euclidean distance truncated to the shorter sequence length."""
    n = min(len(a), len(b))
    return float(np.linalg.norm(a[:n] - b[:n]))


def dtw_distance(a: np.ndarray, b: np.ndarray, radius: int = DTW_RADIUS) -> float:
    """Compute Dynamic Time Warping distance with Sakoe-Chiba band constraint."""
    n, m = len(a), len(b)
    d = np.full((n + 1, m + 1), np.inf)
    d[0, 0] = 0.0
    for i in range(1, n + 1):
        lo = max(1, i - radius)
        hi = min(m, i + radius)
        for j in range(lo, hi + 1):
            cost = abs(a[i - 1] - b[j - 1])
            d[i, j] = cost + min(d[i - 1, j], d[i, j - 1], d[i - 1, j - 1])
    return float(d[n, m])


# Clustering


def extract_beats(
    signal: np.ndarray, r_peak_samples: np.ndarray, half_window: int
) -> tuple[np.ndarray, np.ndarray]:
    """Slice fixed-size windows around R-peaks, dropping boundary beats."""
    n = len(signal)
    keep_mask = (r_peak_samples - half_window >= 0) & (r_peak_samples + half_window < n)
    kept = r_peak_samples[keep_mask]
    windows = np.stack([signal[s - half_window : s + half_window] for s in kept])
    return windows, keep_mask


def cluster_classic(
    beat_windows: np.ndarray, n_clusters: int = 3, seed: int = 0
) -> np.ndarray:
    """Cluster raw beat waveforms using K-Means."""
    return KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit_predict(
        beat_windows
    )


def beat_features(window: np.ndarray) -> list[float]:
    """Extract morphology summary statistics [max, min, std, absolute sum] from a beat."""
    return [
        float(window.max()),
        float(window.min()),
        float(window.std()),
        float(np.abs(window).sum()),
    ]


def cluster_multifactor(
    feature_matrix: np.ndarray, n_clusters: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """Hierarchical agglomerative clustering using Ward linkage."""
    z = linkage(feature_matrix, method="ward")
    labels = fcluster(z, t=n_clusters, criterion="maxclust")
    return labels, z


def validate_clusters(labels: np.ndarray, true_symbols: np.ndarray) -> float:
    """Calculate Adjusted Rand Index (ARI) against ground-truth annotations."""
    return float(adjusted_rand_score(true_symbols, labels))


# Correlation


def correlate(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Pearson correlation coefficient between two signals."""
    n = min(len(a), len(b))
    return float(np.corrcoef(a[:n], b[:n])[0, 1])


def rolling_correlation(a: np.ndarray, b: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling window Pearson correlation between two signals."""
    n = min(len(a), len(b))
    return pd.Series(a[:n]).rolling(window).corr(pd.Series(b[:n])).to_numpy()


# Visualization


def plot_decomposition(x, trend, seasonal, resid, path: Path, title: str) -> None:
    _fig, axes = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
    for ax, data, name in zip(
        axes, [x, trend, seasonal, resid], ["Signal", "Trend", "Seasonal", "Residual"]
    ):
        ax.plot(data, linewidth=0.8)
        ax.set_ylabel(name)
    axes[0].set_title(title)
    axes[-1].set_xlabel("Sample")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_correlation_heatmap(
    matrix: np.ndarray, labels: list[str], path: Path, title: str
) -> None:
    plt.figure(figsize=(4.5, 4))
    im = plt.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
    plt.colorbar(im, label="Pearson r")
    plt.xticks(range(len(labels)), labels)
    plt.yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_rolling_correlation(corr: np.ndarray, path: Path, title: str) -> None:
    plt.figure(figsize=(10, 3.5))
    plt.plot(corr, linewidth=1)
    plt.axhline(0, color="gray", linestyle=":", linewidth=1)
    plt.ylabel("Rolling correlation")
    plt.xlabel("Sample")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_dendrogram(z: np.ndarray, path: Path, title: str) -> None:
    plt.figure(figsize=(9, 4.5))
    dendrogram(z, truncate_mode="lastp", p=30)
    plt.title(title)
    plt.xlabel("Cluster (or beat index)")
    plt.ylabel("Ward distance")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_cluster_examples(
    beat_windows: np.ndarray, labels: np.ndarray, path: Path, title: str
) -> None:
    clusters = np.unique(labels)
    fig, axes = plt.subplots(
        1, len(clusters), figsize=(4 * len(clusters), 3.5), sharey=True
    )
    if len(clusters) == 1:
        axes = [axes]
    for ax, c in zip(axes, clusters):
        members = beat_windows[labels == c]
        for w in members[:: max(1, len(members) // 40)]:
            ax.plot(w, color="steelblue", alpha=0.3, linewidth=0.7)
        ax.plot(members.mean(axis=0), color="crimson", linewidth=2)
        ax.set_title(f"Cluster {c} (n={len(members)})")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
