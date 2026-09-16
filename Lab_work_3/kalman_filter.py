"""
Recurrent smoothing of ECG heart-rate series (MIT-BIH record 200) using alpha-beta filters.

Evaluates filter divergence during acute arrhythmia episodes (VT, bigeminy):
1. Growing-memory (ABF / ABGF): Recursive LSM formulation. Gains alpha(k), beta(k) -> 0,
   causing severe tracking divergence during rapid HR transitions.
2. Limited-memory (Capped ABF): Freezes gain decay at k=MEMORY_CAP to maintain responsiveness.
3. Adaptive (CUSUM-reset ABF): Monitors normalized innovation residuals via CUSUM;
   resets effective memory (k=1) upon detecting regime changes to immediately re-track.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_FILE = Path(__file__).parent / "data" / "ecg_200_heart_rate.csv"
ANNOTATIONS_FILE = Path(__file__).parent / "data" / "ecg_200_annotations.csv"
OUT_DIR = Path(__file__).parent

MEMORY_CAP = 50  # beats; caps the growing-memory gain decay
ADAPTIVE_WINDOW = 30  # beats of innovation history used for the CUSUM's noise scale
ADAPTIVE_C = 5.0  # CUSUM alarm threshold, in robust-scale units (standard choice)
ADAPTIVE_REFRACTORY = 5  # beats to let the filter settle before it can reset again
ANOMALY_WINDOW = 15
ANOMALY_SIGMA = 3.0


def load_heart_rate() -> pd.DataFrame:
    return pd.read_csv(DATA_FILE)


def load_annotations() -> pd.DataFrame:
    return pd.read_csv(ANNOTATIONS_FILE)


def find_rhythm_windows(annotations: pd.DataFrame) -> list[tuple[str, float, float]]:
    """Rhythm-change markers (symbol '+') split the record into labeled
    segments, e.g. ('N', t0, t1), ('B', t1, t2), ('VT', t2, t3), ..."""
    marks = annotations[annotations["symbol"] == "+"].reset_index(drop=True)
    labels = marks["aux_note"].str.strip("\x00").str.lstrip("(")
    starts = marks["time_s"].to_numpy()
    ends = np.append(starts[1:], np.inf)
    return list(zip(labels, starts, ends))


def abnormal_rhythm_mask(
    times: np.ndarray, windows: list[tuple[str, float, float]]
) -> np.ndarray:
    """True where the beat falls inside a labeled non-normal rhythm segment
    (bigeminy / VT / ...) - a genuine regime change, not a measurement glitch."""
    mask = np.zeros(len(times), dtype=bool)
    for label, start, end in windows:
        if label != "N":
            mask |= (times >= start) & (times < end)
    return mask


def detect_isolated_anomalies(
    y: np.ndarray,
    times: np.ndarray,
    windows: list[tuple[str, float, float]],
    window: int = ANOMALY_WINDOW,
    k: float = ANOMALY_SIGMA,
) -> tuple[np.ndarray, np.ndarray]:
    """3-sigma deviation from a local rolling median, restricted to beats
    outside any labeled rhythm episode - a real regime change must not be
    smoothed away, only genuine single-beat glitches are."""
    baseline = (
        pd.Series(y).rolling(window, center=True, min_periods=1).median().to_numpy()
    )
    residual = y - baseline
    is_outlier = np.abs(residual) > k * residual.std()
    mask = is_outlier & ~abnormal_rhythm_mask(times, windows)
    return mask, baseline


def compensate_anomalies(
    y: np.ndarray, mask: np.ndarray, baseline: np.ndarray
) -> np.ndarray:
    y_clean = y.copy()
    y_clean[mask] = baseline[mask]
    return y_clean


def alfa_beta_coeffs(k: int) -> tuple[float, float]:
    """Growing-memory (expanding window) alfa-beta: equivalent to a recursive
    LSM line fit over all k measurements seen so far."""
    return 2 * (2 * k - 1) / (k * (k + 1)), 6 / (k * (k + 1))


def alfa_beta_gamma_coeffs(k: int) -> tuple[float, float, float]:
    """Growing-memory alfa-beta-gamma: recursive LSM quadratic (adds an
    acceleration state), captures curvature that a line-only model misses."""
    denom = k * (k + 1) * (k + 2)
    alpha = 3 * (3 * k**2 - 3 * k + 2) / denom
    beta = 18 * (2 * k - 1) / denom
    gamma = 60 / denom
    return alpha, beta, gamma


def run_ab_filter(y: np.ndarray, coeff_fn, t: float = 1.0) -> np.ndarray:
    n = len(y)
    x_hat = np.empty(n)
    x_hat[0] = y[0]
    v_hat = (y[1] - y[0]) / t if n > 1 else 0.0
    for k in range(1, n):
        x_pred = x_hat[k - 1] + v_hat * t
        alpha, beta = coeff_fn(k)
        innovation = y[k] - x_pred
        x_hat[k] = x_pred + alpha * innovation
        v_hat = v_hat + (beta / t) * innovation
    return x_hat


def run_abg_filter(y: np.ndarray, coeff_fn, t: float = 1.0) -> np.ndarray:
    n = len(y)
    x_hat = np.empty(n)
    x_hat[0] = y[0]
    v_hat = (y[1] - y[0]) / t if n > 1 else 0.0
    a_hat = 0.0
    for k in range(1, n):
        x_pred = x_hat[k - 1] + v_hat * t + 0.5 * a_hat * t**2
        v_pred = v_hat + a_hat * t
        alpha, beta, gamma = coeff_fn(k)
        innovation = y[k] - x_pred
        x_hat[k] = x_pred + alpha * innovation
        v_hat = v_pred + (beta / t) * innovation
        a_hat = a_hat + (2 * gamma / t**2) * innovation
    return x_hat


def capped(coeff_fn, memory: int = MEMORY_CAP):
    """Freeze the growing-memory gain at its value for k=memory - the filter
    keeps a limited (not ever-growing) effective memory, so alpha/beta never
    decay away and the filter stays responsive for the whole record."""
    return lambda k: coeff_fn(min(k, memory))


def adaptive_reset_ab_filter(
    y: np.ndarray,
    memory: int = MEMORY_CAP,
    window: int = ADAPTIVE_WINDOW,
    cusum_threshold: float = ADAPTIVE_C,
    refractory: int = ADAPTIVE_REFRACTORY,
    t: float = 1.0,
) -> np.ndarray:
    """Limited-memory alfa-beta, plus a CUSUM change-point test on the
    innovations (normalized by a robust MAD scale of the last `window`
    innovations - self-calibrating, not tuned to this record). When the
    cumulative sum flags a regime change, the effective memory k is reset to
    1: the filter "forgets" its slow, decayed gain and starts re-learning at
    alpha=1 (full trust in the new measurement), same as a fresh start. A
    short refractory period after each reset stops back-to-back resets on a
    string of noisy beats (e.g. ectopic beats) from letting the velocity
    state run away."""
    n = len(y)
    x_hat = np.empty(n)
    x_hat[0] = y[0]
    v_hat = (y[1] - y[0]) / t if n > 1 else 0.0
    innovations: list[float] = []
    cusum_pos = cusum_neg = 0.0
    k_eff = 1
    cooldown = 0
    for k in range(1, n):
        x_pred = x_hat[k - 1] + v_hat * t
        alpha, beta = alfa_beta_coeffs(k_eff)
        innovation = y[k] - x_pred

        if len(innovations) >= 10:
            recent = np.array(innovations[-window:])
            scale = max(np.median(np.abs(recent - np.median(recent))) / 0.6745, 1e-6)
        else:
            scale = max(np.std(y[: max(k, 2)]), 1e-6)
        z = innovation / scale
        cusum_pos = max(0.0, cusum_pos + z - 0.5)
        cusum_neg = max(0.0, cusum_neg - z - 0.5)

        x_hat[k] = x_pred + alpha * innovation
        v_hat = np.clip(v_hat + (beta / t) * innovation, -100.0, 100.0)
        innovations.append(innovation)

        if cooldown > 0:
            cooldown -= 1
            k_eff = min(k_eff + 1, memory)
        elif max(cusum_pos, cusum_neg) > cusum_threshold:
            k_eff, cusum_pos, cusum_neg, cooldown = 1, 0.0, 0.0, refractory
        else:
            k_eff = min(k_eff + 1, memory)
    return x_hat


def unbiasedness_check(
    n: int = 400, n_trials: int = 200, noise_std: float = 8.0, seed: int = 0
) -> float:
    """Monte-Carlo test on synthetic data with a known baseline: an unbiased
    filter's tracking error should average out to ~0, not be systematically
    positive or negative."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    true = 80 + 0.02 * t
    errors = []
    for _ in range(n_trials):
        y = true + rng.normal(0, noise_std, n)
        x_hat = run_ab_filter(y, capped(alfa_beta_coeffs))
        errors.append((x_hat[50:] - true[50:]).mean())
    return float(np.mean(errors))


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def rmse_by_segment(
    y: np.ndarray, y_hat: np.ndarray, mask: np.ndarray
) -> tuple[float, float]:
    """RMSE of the filter against the raw measurement, split into the
    annotated VT/bigeminy windows vs the rest - a lagging filter shows a much
    higher residual specifically where the rhythm changes."""
    return rmse(y[mask], y_hat[mask]), rmse(y[~mask], y_hat[~mask])


def plot_filters(
    times, y, filters: dict[str, np.ndarray], abnormal_mask, path: Path
) -> None:
    plt.figure(figsize=(12, 5))
    plt.fill_between(
        times,
        0,
        1,
        where=abnormal_mask,
        transform=plt.gca().get_xaxis_transform(),
        color="lightgrey",
        alpha=0.6,
        label="Bigeminy / VT episode",
    )
    plt.plot(times, y, label="Measured HR", linewidth=0.8, alpha=0.5, color="black")
    for name, x_hat in filters.items():
        plt.plot(times, x_hat, label=name, linewidth=1.4)
    plt.xlabel("Time, s")
    plt.ylabel("Heart rate, bpm")
    plt.title("MIT-BIH 200: alfa-beta filter variants vs measured heart rate")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_divergence_zoom(
    times, y, filters: dict[str, np.ndarray], window: tuple[float, float], path: Path
) -> None:
    lo, hi = window
    sel = (times >= lo - 15) & (times <= hi + 15)
    plt.figure(figsize=(9, 4.5))
    plt.axvspan(lo, hi, color="lightgrey", alpha=0.6, label="VT episode")
    plt.plot(
        times[sel], y[sel], label="Measured HR", linewidth=1, alpha=0.6, color="black"
    )
    for name, x_hat in filters.items():
        plt.plot(times[sel], x_hat[sel], label=name, linewidth=1.6)
    plt.xlabel("Time, s")
    plt.ylabel("Heart rate, bpm")
    plt.title(f"Divergence during a VT episode (t={lo:.0f}-{hi:.0f}s)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_anomaly_cleaning(times, y, y_clean, mask, path: Path) -> None:
    plt.figure(figsize=(12, 4))
    plt.plot(times, y, label="Raw HR", linewidth=0.8, alpha=0.6)
    plt.plot(
        times, y_clean, label="After compensating isolated outliers", linewidth=0.9
    )
    plt.scatter(
        times[mask],
        y[mask],
        color="crimson",
        s=14,
        zorder=5,
        label=f"Isolated anomalies, n={mask.sum()}",
    )
    plt.xlabel("Time, s")
    plt.ylabel("Heart rate, bpm")
    plt.title("Isolated-beat anomaly detection (rhythm episodes excluded)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


if __name__ == "__main__":
    hr_df = load_heart_rate()
    annotations = load_annotations()
    windows = find_rhythm_windows(annotations)
    vt_windows = [(s, e) for label, s, e in windows if label == "VT"]

    times = hr_df["time_s"].to_numpy()
    y_raw = hr_df["hr_bpm"].to_numpy()

    anomaly_mask, baseline = detect_isolated_anomalies(y_raw, times, windows)
    print(f"Isolated anomalies (outside labeled rhythm episodes): {anomaly_mask.sum()}")
    y = compensate_anomalies(y_raw, anomaly_mask, baseline)

    filters = {
        "Growing-memory alfa-beta": run_ab_filter(y, alfa_beta_coeffs),
        "Limited-memory alfa-beta": run_ab_filter(y, capped(alfa_beta_coeffs)),
        "Growing-memory alfa-beta-gamma": run_abg_filter(y, alfa_beta_gamma_coeffs),
        "Adaptive (CUSUM reset) alfa-beta": adaptive_reset_ab_filter(y),
    }

    abnormal_mask = abnormal_rhythm_mask(times, windows)
    print("\nRMSE vs raw HR (inside VT episodes | outside):")
    for name, x_hat in filters.items():
        rmse_in, rmse_out = rmse_by_segment(y, x_hat, abnormal_mask)
        print(f"  {name}: {rmse_in:.2f} | {rmse_out:.2f} bpm")

    mean_bias = unbiasedness_check()
    print(
        f"\nUnbiasedness check (Monte-Carlo mean tracking error): {mean_bias:.3f} bpm"
    )

    plot_filters(times, y, filters, abnormal_mask, OUT_DIR / "ecg200_filters.png")
    if vt_windows:
        plot_divergence_zoom(
            times, y, filters, vt_windows[0], OUT_DIR / "ecg200_divergence_zoom.png"
        )
    plot_anomaly_cleaning(
        times, y_raw, y, anomaly_mask, OUT_DIR / "ecg200_anomaly_cleaning.png"
    )

    assert all(np.isfinite(x_hat).all() for x_hat in filters.values())
    assert abs(mean_bias) < 1.0, "filter shows a systematic bias, not just noise"
    growing_rmse_in, _ = rmse_by_segment(
        y, filters["Growing-memory alfa-beta"], abnormal_mask
    )
    for name in ("Limited-memory alfa-beta", "Adaptive (CUSUM reset) alfa-beta"):
        fixed_rmse_in, _ = rmse_by_segment(y, filters[name], abnormal_mask)
        assert fixed_rmse_in < growing_rmse_in, (
            f"{name} should diverge less than growing-memory in VT episodes"
        )
    print("\nAll checks passed.")
