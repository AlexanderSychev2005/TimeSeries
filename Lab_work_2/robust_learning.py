"""
Anomaly detection/cleaning and a robust IRLS regression as the own R&D algorithm - BlackRock (BLK)

IRLS = Iteratively Reweighted Least Squares: refits mnk_fit() from regression.py
several times, each time decreasing the weight of points with a large residual,
so outliers stop pulling the trend line without being manually cut out.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from regression import load_series, lsm_fit, predict, r_squared, select_degree

OUT_DIR = Path(__file__).parent
SIGMA_THRESHOLD = 3
IRLS_ITERATIONS = 8
TUKEY_C = 4.685  # standard tuning constant (~95% efficiency under Gaussian errors)


def detect_anomalies(residual: np.ndarray, k: float = SIGMA_THRESHOLD) -> np.ndarray:
    """3-sigma rule: an anomaly is a residual further than k*std from the mean."""
    mean, std = residual.mean(), residual.std(ddof=1)
    return np.abs(residual - mean) > k * std


def clean_anomalies(
    t: np.ndarray, y: np.ndarray, degree: int, k: float = SIGMA_THRESHOLD
) -> tuple[np.ndarray, np.ndarray]:
    """Detect anomalies on a plain LSM fit, replace them with the model estimate."""
    coeffs = lsm_fit(t, y, degree)
    residual = y - predict(t, coeffs)
    mask = detect_anomalies(residual, k)
    y_clean = y.copy()
    y_clean[mask] = predict(t[mask], coeffs)
    return y_clean, mask


def tukey_weights(residual: np.ndarray, c: float = TUKEY_C) -> np.ndarray:
    """Tukey biweight: weight -> 0 as |residual| grows past c * robust_scale."""
    mad = np.median(np.abs(residual - np.median(residual)))
    scale = max(mad / 0.6745, 1e-9)  # 0.6745 converts MAD to a sigma-equivalent scale
    u = residual / (c * scale)
    return np.where(np.abs(u) < 1, (1 - u**2) ** 2, 0.0)


def irls_fit(
    t: np.ndarray,
    y: np.ndarray,
    degree: int,
    n_iter: int = IRLS_ITERATIONS,
    c: float = TUKEY_C,
) -> tuple[np.ndarray, np.ndarray]:
    """Own anomaly-aware learning algorithm."""
    weights = np.ones_like(y)
    coeffs = lsm_fit(t, y, degree, weights)
    for _ in range(n_iter):
        residual = y - predict(t, coeffs)
        weights = tukey_weights(residual, c)
        coeffs = lsm_fit(t, y, degree, weights)
    return coeffs, weights


def plot_robustness_comparison(
    dates: pd.Series,
    real: np.ndarray,
    t: np.ndarray,
    plain_coeffs,
    clean_coeffs,
    irls_coeffs,
    anomaly_mask: np.ndarray,
    path: Path,
) -> None:
    plt.figure(figsize=(10, 5))
    plt.scatter(
        dates[anomaly_mask],
        real[anomaly_mask],
        color="crimson",
        s=18,
        zorder=5,
        label=f"Outliers (3σ), n={anomaly_mask.sum()}",
    )
    plt.plot(dates, real, label="Real Data BLK", linewidth=1, alpha=0.6)
    plt.plot(dates, predict(t, plain_coeffs), label="Standard LSM", linewidth=2)
    plt.plot(
        dates,
        predict(t, clean_coeffs),
        label="LSM after Cleaning (3σ)",
        linewidth=2,
        linestyle="--",
    )
    plt.plot(
        dates,
        predict(t, irls_coeffs),
        label="IRLS (Robust LSM)",
        linewidth=2,
        linestyle=":",
    )
    plt.xlabel("Date")
    plt.ylabel("Price, USD")
    plt.title("Comparison: Standard LSM vs Cleaning (3σ) vs IRLS")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_irls_weights(dates: pd.Series, weights: np.ndarray, path: Path) -> None:
    plt.figure(figsize=(10, 3.2))
    plt.scatter(dates, weights, s=6, color="darkorange")
    plt.xlabel("Date")
    plt.ylabel("Weight qᵢ")
    plt.title(
        "IRLS: weight of each dimension after convergence (lower weight = suspicion of outlier)"
    )
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


if __name__ == "__main__":
    df = load_series()
    close = df["close"].to_numpy()
    t = np.arange(len(close), dtype=float)
    degree, _ = select_degree(t, close)

    plain_coeffs = lsm_fit(t, close, degree)
    residual = close - predict(t, plain_coeffs)
    anomaly_mask = detect_anomalies(residual)
    print(f"Found outliers (|Δ|>3σ): {anomaly_mask.sum()} out of {len(close)}")

    close_clean, _ = clean_anomalies(t, close, degree)
    clean_coeffs = lsm_fit(t, close_clean, degree)

    irls_coeffs, irls_weights = irls_fit(t, close, degree)
    print(f"R² standard LSM = {r_squared(close, predict(t, plain_coeffs)):.4f}")
    print(f"R² IRLS model = {r_squared(close, predict(t, irls_coeffs)):.4f}")
    print(
        f"Average weight of IRLS = {irls_weights.mean():.4f}, minimum = {irls_weights.min():.4f}"
    )

    plot_robustness_comparison(
        df["date"],
        close,
        t,
        plain_coeffs,
        clean_coeffs,
        irls_coeffs,
        anomaly_mask,
        OUT_DIR / "blk_robustness.png",
    )
    plot_irls_weights(df["date"], irls_weights, OUT_DIR / "blk_irls_weights.png")

    assert anomaly_mask.sum() < len(close) * 0.05, (
        "too many outliers - check the 3σ threshold"
    )
    assert irls_weights.min() >= 0
    assert np.isfinite(irls_coeffs).all()
