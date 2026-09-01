# BlackRock (BLK) preprocessing and analysis of historical stock prices from Yahoo Finance.

"""
1) Trend identification and residual analysis.
2) Residual statistical characteristics (mean, variance, std, histogram).
3) Synthesizing an additive model B_measured = B_ideal + xi with matching characteristics.
4) Verification: comparing the synthetic series against the real data.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_FILE = Path(__file__).parent / "data" / "blk_history.csv"
OUT_DIR = Path(__file__).parent
TREND_DEGREE = 2  # quadratic trend
NOISE_SEED = 42


def load_series(path: Path = DATA_FILE) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])


def fit_trend(
    y: np.ndarray, degree: int = TREND_DEGREE
) -> tuple[np.ndarray, np.ndarray]:
    """LSM trend estimate (np.polyfit) and residual - time redundancy"""
    t = np.arange(len(y), dtype=float)
    coeffs = np.polyfit(
        t, y, degree
    )  # Least Squares Method to fit a polynomial of given degree to the data
    # T(x) = coeffs[0] * x^degree + coeffs[1] * x + coeffs[2] ** 2
    residual = y - np.polyval(coeffs, t)
    return coeffs, residual


def r_squared(y: np.ndarray, trend: np.ndarray) -> float:
    ss_res = np.sum((y - trend) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0


def stat_characteristics(x: np.ndarray, label: str) -> dict[str, float]:
    mean, std = float(np.mean(x)), float(np.std(x, ddof=1))
    print(f"{label}: mean={mean:.4f}, var={std**2:.4f}, std={std:.4f}")
    return {"mean": mean, "std": std}


def synthesize(
    n: int,
    coeffs: np.ndarray,
    noise_mean: float,
    noise_std: float,
    seed: int = NOISE_SEED,
) -> np.ndarray:
    """Additive model B_measured = B_ideal + xi, xi ~ N(noise_mean, noise_std)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    return np.polyval(coeffs, t) + rng.normal(noise_mean, noise_std, n)


def plot_trend(dates, real, trend, path: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(dates, real, label="BLK real close price", linewidth=1)
    plt.plot(dates, trend, label=f"Trend (degree={TREND_DEGREE})", linewidth=2)
    plt.xlabel("Date")
    plt.ylabel("Price, USD")
    plt.title("BlackRock (BLK): real close price and trend")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_histogram(residual: np.ndarray, path: Path, label: str) -> None:
    plt.figure(figsize=(7, 4))
    plt.hist(residual, bins=30, facecolor="steelblue", alpha=0.7)
    plt.title(f"Histogram of Residual Distribution: {label}")
    plt.xlabel("Δ, USD")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_verification(dates, real, synthetic, path: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(dates, real, label="Real Data BLK", linewidth=1)
    plt.plot(dates, synthetic, label="Synthesized Model", linewidth=1, alpha=0.8)
    plt.xlabel("Date")
    plt.ylabel("Price, USD")
    plt.title("Verification: Real Data vs Synthesized Additive Model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


if __name__ == "__main__":
    df = load_series()
    close = df["close"].to_numpy()
    t = np.arange(len(close), dtype=float)

    coeffs, residual = fit_trend(close)
    print(f"Coefficients of the trend (highest→constant): {coeffs}")
    print(f"R² of the trend = {r_squared(close, np.polyval(coeffs, t)):.4f}")

    real_stats = stat_characteristics(residual, "real data BLK (residual)")
    plot_trend(df["date"], close, np.polyval(coeffs, t), OUT_DIR / "blk_trend.png")
    plot_histogram(residual, OUT_DIR / "blk_residual_hist.png", "real data")

    synthetic = synthesize(len(close), coeffs, real_stats["mean"], real_stats["std"])
    _, synth_residual = fit_trend(synthetic)
    synth_stats = stat_characteristics(synth_residual, "synthesized model (residual)")
    plot_verification(df["date"], close, synthetic, OUT_DIR / "blk_verification.png")

    print("Verification Results:")
    print(f"Δ mean = {abs(real_stats['mean'] - synth_stats['mean']):.4f}")
    print(f"Δ std = {abs(real_stats['std'] - synth_stats['std']):.4f}")

    assert len(synthetic) == len(close)
    assert abs(real_stats["std"] - synth_stats["std"]) < 0.1 * real_stats["std"], (
        "synthesize() does not match the std of the real data residuals"
    )
