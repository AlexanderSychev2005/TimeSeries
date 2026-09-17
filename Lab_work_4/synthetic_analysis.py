"""Synthetic ECG Generation and Validation.

Generates a synthetic ECG time series reproducing empirical beat morphology,
bootstrap-resampled RR intervals (HRV), and residual noise from record 200,
then validates its statistical properties against the ground truth.
"""

from pathlib import Path

import common
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).parent
FS = 360
HALF_WINDOW_S = 0.15
SEED = 0


def load_real() -> tuple[np.ndarray, np.ndarray]:
    signal_df, annotations_df, _ = common.load_record("200")
    r_peak_samples, _ = common.beat_samples_and_symbols(annotations_df)
    return signal_df["MLII"].to_numpy(), r_peak_samples


def build_template(
    signal: np.ndarray, r_peak_samples: np.ndarray, half_window: int
) -> np.ndarray:
    """Compute empirical mean beat waveform from detected R-peaks."""
    windows, _ = common.extract_beats(signal, r_peak_samples, half_window)
    return windows.mean(axis=0)


def generate_synthetic(
    template: np.ndarray,
    real_rr_samples: np.ndarray,
    residual_std: float,
    n_beats: int,
    seed: int = SEED,
) -> np.ndarray:
    """Generate synthetic ECG using bootstrapped RR intervals and additive Gaussian noise."""
    rng = np.random.default_rng(seed)
    rr_draws = rng.choice(real_rr_samples, size=n_beats, replace=True)
    beat_starts = np.cumsum(rr_draws)
    total_len = beat_starts[-1] + len(template)
    signal = np.zeros(total_len)
    half = len(template) // 2
    for start in beat_starts:
        center = int(start)
        lo, hi = center - half, center - half + len(template)
        if lo >= 0 and hi <= total_len:
            signal[lo:hi] += template
    signal += rng.normal(0, residual_std, total_len)
    return signal


if __name__ == "__main__":
    half_window = round(HALF_WINDOW_S * FS)
    real_signal, r_peak_samples = load_real()
    real_slice = real_signal[:8000]

    template = build_template(real_signal, r_peak_samples, half_window)
    real_rr_samples = np.diff(r_peak_samples)
    # Decompose using mean RR instead of dominant_period to prevent bigeminy
    # pairs from distorting the residual noise estimate.
    true_beat_period = round(real_rr_samples.mean())
    _, _, real_resid, _ = common.decompose(real_slice, true_beat_period)
    residual_std = float(real_resid.std())

    synthetic = generate_synthetic(
        template,
        real_rr_samples,
        residual_std,
        n_beats=len(real_slice) // int(real_rr_samples.mean()) + 5,
    )
    synthetic = synthetic[: len(real_slice)]

    # Properties and Comparison
    print("Real vs synthetic: side-by-side comparison")
    real_props = common.properties(real_slice, "real MLII")
    synth_props = common.properties(synthetic, "synthetic MLII")

    real_period = common.dominant_period(real_slice, FS)
    synth_period = common.dominant_period(synthetic, FS)
    print(
        f"\nDominant period: real={real_period} samples, synthetic={synth_period} samples"
    )

    # Plotting
    plt.figure(figsize=(11, 4.5))
    plt.plot(real_slice, label="Real MLII", linewidth=0.9, alpha=0.8)
    plt.plot(synthetic, label="Synthetic MLII", linewidth=0.9, alpha=0.8)
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.title(
        "Record 200: real vs synthetic ECG (same length, same RR/noise statistics)"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "ecg200_synthetic_vs_real.png", dpi=150)
    plt.close()

    # Metrics delta
    mean_diff = abs(real_props["mean"] - synth_props["mean"])
    std_diff_rel = abs(real_props["std"] - synth_props["std"]) / real_props["std"]
    hurst_diff = abs(real_props["hurst"] - synth_props["hurst"])
    period_diff_rel = abs(real_period - synth_period) / real_period

    print(f"\nDelta mean = {mean_diff:.4f}")
    print(f"Delta std (relative) = {std_diff_rel:.2%}")
    print(f"Delta Hurst = {hurst_diff:.4f}")
    print(f"Delta dominant period (relative) = {period_diff_rel:.2%}")
    print(
        "\nNote: Dominant period mismatch is expected. Real record 200 exhibits "
        "bigeminy (alternating beat pairs), while synthetic model uses a single template."
    )

    assert len(synthetic) == len(real_slice)
    assert np.isfinite(synthetic).all()
    assert std_diff_rel < 0.3, "synthetic noise level does not match the real residual"
    assert hurst_diff < 0.3, "synthetic fractal structure does not match the real one"
    print("\nAll checks passed.")
