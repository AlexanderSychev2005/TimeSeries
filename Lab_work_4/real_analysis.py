"""ECG Analysis Pipeline: decomposition, lead correlation, and clustering.

Processes PhysioNet MIT-BIH records (e.g., 200 and 100) and compares
cross-patient heart rate time-series similarity via Euclidean and DTW distances.
"""

from pathlib import Path

import common
import numpy as np

OUT_DIR = Path(__file__).parent

DECOMP_SAMPLES = 8000  # ~22s @ 360Hz
PROPS_SAMPLES = 50000  # ~139s @ 360Hz (for stable ADF/KPSS/Hurst)
BEAT_HALF_WINDOW_S = 0.15  # ±150ms around R-peak


def analyze(record: str, label: str) -> dict:
    print(f"\nRecord {record} ({label}) analysis:")
    signal_df, annotations_df, fs = common.load_record(record)
    lead_names = [c for c in signal_df.columns if c != "time_s"]
    lead1, lead2 = (
        signal_df[lead_names[0]].to_numpy(),
        signal_df[lead_names[1]].to_numpy(),
    )
    r_peak_samples, beat_symbols = common.beat_samples_and_symbols(annotations_df)
    hr = common.heart_rate_series(r_peak_samples, fs)

    print(f"Leads: {lead_names}, fs={fs} Hz, beats={len(r_peak_samples)}")

    # 1. STL Decomposition
    decomp_slice = lead1[:DECOMP_SAMPLES]
    period = common.dominant_period(decomp_slice, fs)
    trend, seasonal, resid, r2_trend = common.decompose(decomp_slice, period)
    print(
        f"Decomposition: dominant period={period} samples ({fs / period:.2f} Hz), trend R^2={r2_trend:.4f}"
    )
    common.plot_decomposition(
        decomp_slice,
        trend,
        seasonal,
        resid,
        OUT_DIR / f"ecg{record}_decomposition.png",
        f"Record {record} ({lead_names[0]}): STL decomposition",
    )

    # 2. Time-Series Properties
    props_lead1 = common.properties(
        lead1[:PROPS_SAMPLES], f"{record}/{lead_names[0]} (raw signal)"
    )
    props_hr = common.properties(hr, f"{record}/HR (derived series)")

    # 3. Inter-lead Correlation
    r = common.correlate(lead1[:PROPS_SAMPLES], lead2[:PROPS_SAMPLES])
    rolling_r = common.rolling_correlation(
        lead1[:PROPS_SAMPLES], lead2[:PROPS_SAMPLES], window=fs
    )
    print(f"Correlation {lead_names[0]} vs {lead_names[1]}: r={r:.4f}")
    common.plot_correlation_heatmap(
        np.array([[1, r], [r, 1]]),
        lead_names,
        OUT_DIR / f"ecg{record}_correlation_heatmap.png",
        f"Record {record}: lead correlation",
    )
    common.plot_rolling_correlation(
        rolling_r,
        OUT_DIR / f"ecg{record}_rolling_correlation.png",
        f"Record {record}: rolling correlation between {lead_names[0]} and {lead_names[1]} (1s window)",
    )

    # 4. Beat Clustering: Morphology (K-Means)
    half_window = round(BEAT_HALF_WINDOW_S * fs)
    windows1, keep_mask = common.extract_beats(lead1, r_peak_samples, half_window)
    labels_classic = common.cluster_classic(windows1, n_clusters=3)
    ari = common.validate_clusters(labels_classic, beat_symbols[keep_mask])
    print(
        f"Classic clustering (k-means, {lead_names[0]} shape): ARI vs true beat type = {ari:.4f}"
    )
    common.plot_cluster_examples(
        windows1,
        labels_classic,
        OUT_DIR / f"ecg{record}_classic_clusters.png",
        f"Record {record}: classic clustering ({lead_names[0]} beat shape)",
    )

    # 5. Beat Clustering: Multi-factor (HR + Morphology)
    windows2, _ = common.extract_beats(lead2, r_peak_samples, half_window)
    kept_indices = np.where(keep_mask)[0]
    local_hr = np.array(
        [hr[i - 1] if 0 < i <= len(hr) else hr[0] for i in kept_indices]
    )
    feats1 = np.array([common.beat_features(w) for w in windows1])
    feats2 = np.array([common.beat_features(w) for w in windows2])
    features = np.column_stack([local_hr, feats1, feats2])
    labels_multi, z = common.cluster_multifactor(features, n_clusters=3)
    ari_multi = common.validate_clusters(labels_multi, beat_symbols[keep_mask])
    print(
        f"Multi-factor clustering (HR + {lead_names[0]} + {lead_names[1]}): ARI vs true beat type = {ari_multi:.4f}"
    )
    common.plot_dendrogram(
        z,
        OUT_DIR / f"ecg{record}_multifactor_dendrogram.png",
        f"Record {record}: multi-factor hierarchical clustering",
    )

    return {"hr": hr, "props_lead1": props_lead1, "props_hr": props_hr, "lead_corr": r}


if __name__ == "__main__":
    results_200 = analyze("200", "Arrhythmia")
    results_100 = analyze("100", "Normal sinus rhythm")

    print("\nCross-Record Similarity: ")
    euclid = common.euclidean_distance(results_200["hr"], results_100["hr"])
    dtw = common.dtw_distance(results_200["hr"][:1500], results_100["hr"][:1500])
    print(f"Euclidean distance (HR): {euclid:.2f}")
    print(f"DTW distance (HR, first 1500 beats): {dtw:.2f}")

    assert 0 < len(results_200["hr"])
    assert -1 <= results_200["lead_corr"] <= 1
    assert np.isfinite(euclid) and np.isfinite(dtw)
    print("\nAll checks passed.")
