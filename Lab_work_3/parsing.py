# Fetching ECG record 200 (MIT-BIH Arrhythmia Database) from PhysioNet and saving
# the raw signal, beat annotations, and derived heart-rate series to CSV files.

from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

RECORD = "200"
PN_DIR = "mitdb"
OUT_DIR = Path(__file__).parent / "data"

# Annotation symbols that mark something other than an actual heartbeat
# (rhythm-change markers, artifact/noise segments, etc.) - excluded from HR computation.
NON_BEAT_SYMBOLS = {"+", "~", "|", "x", "[", "!", "]"}


def fetch_signal() -> tuple[wfdb.Record, wfdb.Annotation]:
    record = wfdb.rdrecord(RECORD, pn_dir=PN_DIR)
    annotation = wfdb.rdann(RECORD, "atr", pn_dir=PN_DIR)
    return record, annotation


def build_signal_df(record: wfdb.Record) -> pd.DataFrame:
    df = pd.DataFrame(record.p_signal, columns=record.sig_name)
    df.insert(0, "time_s", np.arange(len(df)) / record.fs)
    return df


def build_annotations_df(annotation: wfdb.Annotation) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample": annotation.sample,
            "time_s": annotation.sample / annotation.fs,
            "symbol": annotation.symbol,
            "aux_note": annotation.aux_note,
        }
    )


def build_heart_rate_df(annotation: wfdb.Annotation) -> pd.DataFrame:
    """Instantaneous HR from RR intervals between consecutive beat annotations.

    Indexed by beat number rather than resampled onto a uniform time grid - the alfa-beta filter
    formulas only need a measurement index, not a fixed sampling period.
    """
    is_beat = ~np.isin(annotation.symbol, list(NON_BEAT_SYMBOLS))
    samples = annotation.sample[is_beat]
    symbols = np.array(annotation.symbol)[is_beat]
    rr = np.diff(samples) / annotation.fs
    return pd.DataFrame(
        {
            "beat_index": np.arange(len(rr)),
            "sample": samples[1:],
            "time_s": samples[1:] / annotation.fs,
            "symbol": symbols[1:],
            "hr_bpm": 60 / rr,
        }
    )


if __name__ == "__main__":
    record, annotation = fetch_signal()
    signal_df = build_signal_df(record)
    annotations_df = build_annotations_df(annotation)
    heart_rate_df = build_heart_rate_df(annotation)

    assert signal_df.shape[0] == record.sig_len
    assert heart_rate_df["hr_bpm"].between(20, 300).all(), "implausible HR values"
    assert set(annotations_df["symbol"]) >= {"N", "V"}, "expected N/V beats missing"

    OUT_DIR.mkdir(exist_ok=True)
    signal_df.to_csv(OUT_DIR / "ecg_200_signal.csv", index=False)
    annotations_df.to_csv(OUT_DIR / "ecg_200_annotations.csv", index=False)
    heart_rate_df.to_csv(OUT_DIR / "ecg_200_heart_rate.csv", index=False)

    print(f"Data source: PhysioNet MIT-BIH Arrhythmia Database, record {RECORD}")
    print(f"Signal: {record.sig_len} samples @ {record.fs} Hz, leads={record.sig_name}")
    print(f"Annotations: {len(annotations_df)} events")
    print(
        f"Heart rate: {len(heart_rate_df)} beats, "
        f"{heart_rate_df['hr_bpm'].min():.0f}-{heart_rate_df['hr_bpm'].max():.0f} bpm"
    )
    print(f"Saved to {OUT_DIR}")
