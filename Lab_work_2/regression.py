"""
LSM regression, regression, model-degree selection, forecasting - BlackRock (BLK).

Reuses the data from the Lab 1 and implement the matrix form of the Least Squares Method (LSM) regression:
C_hat = (F^T R^-1 F)^-1 F^T Y
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_FILE = Path(__file__).parent.parent / "Lab_work_1" / "data" / "blk_history.csv"
OUT_DIR = Path(__file__).parent
CANDIDATE_DEGREES = [1, 2, 3]
TEST_FRACTION = 0.2
FORECAST_FRACTION = 0.5


def load_series(path: Path = DATA_FILE) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])


def build_basis(t: np.ndarray, degree: int) -> np.ndarray:
    """Basis-function matrix F: column j is t**j, j = 0..degree."""
    return np.column_stack([t**p for p in range(degree + 1)])


def lsm_fit(
    t: np.ndarray, y: np.ndarray, degree: int, weights: np.ndarray | None = None
) -> np.ndarray:
    """Matrix-form least squares: C = (F^T R^-1 F)^-1 F^T Y.

    weights=None means R^-1 = identity (equal-precision measurements, ordinary LSM);
    a weight vector turns this into weighted LSM (reused for IRLS in robust_learning.py).
    Solved via np.linalg.solve on the normal equations, equivalent to forming the
    inverse explicitly but numerically more stable.
    """
    F = build_basis(t, degree)
    if weights is None:
        weights = np.ones_like(y)
    FtW = F.T * weights
    coeffs = np.linalg.solve(FtW @ F, FtW @ y)
    return coeffs


def predict(t: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    return build_basis(t, len(coeffs) - 1) @ coeffs


def r_squared(y: np.ndarray, y_hat: np.ndarray) -> float:
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot != 0 else 0.0


def select_degree(
    t: np.ndarray,
    y: np.ndarray,
    degrees=CANDIDATE_DEGREES,
    test_fraction: float = TEST_FRACTION,
) -> tuple[int, dict[int, float]]:
    """Quality metric + model optimization: chronological
    train/test split, pick the degree with the best test R²."""
    n_test = int(len(y) * test_fraction)
    t_train, y_train = t[:-n_test], y[:-n_test]
    t_test, y_test = t[-n_test:], y[-n_test:]

    scores = {}
    for degree in degrees:
        coeffs = lsm_fit(t_train, y_train, degree)
        scores[degree] = r_squared(y_test, predict(t_test, coeffs))
        print(f"degree={degree}: R² for test data = {scores[degree]:.4f}")

    best_degree = max(scores, key=scores.get)
    print(f"Best degree: {best_degree} (based on R² on the test data)")
    return best_degree, scores


def forecast(
    coeffs: np.ndarray, n_observed: int, fraction: float = FORECAST_FRACTION
) -> tuple[np.ndarray, np.ndarray]:
    """Extrapolation for 0.5 * sample volume beyond the last observed point."""
    n_forecast = int(n_observed * fraction)
    t_future = np.arange(n_observed, n_observed + n_forecast, dtype=float)
    return t_future, predict(t_future, coeffs)


def plot_degree_scores(scores: dict[int, float], path: Path) -> None:
    plt.figure(figsize=(6, 4))
    degrees = list(scores.keys())
    plt.bar([str(d) for d in degrees], [scores[d] for d in degrees], color="steelblue")
    plt.xlabel("Degree of polynomial")
    plt.ylabel("R² on test data")
    plt.title("Model optimization: selecting polynomial degree")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_forecast(
    dates: pd.Series,
    real: np.ndarray,
    fitted: np.ndarray,
    t_future: np.ndarray,
    forecast_values: np.ndarray,
    path: Path,
) -> None:
    future_dates = pd.bdate_range(
        start=dates.iloc[-1] + pd.Timedelta(days=1), periods=len(t_future)
    )
    plt.figure(figsize=(10, 5))
    plt.plot(dates, real, label="BLK real data", linewidth=1)
    plt.plot(dates, fitted, label="LSM model (trained)", linewidth=2)
    plt.plot(
        future_dates,
        forecast_values,
        label="Forecast (extrapolation)",
        linewidth=2,
        linestyle="--",
        color="crimson",
    )
    plt.axvline(dates.iloc[-1], color="gray", linestyle=":", linewidth=1)
    plt.xlabel("Date")
    plt.ylabel("Price, USD")
    plt.title("BlackRock (BLK): LSM model and forecast for 0.5 sample volume")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


if __name__ == "__main__":
    df = load_series()
    close = df["close"].to_numpy()
    t = np.arange(len(close), dtype=float)

    best_degree, scores = select_degree(t, close)

    coeffs = lsm_fit(t, close, best_degree)
    fitted = predict(t, coeffs)
    print(f"LSM coefficients (C0..C{best_degree}): {coeffs}")
    print(f"R² on full dataset (degree={best_degree}) = {r_squared(close, fitted):.4f}")

    t_future, forecast_values = forecast(coeffs, len(close))
    print(
        f"Forecast for {len(t_future)} trading days ahead: "
        f"{forecast_values[0]:.2f} → {forecast_values[-1]:.2f} USD"
    )
    if (forecast_values < 0).any():
        print(
            "Warning: negative forecast values detected, extrapolation may be unreliable."
        )

    plot_degree_scores(scores, OUT_DIR / "blk_degree_selection.png")
    plot_forecast(
        df["date"],
        close,
        fitted,
        t_future,
        forecast_values,
        OUT_DIR / "blk_forecast.png",
    )

    assert best_degree in CANDIDATE_DEGREES
    assert len(t_future) == int(len(close) * FORECAST_FRACTION)
    assert np.isfinite(forecast_values).all()
