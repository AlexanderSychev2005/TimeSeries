# Parsing historical stock prices for a given ticker from Yahoo Finance and saving it to a CSV file.


from pathlib import Path

import pandas as pd
import requests

TICKER = "BLK"
RANGE = "5y"
INTERVAL = "1d"
OUT_FILE = Path(__file__).parent / "data" / "blk_history.csv"


def fetch_history(
    ticker: str = TICKER, range_: str = RANGE, interval: str = INTERVAL
) -> pd.DataFrame:
    """HTTP GET to Yahoo Finance chart API, JSON parsing into DataFrame [date, close]."""
    resp = requests.get(
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
        params={"range": range_, "interval": interval},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=15,
    )
    resp.raise_for_status()
    result = resp.json()["chart"]["result"][0]
    close = result["indicators"]["quote"][0]["close"]
    df = (
        pd.DataFrame(
            {
                "date": pd.to_datetime(result["timestamp"], unit="s").date,
                "close": close,
            }
        )
        .dropna()
        .reset_index(drop=True)
    )
    return df


if __name__ == "__main__":
    df = fetch_history()
    assert len(df) > 200, f"Too few: {len(df)} rows"
    assert df["close"].gt(0).all(), "There are non-positive prices in the sample"

    OUT_FILE.parent.mkdir(exist_ok=True)
    df.to_csv(OUT_FILE, index=False)

    print(
        f"Data source: https://query1.finance.yahoo.com/v8/finance/chart/{TICKER} "
        f"(range={RANGE}, interval={INTERVAL})"
    )
    print(f"Saved {len(df)} trading days to {OUT_FILE}")
