"""
data_loader.py — Step 2: Data Acquisition
Downloads daily OHLCV data (yfinance) and macro indicators (FRED API),
merges into a single master DataFrame, and saves to data/cleaned/master.parquet.
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import yfinance as yf

# Optional FRED support — gracefully degrade if no API key
try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_CLEANED = PROJECT_ROOT / "data" / "cleaned"

TICKERS = ["SPY", "^VIX", "TLT", "BTC-USD", "GLD", "QQQ", "DIA"]
TRADEABLE_ASSETS = ["SPY", "TLT", "GLD", "QQQ", "DIA"]  # ^VIX and BTC-USD are feature-only

START_DATE = "1993-01-01"
END_DATE = "2026-05-31"

FRED_SERIES = {
    "DGS10": "10Y Treasury Yield",
    "DGS2": "2Y Treasury Yield",
    "FEDFUNDS": "Fed Funds Rate",
    "CPIAUCSL": "CPI (All Urban Consumers)",
    "UNRATE": "Unemployment Rate",
}


# ──────────────────────────────────────────────────────────────
# OHLCV Download
# ──────────────────────────────────────────────────────────────
def download_ohlcv(
    tickers: list[str] = TICKERS,
    start: str = START_DATE,
    end: str = END_DATE,
    save_raw: bool = True,
) -> pd.DataFrame:
    """
    Download daily OHLCV data from Yahoo Finance for all tickers.
    Returns a multi-level DataFrame with (ticker, OHLCV) columns.
    """
    logger.info(f"Downloading OHLCV for {tickers} from {start} to {end}")

    # Download all tickers in one call
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        group_by="ticker",
        threads=True,
    )

    if raw.empty:
        raise RuntimeError("yfinance returned empty DataFrame — check tickers/dates")

    # Save individual ticker CSVs for reproducibility
    if save_raw:
        DATA_RAW.mkdir(parents=True, exist_ok=True)
        for ticker in tickers:
            try:
                if len(tickers) == 1:
                    ticker_df = raw.copy()
                else:
                    ticker_df = raw[ticker].dropna(how="all")
                ticker_clean = ticker.replace("^", "").replace("-", "_")
                ticker_df.to_csv(DATA_RAW / f"{ticker_clean}_ohlcv.csv")
                logger.info(f"  Saved {ticker}: {len(ticker_df)} rows")
            except KeyError:
                logger.warning(f"  Ticker {ticker} not found in downloaded data")

    logger.info(f"OHLCV download complete: {raw.shape}")
    return raw


# ──────────────────────────────────────────────────────────────
# FRED Macro Download
# ──────────────────────────────────────────────────────────────
def download_fred_macro(
    series_ids: dict[str, str] = FRED_SERIES,
    start: str = START_DATE,
    end: str = END_DATE,
) -> pd.DataFrame:
    """
    Fetch macro indicators from FRED.
    Falls back to synthetic proxies if API key is unavailable.
    """
    api_key = os.getenv("FRED_API_KEY", "")

    if not api_key or api_key == "your_api_key_here" or not FRED_AVAILABLE:
        logger.warning("FRED API key not configured — generating synthetic macro proxies")
        return _generate_synthetic_macro(start, end)

    try:
        fred = Fred(api_key=api_key)
        macro_frames = {}

        for series_id, description in series_ids.items():
            logger.info(f"  Fetching FRED: {series_id} ({description})")
            try:
                s = fred.get_series(series_id, observation_start=start, observation_end=end)
                macro_frames[series_id] = s
            except Exception as e:
                logger.warning(f"  Failed to fetch {series_id}: {e}")

        if not macro_frames:
            logger.warning("No FRED data retrieved — falling back to synthetic")
            return _generate_synthetic_macro(start, end)

        macro_df = pd.DataFrame(macro_frames)
        macro_df.index = pd.to_datetime(macro_df.index)
        macro_df.index.name = "Date"

        # Save raw
        DATA_RAW.mkdir(parents=True, exist_ok=True)
        macro_df.to_csv(DATA_RAW / "fred_macro.csv")
        logger.info(f"FRED macro download complete: {macro_df.shape}")

        return macro_df

    except Exception as e:
        logger.warning(f"FRED API error: {e} — falling back to synthetic")
        return _generate_synthetic_macro(start, end)


def _generate_synthetic_macro(start: str, end: str) -> pd.DataFrame:
    """
    Generate realistic synthetic macro data when FRED API is unavailable.
    Uses historical-ish patterns for demonstration purposes.
    """
    logger.info("Generating synthetic macro indicators...")
    dates = pd.bdate_range(start=start, end=end)

    np.random.seed(42)
    n = len(dates)

    # Simulate a mean-reverting 10Y yield (roughly 1.5% - 6%)
    dgs10 = np.cumsum(np.random.normal(0, 0.01, n)) + 4.0
    dgs10 = np.clip(dgs10, 0.5, 8.0)

    # 2Y yield slightly below 10Y with occasional inversions
    dgs2 = dgs10 - np.abs(np.random.normal(0.5, 0.3, n))
    dgs2 = np.clip(dgs2, 0.1, 7.5)

    # Fed Funds Rate — step function-ish
    fedfunds = np.clip(dgs2 - np.random.normal(0.5, 0.2, n), 0, 6.0)

    # CPI — trending upward from ~140 to ~310
    cpi = np.linspace(140, 310, n) + np.cumsum(np.random.normal(0, 0.1, n))

    # Unemployment — cyclic between 3.5% and 10%
    unrate = 5.5 + 2.0 * np.sin(np.linspace(0, 8 * np.pi, n)) + np.random.normal(0, 0.2, n)
    unrate = np.clip(unrate, 3.0, 12.0)

    macro_df = pd.DataFrame(
        {
            "DGS10": dgs10,
            "DGS2": dgs2,
            "FEDFUNDS": fedfunds,
            "CPIAUCSL": cpi,
            "UNRATE": unrate,
        },
        index=dates,
    )
    macro_df.index.name = "Date"

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    macro_df.to_csv(DATA_RAW / "fred_macro_synthetic.csv")
    return macro_df


# ──────────────────────────────────────────────────────────────
# Merge & Clean
# ──────────────────────────────────────────────────────────────
def merge_datasets(ohlcv_df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge OHLCV and macro data into a single master DataFrame.
    - Extracts Close prices for each ticker into flat columns
    - Forward-fills macro data (different publication frequencies)
    - Saves to data/cleaned/master.parquet
    """
    logger.info("Merging OHLCV and macro datasets...")

    # Detect MultiIndex column format
    # yfinance >= 0.2.40 uses (Price, Ticker), older uses (Ticker, Price)
    close_dict = {}
    volume_dict = {}

    if isinstance(ohlcv_df.columns, pd.MultiIndex):
        level0_values = ohlcv_df.columns.get_level_values(0).unique().tolist()
        # Check if level 0 is tickers or price fields
        price_fields = {"Open", "High", "Low", "Close", "Volume", "Adj Close"}
        if any(v in price_fields for v in level0_values):
            # Format: (Price, Ticker) — newer yfinance
            col_format = "price_first"
            logger.info("Detected yfinance MultiIndex format: (Price, Ticker)")
        else:
            # Format: (Ticker, Price) — older yfinance
            col_format = "ticker_first"
            logger.info("Detected yfinance MultiIndex format: (Ticker, Price)")
    else:
        col_format = "flat"
        logger.info("Detected flat column format")

    for ticker in TICKERS:
        ticker_clean = ticker.replace("^", "").replace("-", "_")
        try:
            if col_format == "price_first":
                close_dict[f"{ticker_clean}_Close"] = ohlcv_df[("Close", ticker)]
                if ("Volume", ticker) in ohlcv_df.columns:
                    volume_dict[f"{ticker_clean}_Volume"] = ohlcv_df[("Volume", ticker)]
                if ticker == "SPY":
                    for col in ["Open", "High", "Low"]:
                        if (col, ticker) in ohlcv_df.columns:
                            close_dict[f"SPY_{col}"] = ohlcv_df[(col, ticker)]
            elif col_format == "ticker_first":
                close_dict[f"{ticker_clean}_Close"] = ohlcv_df[(ticker, "Close")]
                if (ticker, "Volume") in ohlcv_df.columns:
                    volume_dict[f"{ticker_clean}_Volume"] = ohlcv_df[(ticker, "Volume")]
                if ticker == "SPY":
                    for col in ["Open", "High", "Low"]:
                        if (ticker, col) in ohlcv_df.columns:
                            close_dict[f"SPY_{col}"] = ohlcv_df[(ticker, col)]
            else:
                close_dict[f"{ticker_clean}_Close"] = ohlcv_df["Close"]
        except KeyError:
            logger.warning(f"  {ticker} not found in OHLCV data — skipping")

    prices_df = pd.DataFrame(close_dict)
    volumes_df = pd.DataFrame(volume_dict)

    # Flatten index if multi-level
    for frame in [prices_df, volumes_df]:
        if hasattr(frame.index, 'nlevels') and frame.index.nlevels > 1:
            frame.index = frame.index.droplevel(list(range(1, frame.index.nlevels)))

    prices_df.index = pd.to_datetime(prices_df.index)
    prices_df.index.name = "Date"

    volumes_df.index = pd.to_datetime(volumes_df.index)
    volumes_df.index.name = "Date"

    # Merge prices + volumes
    master = prices_df.join(volumes_df, how="left")

    # Merge macro data — reindex to trading days and forward-fill
    macro_df.index = pd.to_datetime(macro_df.index)
    macro_reindexed = macro_df.reindex(master.index, method="ffill")
    master = master.join(macro_reindexed, how="left")

    # Forward-fill remaining gaps (weekends, holidays in macro)
    master = master.ffill()

    # Log coverage statistics
    logger.info(f"Master DataFrame shape: {master.shape}")
    logger.info(f"Date range: {master.index.min()} to {master.index.max()}")
    for col in master.columns:
        null_pct = master[col].isna().mean() * 100
        first_valid = master[col].first_valid_index()
        logger.info(f"  {col}: {null_pct:.1f}% null, starts {first_valid}")

    # Save
    DATA_CLEANED.mkdir(parents=True, exist_ok=True)
    master.to_parquet(DATA_CLEANED / "master.parquet")
    master.to_csv(DATA_CLEANED / "master.csv")
    logger.info(f"Saved master dataset to {DATA_CLEANED / 'master.parquet'}")

    return master


# ──────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────
def validate_data(df: pd.DataFrame) -> dict:
    """Run data quality checks and return a summary report."""
    report = {
        "shape": df.shape,
        "date_range": (str(df.index.min()), str(df.index.max())),
        "columns": list(df.columns),
        "null_counts": df.isna().sum().to_dict(),
        "null_pct": (df.isna().mean() * 100).round(2).to_dict(),
    }

    # Check for suspicious gaps
    date_diffs = pd.Series(df.index).diff().dt.days
    max_gap = date_diffs.max()
    if max_gap > 5:
        logger.warning(f"Maximum date gap: {max_gap} days — check for missing data")
    report["max_date_gap_days"] = int(max_gap) if pd.notna(max_gap) else 0

    return report


# ──────────────────────────────────────────────────────────────
# Main entrypoint
# ──────────────────────────────────────────────────────────────
def run_data_pipeline() -> pd.DataFrame:
    """Execute the full data acquisition pipeline."""
    ohlcv = download_ohlcv()
    macro = download_fred_macro()
    master = merge_datasets(ohlcv, macro)
    report = validate_data(master)
    logger.info(f"Data pipeline complete. Report: {report['shape']}, "
                f"range: {report['date_range']}")
    return master


if __name__ == "__main__":
    master = run_data_pipeline()
    print(f"\nMaster DataFrame: {master.shape}")
    print(master.head())
    print(master.tail())
