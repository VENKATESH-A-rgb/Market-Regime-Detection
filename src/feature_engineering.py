"""
feature_engineering.py — Step 3: Feature Engineering & Stationarity
Engineers 33+ features including momentum, volatility, macro indicators,
market breadth, and applies López de Prado's FFD for stationarity.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_CLEANED = PROJECT_ROOT / "data" / "cleaned"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


# ══════════════════════════════════════════════════════════════
# FRACTIONAL DIFFERENTIATION (FFD)
# Marcos López de Prado — Advances in Financial Machine Learning, Ch. 5
# ══════════════════════════════════════════════════════════════

def _get_weights_ffd(d: float, threshold: float = 1e-5) -> np.ndarray:
    """
    Compute FFD weights for fractional differentiation order `d`.
    Weights are truncated when they fall below `threshold`.
    """
    weights = [1.0]
    k = 1
    while True:
        w = -weights[-1] * (d - k + 1) / k
        if abs(w) < threshold:
            break
        weights.append(w)
        k += 1
    return np.array(weights[::-1]).reshape(-1, 1)


def frac_diff_ffd(series: pd.Series, d: float, threshold: float = 1e-5) -> pd.Series:
    """
    Apply Fixed-Width Window Fractional Differentiation to a series.

    Parameters
    ----------
    series : pd.Series
        The price series to differentiate (e.g., log prices).
    d : float
        The fractional differentiation order (0 < d < 1).
    threshold : float
        Weight truncation threshold.

    Returns
    -------
    pd.Series
        The fractionally differentiated series.
    """
    weights = _get_weights_ffd(d, threshold)
    width = len(weights)

    # Apply the filter as a dot product
    result = {}
    series_values = series.values
    for i in range(width - 1, len(series_values)):
        window = series_values[i - width + 1 : i + 1]
        result[series.index[i]] = np.dot(weights.T, window.reshape(-1, 1))[0, 0]

    return pd.Series(result, name=f"{series.name}_ffd_{d:.2f}")


def find_min_ffd_d(
    series: pd.Series,
    d_range: np.ndarray = np.arange(0.0, 1.01, 0.05),
    significance: float = 0.05,
    threshold: float = 1e-5,
) -> float:
    """
    Find the minimum `d` that makes the series stationary (ADF test).

    Parameters
    ----------
    series : pd.Series
        Log price series.
    d_range : np.ndarray
        Range of d values to test.
    significance : float
        ADF test significance level.

    Returns
    -------
    float
        The minimum d for stationarity.
    """
    for d in d_range:
        if d == 0:
            continue
        try:
            ffd_series = frac_diff_ffd(series, d, threshold)
            ffd_clean = ffd_series.dropna()
            if len(ffd_clean) < 50:
                continue
            adf_stat, p_value, *_ = adfuller(ffd_clean, maxlag=1)
            if p_value < significance:
                logger.info(f"  {series.name}: min d={d:.2f} (ADF p={p_value:.4f})")
                return d
        except Exception as e:
            logger.debug(f"  ADF failed for d={d}: {e}")
            continue

    logger.warning(f"  {series.name}: no d found for stationarity, using d=1.0")
    return 1.0


# ══════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════

def engineer_features(master: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer 33+ features from the master dataset.

    Categories:
    1. Moving Average Distances (4 features)
    2. Momentum (4 features)
    3. Volatility (2 features)
    4. VIX Regime (2 features)
    5. Yield Curve (2 features)
    6. Real Rate & Fed Policy (2 features)
    7. Labor Market (1 feature)
    8. Market Breadth / McClellan (2 features)
    9. RSI (2 features)
    10. MACD (2 features)
    11. Bollinger Bands (2 features)
    12. Volume (2 features)
    13. Cross-Asset (2 features)
    14. Crypto Signal (1 feature)
    15. Correlation (2 features)
    16. Distribution (2 features)
    17. FFD-transformed prices (5 features)
    """
    logger.info("Engineering features...")
    df = master.copy()
    spy_close = df["SPY_Close"]

    # ── 1. Moving Average Distances ──────────────────────────
    for window in [20, 50, 100, 200]:
        sma = spy_close.rolling(window).mean()
        df[f"SPY_SMA{window}_dist"] = (spy_close - sma) / sma * 100

    # ── 2. Momentum (log returns) ────────────────────────────
    for period, label in [(21, "1m"), (63, "3m"), (126, "6m"), (252, "1y")]:
        df[f"SPY_mom_{label}"] = np.log(spy_close / spy_close.shift(period))

    # ── 3. Volatility ────────────────────────────────────────
    spy_returns = np.log(spy_close / spy_close.shift(1))
    df["SPY_returns"] = spy_returns
    df["realized_vol_20d"] = spy_returns.rolling(20).std() * np.sqrt(252)
    df["realized_vol_60d"] = spy_returns.rolling(60).std() * np.sqrt(252)

    # ── 4. VIX Regime ────────────────────────────────────────
    vix = df.get("VIX_Close", pd.Series(dtype=float))
    if not vix.empty and vix.notna().sum() > 252:
        df["VIX_zscore"] = (vix - vix.rolling(252).mean()) / vix.rolling(252).std()
        df["VIX_term_structure"] = vix / vix.rolling(20).mean()
    else:
        # Use realized vol as VIX proxy
        df["VIX_zscore"] = (df["realized_vol_20d"] - df["realized_vol_20d"].rolling(252).mean()) / \
                           df["realized_vol_20d"].rolling(252).std()
        df["VIX_term_structure"] = df["realized_vol_20d"] / df["realized_vol_20d"].rolling(20).mean()

    # ── 5. Yield Curve ───────────────────────────────────────
    if "DGS10" in df.columns and "DGS2" in df.columns:
        df["yield_spread"] = df["DGS10"] - df["DGS2"]
        df["yield_spread_zscore"] = (
            (df["yield_spread"] - df["yield_spread"].rolling(252).mean())
            / df["yield_spread"].rolling(252).std()
        )
    else:
        df["yield_spread"] = 0.0
        df["yield_spread_zscore"] = 0.0

    # ── 6. Real Rate & Fed Policy ────────────────────────────
    if "DGS10" in df.columns and "CPIAUCSL" in df.columns:
        cpi_yoy = df["CPIAUCSL"].pct_change(252) * 100  # Annualized
        df["real_rate"] = df["DGS10"] - cpi_yoy
    else:
        df["real_rate"] = 0.0

    if "FEDFUNDS" in df.columns:
        df["fed_funds_3m_chg"] = df["FEDFUNDS"].diff(63)
    else:
        df["fed_funds_3m_chg"] = 0.0

    # ── 7. Labor Market ──────────────────────────────────────
    if "UNRATE" in df.columns:
        df["unrate_3m_chg"] = df["UNRATE"].diff(63)
    else:
        df["unrate_3m_chg"] = 0.0

    # ── 8. McClellan Oscillator (Market Breadth Proxy) ───────
    # Proxy: use relative performance of SPY vs QQQ and DIA
    if "QQQ_Close" in df.columns and "DIA_Close" in df.columns:
        # Breadth proxy: normalized advancing ratio
        spy_ret = spy_close.pct_change()
        qqq_ret = df["QQQ_Close"].pct_change()
        dia_ret = df["DIA_Close"].pct_change()

        # "Advancing" = assets outperforming their 20-day mean
        breadth = (spy_ret + qqq_ret + dia_ret) / 3
        ema_19 = breadth.ewm(span=19).mean()
        ema_39 = breadth.ewm(span=39).mean()
        df["mcclellan_osc"] = ema_19 - ema_39
        df["mcclellan_sum"] = df["mcclellan_osc"].cumsum()
    else:
        df["mcclellan_osc"] = 0.0
        df["mcclellan_sum"] = 0.0

    # ── 9. RSI ───────────────────────────────────────────────
    delta = spy_close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI_14"] = 100 - (100 / (1 + rs))
    # RSI divergence: price momentum vs RSI momentum
    df["RSI_divergence"] = df["SPY_mom_1m"].rank(pct=True) - df["RSI_14"].rank(pct=True)

    # ── 10. MACD ─────────────────────────────────────────────
    ema12 = spy_close.ewm(span=12).mean()
    ema26 = spy_close.ewm(span=26).mean()
    df["MACD_line"] = ema12 - ema26
    df["MACD_histogram"] = df["MACD_line"] - df["MACD_line"].ewm(span=9).mean()

    # ── 11. Bollinger Bands ──────────────────────────────────
    sma20 = spy_close.rolling(20).mean()
    std20 = spy_close.rolling(20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    df["BB_pct_b"] = (spy_close - lower) / (upper - lower)
    df["BB_bandwidth"] = (upper - lower) / sma20

    # ── 12. Volume ───────────────────────────────────────────
    spy_vol = df.get("SPY_Volume", pd.Series(dtype=float))
    if not spy_vol.empty and spy_vol.notna().sum() > 20:
        df["volume_zscore"] = (spy_vol - spy_vol.rolling(20).mean()) / spy_vol.rolling(20).std()
        # On-Balance Volume slope
        obv = (np.sign(spy_returns) * spy_vol).cumsum()
        df["OBV_slope"] = obv.rolling(20).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 20 else np.nan,
            raw=False,
        )
    else:
        df["volume_zscore"] = 0.0
        df["OBV_slope"] = 0.0

    # ── 13. Cross-Asset ──────────────────────────────────────
    if "TLT_Close" in df.columns:
        df["SPY_TLT_ratio"] = spy_close / df["TLT_Close"]
    else:
        df["SPY_TLT_ratio"] = 1.0

    if "GLD_Close" in df.columns:
        df["SPY_GLD_ratio"] = spy_close / df["GLD_Close"]
    else:
        df["SPY_GLD_ratio"] = 1.0

    # ── 14. Crypto Signal ────────────────────────────────────
    if "BTC_USD_Close" in df.columns:
        btc = df["BTC_USD_Close"]
        df["BTC_mom_30d"] = np.log(btc / btc.shift(30))
    else:
        df["BTC_mom_30d"] = np.nan

    # ── 15. Rolling Correlations ─────────────────────────────
    if "TLT_Close" in df.columns:
        tlt_ret = np.log(df["TLT_Close"] / df["TLT_Close"].shift(1))
        df["corr_SPY_TLT_60d"] = spy_returns.rolling(60).corr(tlt_ret)
    else:
        df["corr_SPY_TLT_60d"] = 0.0

    if "GLD_Close" in df.columns:
        gld_ret = np.log(df["GLD_Close"] / df["GLD_Close"].shift(1))
        df["corr_SPY_GLD_60d"] = spy_returns.rolling(60).corr(gld_ret)
    else:
        df["corr_SPY_GLD_60d"] = 0.0

    # ── 16. Distribution ─────────────────────────────────────
    df["return_skew_20d"] = spy_returns.rolling(20).skew()
    df["return_kurt_20d"] = spy_returns.rolling(20).kurt()

    logger.info(f"Engineered {len([c for c in df.columns if c not in master.columns])} features")
    return df


# ══════════════════════════════════════════════════════════════
# FFD TRANSFORMATION
# ══════════════════════════════════════════════════════════════

def apply_ffd_to_prices(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Apply FFD to price series to achieve stationarity while preserving memory.

    Returns
    -------
    df : pd.DataFrame with FFD columns added
    ffd_params : dict mapping column name → optimal d
    """
    logger.info("Applying Fractional Differentiation (FFD)...")

    price_cols = [c for c in df.columns if c.endswith("_Close") and "VIX" not in c]
    ffd_params = {}

    for col in price_cols:
        series = np.log(df[col].dropna())  # Log transform first
        if len(series) < 200:
            logger.info(f"  Skipping {col} — too few observations ({len(series)})")
            continue

        # Find minimum d for stationarity
        optimal_d = find_min_ffd_d(series)
        ffd_params[col] = optimal_d

        # Apply FFD with optimal d
        ffd_series = frac_diff_ffd(series, optimal_d)
        df[f"{col.replace('_Close', '')}_ffd"] = ffd_series

    logger.info(f"FFD params: {ffd_params}")
    return df, ffd_params


# ══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════

def run_feature_pipeline(master: pd.DataFrame = None) -> tuple[pd.DataFrame, dict]:
    """
    Run the full feature engineering pipeline.

    Parameters
    ----------
    master : pd.DataFrame, optional
        If None, loads from data/cleaned/master.parquet.

    Returns
    -------
    features_df : pd.DataFrame
    ffd_params : dict
    """
    if master is None:
        master = pd.read_parquet(DATA_CLEANED / "master.parquet")
        logger.info(f"Loaded master: {master.shape}")

    # Engineer features
    df = engineer_features(master)

    # Apply FFD
    df, ffd_params = apply_ffd_to_prices(df)

    # Create target variable: next-day SPY return direction
    df["target_next_day"] = np.sign(df["SPY_returns"].shift(-1))
    df["target_next_day"] = df["target_next_day"].map({1.0: 1, -1.0: 0, 0.0: 0})

    # Drop rows with insufficient data (warmup period)
    initial_rows = len(df)
    df = df.dropna(subset=["realized_vol_20d", "RSI_14", "MACD_line"])
    
    # Impute any remaining NaNs in features (e.g. BTC, TLT, GLD before their start dates)
    exclude_pats = [
        "_Close", "_Open", "_High", "_Low", "_Volume",
        "regime_", "prob_", "target_", "Date",
        "CPIAUCSL", "UNRATE", "DGS10", "DGS2", "FEDFUNDS",
    ]
    feature_cols = []
    for col in df.columns:
        if any(pat in col for pat in exclude_pats):
            continue
        if df[col].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]:
            feature_cols.append(col)
            
    df[feature_cols] = df[feature_cols].ffill().bfill()
    
    # Drop rows with NaN in target (the very last row due to shift)
    df = df.dropna(subset=["target_next_day"])
    
    logger.info(f"Dropped {initial_rows - len(df)} warmup rows, {len(df)} remaining")

    # Save
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATA_PROCESSED / "features.parquet")
    logger.info(f"Saved features to {DATA_PROCESSED / 'features.parquet'}")

    # Save FFD params
    pd.Series(ffd_params).to_json(DATA_PROCESSED / "ffd_params.json")

    return df, ffd_params


if __name__ == "__main__":
    features, params = run_feature_pipeline()
    print(f"\nFeatures DataFrame: {features.shape}")
    print(f"FFD params: {params}")
    print(f"\nFeature columns ({len(features.columns)}):")
    for col in sorted(features.columns):
        print(f"  {col}")
