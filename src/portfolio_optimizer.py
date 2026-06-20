"""
portfolio_optimizer.py — Step 6: Adaptive Portfolio Allocation
Regime-aware convex optimization using PyPortfolioOpt + cvxpy.
Bull → max_sharpe(), Crisis → min_volatility() with safe-haven constraints.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUT_METRICS = PROJECT_ROOT / "output" / "metrics"

# Tradeable assets (^VIX is feature-only)
TRADEABLE_ASSETS = ["LIQUIDBEES.NS", "GOLDBEES.NS", "NIFTYBEES.NS", "JUNIORBEES.NS", "BANKBEES.NS"]
SAFE_HAVEN_ASSETS = ["LIQUIDBEES.NS", "GOLDBEES.NS"]

# Regime-specific optimization configurations
REGIME_CONFIGS = {
    "Bull": {
        "objective": "max_sharpe",
        "weight_bounds": (0.0, 0.60),
        "description": "Maximize risk-adjusted return; allow concentration",
    },
    "Recovery": {
        "objective": "max_sharpe",
        "weight_bounds": (0.0, 0.50),
        "description": "Capture recovery upside with moderate risk",
    },
    "Bear": {
        "objective": "min_volatility",
        "weight_bounds": (0.0, 0.40),
        "description": "Defensive: shift to safe havens",
    },
    "Crisis": {
        "objective": "min_volatility",
        "weight_bounds": (0.0, 0.40),
        "safe_haven_min": 0.30,
        "description": "Maximum defense; hard safe-haven floor",
    },
}


class RegimePortfolioOptimizer:
    """
    Regime-aware portfolio optimizer using PyPortfolioOpt.

    Dynamically switches between max_sharpe and min_volatility
    objectives based on the detected market regime.
    """

    def __init__(
        self,
        assets: list[str] = None,
        risk_free_rate: float = 0.02,
        rebalance_days: int = 21,
    ):
        self.assets = assets or TRADEABLE_ASSETS
        self.risk_free_rate = risk_free_rate
        self.rebalance_days = rebalance_days
        self.weight_history = []

    def _get_asset_columns(self, df: pd.DataFrame, lookback: int = None) -> list[str]:
        """Find available asset close price columns with sufficient data in the active window."""
        available = []
        check_df = df.tail(lookback) if lookback is not None else df
        for asset in self.assets:
            col = f"{asset.replace('-', '_')}_Close"
            if col in df.columns:
                # Check if asset has enough data in the active window
                min_days = min(126, len(check_df))
                if check_df[col].notna().sum() >= min_days:
                    available.append(asset)
        return available

    def _get_returns(
        self, df: pd.DataFrame, assets: list[str], lookback: int = 252
    ) -> pd.DataFrame:
        """Compute log returns for available assets over lookback period."""
        price_cols = {a: f"{a.replace('-', '_')}_Close" for a in assets}
        prices = df[[price_cols[a] for a in assets]].tail(lookback).copy()
        prices.columns = assets
        # Clip returns to +/- 15% to protect against Yahoo Finance data glitches
        returns = np.log(prices / prices.shift(1)).clip(-0.15, 0.15)
        returns = returns.replace([np.inf, -np.inf], np.nan)
        returns = returns.ffill().fillna(0.0)
        return returns

    def optimize_weights(
        self,
        df: pd.DataFrame,
        regime: str,
        lookback: int = 252,
    ) -> dict[str, float]:
        """
        Compute optimal portfolio weights for the given regime.

        Parameters
        ----------
        df : pd.DataFrame
            Historical data up to the current date.
        regime : str
            Current regime label (Bull/Recovery/Bear/Crisis).
        lookback : int
            Number of days for return/covariance estimation.

        Returns
        -------
        dict mapping asset name → weight.
        """
        config = REGIME_CONFIGS.get(regime, REGIME_CONFIGS["Bear"])
        available_assets = self._get_asset_columns(df, lookback)

        if len(available_assets) < 2:
            # Too few assets — equal weight
            weights = {a: 1.0 / len(available_assets) for a in available_assets}
            logger.debug(f"Too few assets, equal weight: {weights}")
            return weights

        returns = self._get_returns(df, available_assets, lookback)

        if len(returns) < 60:
            weights = {a: 1.0 / len(available_assets) for a in available_assets}
            return weights

        try:
            # Try PyPortfolioOpt
            weights = self._optimize_pypfopt(
                returns, available_assets, config, regime
            )
        except Exception as e:
            logger.warning(f"PyPortfolioOpt failed: {e} — using analytical fallback")
            weights = self._optimize_analytical(
                returns, available_assets, config, regime
            )

        return weights

    def _optimize_pypfopt(
        self,
        returns: pd.DataFrame,
        assets: list[str],
        config: dict,
        regime: str,
    ) -> dict[str, float]:
        """Optimize using PyPortfolioOpt EfficientFrontier."""
        from pypfopt import EfficientFrontier, risk_models, expected_returns

        # Expected returns: exponentially weighted mean
        mu = expected_returns.ema_historical_return(
            returns, span=126, frequency=252, returns_data=True, log_returns=True
        )

        # Covariance: Ledoit-Wolf shrinkage
        cov = risk_models.CovarianceShrinkage(returns, returns_data=True, log_returns=True).ledoit_wolf()

        # Build EfficientFrontier
        weight_bounds = config.get("weight_bounds", (0.0, 0.50))
        ef = EfficientFrontier(mu, cov, weight_bounds=weight_bounds)

        # Add safe-haven constraint for Crisis regime
        safe_haven_min = config.get("safe_haven_min")
        if safe_haven_min:
            safe_indices = [
                assets.index(a) for a in SAFE_HAVEN_ASSETS if a in assets
            ]
            if safe_indices:
                # Add constraint: sum of safe haven weights >= safe_haven_min
                ef.add_constraint(
                    lambda w: sum(w[i] for i in safe_indices) >= safe_haven_min
                )

        # Optimize
        objective = config.get("objective", "min_volatility")
        if objective == "max_sharpe":
            try:
                ef.max_sharpe(risk_free_rate=self.risk_free_rate)
            except Exception:
                logger.debug(f"max_sharpe failed for {regime}, falling back to min_vol")
                ef = EfficientFrontier(mu, cov, weight_bounds=weight_bounds)
                ef.min_volatility()
        else:
            ef.min_volatility()

        cleaned = ef.clean_weights()
        return dict(cleaned)

    def _optimize_analytical(
        self,
        returns: pd.DataFrame,
        assets: list[str],
        config: dict,
        regime: str,
    ) -> dict[str, float]:
        """
        Analytical fallback optimizer (inverse-variance weighting).
        Used when PyPortfolioOpt/cvxpy fails.
        """
        variances = returns.var()
        inv_var = 1.0 / variances.replace(0, np.nan).dropna()
        weights = inv_var / inv_var.sum()

        # Apply bounds
        bounds = config.get("weight_bounds", (0.0, 0.50))
        for asset in weights.index:
            weights[asset] = np.clip(weights[asset], bounds[0], bounds[1])

        # Enforce safe-haven minimum in Crisis
        safe_haven_min = config.get("safe_haven_min")
        if safe_haven_min:
            safe_havens = [a for a in SAFE_HAVEN_ASSETS if a in weights.index]
            if safe_havens:
                current_safe = weights[safe_havens].sum()
                if current_safe < safe_haven_min:
                    deficit = safe_haven_min - current_safe
                    for sh in safe_havens:
                        weights[sh] += deficit / len(safe_havens)

        # Renormalize
        weights = weights / weights.sum()
        return weights.to_dict()

    def run_allocation_history(
        self,
        df: pd.DataFrame,
        regime_col: str = "regime_label_stable",
        lookback: int = 252,
    ) -> pd.DataFrame:
        """
        Compute portfolio weights over the entire history with regime-aware
        rebalancing.

        Rebalancing triggers on:
        1. Regime change
        2. Every `self.rebalance_days` days (monthly default)

        Returns
        -------
        pd.DataFrame with date index and weight per asset.
        """
        logger.info("Computing historical portfolio allocation...")

        available_assets = self._get_asset_columns(df)
        weight_records = []
        current_weights = {a: 1.0 / len(available_assets) for a in available_assets}
        current_regime = None
        days_since_rebalance = 0

        # Start after warmup period
        start_idx = max(lookback, 252)
        dates = df.index[start_idx:]

        for i, date in enumerate(dates):
            regime = df.loc[date].get(regime_col, "Bear")
            if pd.isna(regime):
                regime = "Bear"  # Default to defensive

            # Check if rebalance is needed
            regime_changed = (regime != current_regime)
            periodic_rebalance = (days_since_rebalance >= self.rebalance_days)

            if regime_changed or periodic_rebalance:
                hist_slice = df.loc[:date]
                try:
                    new_weights = self.optimize_weights(hist_slice, regime, lookback)
                    current_weights = new_weights
                    current_regime = regime
                    days_since_rebalance = 0
                except Exception as e:
                    logger.debug(f"Optimization failed at {date}: {e}")
                    days_since_rebalance += 1
            else:
                days_since_rebalance += 1

            record = {"Date": date, "regime": regime}
            record.update({f"w_{a}": current_weights.get(a, 0.0) for a in available_assets})
            weight_records.append(record)

        weights_df = pd.DataFrame(weight_records).set_index("Date")

        # Save
        OUTPUT_METRICS.mkdir(parents=True, exist_ok=True)
        weights_df.to_parquet(OUTPUT_METRICS / "weights_history.parquet")
        weights_df.to_csv(OUTPUT_METRICS / "weights_history.csv")
        logger.info(f"Weight history: {weights_df.shape}, saved to output/metrics/")

        return weights_df


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def run_portfolio_optimization(
    features_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """Run the full portfolio optimization pipeline."""
    if features_df is None:
        features_df = pd.read_parquet(DATA_PROCESSED / "features_with_regimes.parquet")
        logger.info(f"Loaded data: {features_df.shape}")

    optimizer = RegimePortfolioOptimizer()
    weights_df = optimizer.run_allocation_history(features_df)

    logger.info("\n=== Allocation Summary by Regime ===")
    weight_cols = [c for c in weights_df.columns if c.startswith("w_")]
    for regime in weights_df["regime"].unique():
        mask = weights_df["regime"] == regime
        avg_weights = weights_df.loc[mask, weight_cols].mean()
        logger.info(f"  {regime}: {avg_weights.round(3).to_dict()}")

    return weights_df


if __name__ == "__main__":
    weights = run_portfolio_optimization()
    print(f"\nWeights history: {weights.shape}")
    print(weights.tail(10))
