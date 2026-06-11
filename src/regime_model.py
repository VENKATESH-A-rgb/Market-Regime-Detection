"""
regime_model.py — Step 4: Regime Detection Engine
Gaussian Hidden Markov Model (HMM) for market regime classification.
Outputs daily regime labels and probability vectors.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUT_MODELS = PROJECT_ROOT / "output" / "models"

# Regime labels (sorted by mean return after fitting)
REGIME_NAMES = {0: "Bull", 1: "Recovery", 2: "Bear", 3: "Crisis"}
REGIME_COLORS = {"Bull": "#2ecc71", "Recovery": "#3498db", "Bear": "#e67e22", "Crisis": "#e74c3c"}

# Features used for HMM (volatility + macro + momentum subset)
HMM_FEATURES = [
    "realized_vol_20d",
    "realized_vol_60d",
    "VIX_zscore",
    "VIX_term_structure",
    "yield_spread",
    "yield_spread_zscore",
    "SPY_mom_1m",
    "SPY_mom_3m",
    "SPY_TLT_ratio",
    "mcclellan_osc",
]


class RegimeDetector:
    """
    Gaussian HMM-based market regime detector.

    Classifies market conditions into 4 regimes:
    - Bull: high return, moderate vol
    - Recovery: positive return, elevated vol
    - Bear: negative return, moderate vol
    - Crisis: negative return, extreme vol
    """

    def __init__(self, n_regimes: int = 4, n_iter: int = 200, random_state: int = 42):
        self.n_regimes = n_regimes
        self.n_iter = n_iter
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.regime_mapping = {}  # HMM state → semantic label
        self.feature_names = HMM_FEATURES

    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select and validate HMM input features."""
        available = [f for f in self.feature_names if f in df.columns]
        if len(available) < 5:
            raise ValueError(
                f"Need at least 5 HMM features, found {len(available)}: {available}"
            )
        if len(available) < len(self.feature_names):
            missing = set(self.feature_names) - set(available)
            logger.warning(f"Missing HMM features (will skip): {missing}")
        self.feature_names = available
        return df[available]

    def fit(self, df: pd.DataFrame, returns_col: str = "SPY_returns") -> "RegimeDetector":
        """
        Fit the HMM on historical feature data.

        Parameters
        ----------
        df : pd.DataFrame
            Feature-engineered DataFrame (output of feature_engineering.py).
        returns_col : str
            Column used to compute mean returns per regime for label mapping.
        """
        logger.info(f"Fitting HMM with {self.n_regimes} regimes...")

        # Select and scale features
        X_raw = self._select_features(df).copy()
        X_clean = X_raw.dropna()

        if len(X_clean) < 100:
            raise ValueError(f"Insufficient data for HMM: {len(X_clean)} rows")

        X_scaled = self.scaler.fit_transform(X_clean.values)

        # Fit Gaussian HMM
        self.model = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=self.n_iter,
            random_state=self.random_state,
            verbose=False,
        )
        self.model.fit(X_scaled)

        if not self.model.monitor_.converged:
            logger.warning("HMM did NOT converge — consider increasing n_iter")
        else:
            logger.info(f"HMM converged in {self.model.monitor_.iter} iterations")

        # Decode states
        states = self.model.predict(X_scaled)

        # Map states to semantic labels using mean returns
        returns = df.loc[X_clean.index, returns_col]
        self._map_regimes(states, returns, X_clean.index)

        logger.info(f"Regime distribution: {pd.Series(states).value_counts().to_dict()}")
        return self

    def _map_regimes(
        self, states: np.ndarray, returns: pd.Series, index: pd.DatetimeIndex
    ):
        """
        Map HMM numerical states to semantic regime labels.
        Sort by (mean_return, -mean_vol) to assign Bull/Recovery/Bear/Crisis.
        """
        regime_stats = []
        for state in range(self.n_regimes):
            mask = states == state
            state_returns = returns.loc[index[mask]]
            regime_stats.append({
                "state": state,
                "mean_return": state_returns.mean(),
                "std_return": state_returns.std(),
                "count": mask.sum(),
            })

        # Sort: highest mean return first
        regime_stats.sort(key=lambda x: x["mean_return"], reverse=True)

        labels = list(REGIME_NAMES.values())[:self.n_regimes]
        self.regime_mapping = {}
        for i, stat in enumerate(regime_stats):
            label = labels[i] if i < len(labels) else f"Regime_{stat['state']}"
            self.regime_mapping[stat["state"]] = label
            logger.info(
                f"  State {stat['state']} → {label}: "
                f"mean_ret={stat['mean_return']:.6f}, "
                f"std={stat['std_return']:.6f}, "
                f"n={stat['count']}"
            )

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict regime labels and probabilities for the given data.

        Returns DataFrame with columns: regime_label, regime_id, and
        probability columns for each regime.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted — call fit() first")

        X_raw = df[self.feature_names].copy()
        X_clean = X_raw.dropna()
        X_scaled = self.scaler.transform(X_clean.values)

        # Predict states and probabilities
        states = self.model.predict(X_scaled)
        proba = self.model.predict_proba(X_scaled)

        # Build results DataFrame
        result = pd.DataFrame(index=X_clean.index)
        result["regime_id"] = states
        result["regime_label"] = [self.regime_mapping.get(s, f"Unknown_{s}") for s in states]

        # Regime probabilities
        for state, label in self.regime_mapping.items():
            if state < proba.shape[1]:
                result[f"prob_{label}"] = proba[:, state]

        # Apply minimum holding period (5 days) to prevent flickering
        result["regime_label_stable"] = self._stabilize_regimes(
            result["regime_label"], min_days=5
        )

        return result

    def _stabilize_regimes(self, regimes: pd.Series, min_days: int = 5) -> pd.Series:
        """
        Apply minimum holding period to prevent rapid regime switching.
        A regime change must persist for `min_days` before it is accepted.
        """
        stable = regimes.copy()
        current_regime = stable.iloc[0]
        days_in_regime = 0

        for i in range(len(stable)):
            if stable.iloc[i] == current_regime:
                days_in_regime += 1
            else:
                if days_in_regime >= min_days:
                    # Previous regime was stable, try new regime
                    current_regime = stable.iloc[i]
                    days_in_regime = 1
                else:
                    # Revert to current stable regime
                    stable.iloc[i] = current_regime
                    days_in_regime += 1

        return stable

    def predict_next_regime(self, df: pd.DataFrame) -> dict:
        """
        Predict tomorrow's most likely regime based on today's features.
        Uses the HMM's transition matrix for one-step-ahead prediction.
        """
        result = self.predict(df.tail(1))
        current_state = result["regime_id"].iloc[-1]

        # Get transition probabilities from current state
        trans_probs = self.model.transmat_[current_state]
        next_state = np.argmax(trans_probs)
        next_label = self.regime_mapping.get(next_state, f"Unknown_{next_state}")

        return {
            "current_regime": result["regime_label"].iloc[-1],
            "predicted_next_regime": next_label,
            "next_regime_prob": float(trans_probs[next_state]),
            "all_probs": {
                self.regime_mapping.get(s, f"State_{s}"): float(trans_probs[s])
                for s in range(self.n_regimes)
            },
        }

    def save(self, path: Path = None):
        """Save the fitted model, scaler, and regime mapping."""
        if path is None:
            path = OUTPUT_MODELS
        path.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.model, path / "hmm_model.pkl")
        joblib.dump(self.scaler, path / "hmm_scaler.pkl")
        joblib.dump(self.regime_mapping, path / "regime_mapping.pkl")
        joblib.dump(self.feature_names, path / "hmm_features.pkl")
        logger.info(f"Saved HMM model to {path}")

    @classmethod
    def load(cls, path: Path = None) -> "RegimeDetector":
        """Load a previously fitted model."""
        if path is None:
            path = OUTPUT_MODELS

        detector = cls()
        detector.model = joblib.load(path / "hmm_model.pkl")
        detector.scaler = joblib.load(path / "hmm_scaler.pkl")
        detector.regime_mapping = joblib.load(path / "regime_mapping.pkl")
        detector.feature_names = joblib.load(path / "hmm_features.pkl")
        detector.n_regimes = detector.model.n_components
        logger.info(f"Loaded HMM model from {path}")
        return detector


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def run_regime_detection(features_df: pd.DataFrame = None) -> tuple[pd.DataFrame, RegimeDetector]:
    """
    Run the full regime detection pipeline.

    Returns
    -------
    regime_df : pd.DataFrame with regime labels/probabilities
    detector : fitted RegimeDetector instance
    """
    if features_df is None:
        features_df = pd.read_parquet(DATA_PROCESSED / "features.parquet")
        logger.info(f"Loaded features: {features_df.shape}")

    detector = RegimeDetector(n_regimes=4)
    detector.fit(features_df)
    regime_df = detector.predict(features_df)

    # Merge regime info back into features
    features_with_regimes = features_df.join(regime_df, how="left")
    features_with_regimes.to_parquet(DATA_PROCESSED / "features_with_regimes.parquet")

    # Save model
    detector.save()

    # Log summary
    logger.info("\n=== Regime Summary ===")
    for label in regime_df["regime_label_stable"].unique():
        mask = regime_df["regime_label_stable"] == label
        n_days = mask.sum()
        pct = n_days / len(regime_df) * 100
        logger.info(f"  {label}: {n_days} days ({pct:.1f}%)")

    return features_with_regimes, detector


if __name__ == "__main__":
    result_df, det = run_regime_detection()
    print(f"\nRegime DataFrame: {result_df.shape}")
    print(f"Regime distribution:\n{result_df['regime_label_stable'].value_counts()}")

    # Predict next regime
    prediction = det.predict_next_regime(result_df)
    print(f"\nNext regime prediction: {prediction}")
