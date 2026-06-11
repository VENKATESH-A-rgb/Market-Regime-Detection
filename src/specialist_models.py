"""
specialist_models.py — Step 5: Specialist Predictive Models
Per-regime Random Forest classifiers that generate trading signals.
HMM predicts tomorrow's regime → corresponding RF specialist is activated.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUT_MODELS = PROJECT_ROOT / "output" / "models"

# Features used for Random Forest prediction (exclude target, regime, raw prices)
EXCLUDE_PATTERNS = [
    "_Close", "_Open", "_High", "_Low", "_Volume",
    "regime_", "prob_", "target_", "Date",
    "CPIAUCSL", "UNRATE", "DGS10", "DGS2", "FEDFUNDS",
]


def get_rf_features(df: pd.DataFrame) -> list[str]:
    """Select features suitable for RF training (engineered features only)."""
    features = []
    for col in df.columns:
        if any(pat in col for pat in EXCLUDE_PATTERNS):
            continue
        if df[col].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]:
            features.append(col)
    return features


class RegimeSpecialistEnsemble:
    """
    Ensemble of regime-specific Random Forest classifiers.

    Each specialist is trained only on data from its assigned regime,
    learning regime-specific patterns for signal generation.
    """

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 8,
        min_samples_leaf: int = 20,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.specialists = {}  # regime_label → fitted RandomForestClassifier
        self.feature_names = []
        self.feature_importances = {}  # regime_label → importance dict
        self.regime_metrics = {}  # regime_label → classification metrics

    def fit(
        self,
        df: pd.DataFrame,
        regime_col: str = "regime_label_stable",
        target_col: str = "target_next_day",
    ) -> "RegimeSpecialistEnsemble":
        """
        Train one RF specialist per regime.

        Parameters
        ----------
        df : pd.DataFrame
            Feature-engineered DataFrame with regime labels and target.
        regime_col : str
            Column containing regime labels.
        target_col : str
            Binary target column (1 = up, 0 = down).
        """
        logger.info("Training regime-specialist Random Forests...")

        self.feature_names = get_rf_features(df)
        logger.info(f"Using {len(self.feature_names)} features for RF")

        regimes = df[regime_col].dropna().unique()

        for regime in regimes:
            logger.info(f"\n--- Training specialist: {regime} ---")

            # Filter data for this regime
            mask = df[regime_col] == regime
            regime_data = df.loc[mask].copy()

            X = regime_data[self.feature_names].copy()
            y = regime_data[target_col].copy()

            # Drop rows with NaN in features or target
            valid_mask = X.notna().all(axis=1) & y.notna()
            X = X[valid_mask]
            y = y[valid_mask]

            if len(X) < 100:
                logger.warning(
                    f"  {regime}: only {len(X)} samples — skipping specialist"
                )
                continue

            logger.info(f"  {regime}: {len(X)} samples, class balance: "
                       f"{y.value_counts().to_dict()}")

            # Train RF
            rf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                class_weight="balanced",
                random_state=self.random_state,
                n_jobs=-1,
            )
            rf.fit(X, y)
            self.specialists[regime] = rf

            # Cross-validation score
            cv_scores = cross_val_score(rf, X, y, cv=5, scoring="accuracy")
            logger.info(f"  {regime}: CV accuracy = {cv_scores.mean():.4f} "
                       f"(±{cv_scores.std():.4f})")

            # Feature importances
            importances = pd.Series(
                rf.feature_importances_, index=self.feature_names
            ).sort_values(ascending=False)
            self.feature_importances[regime] = importances
            logger.info(f"  Top 5 features: {importances.head().to_dict()}")

            # Store metrics
            self.regime_metrics[regime] = {
                "n_samples": len(X),
                "cv_accuracy_mean": float(cv_scores.mean()),
                "cv_accuracy_std": float(cv_scores.std()),
                "class_balance": y.value_counts().to_dict(),
            }

        logger.info(f"\nTrained {len(self.specialists)} specialists: "
                    f"{list(self.specialists.keys())}")
        return self

    def predict_signal(
        self,
        features: pd.DataFrame,
        regime_label: str,
        regime_probs: dict = None,
    ) -> dict:
        """
        Generate a trading signal using the appropriate specialist.

        Parameters
        ----------
        features : pd.DataFrame
            Single row (or few rows) of feature data.
        regime_label : str
            The predicted regime label.
        regime_probs : dict, optional
            Probability of each regime (for ensemble fallback).

        Returns
        -------
        dict with keys: signal, confidence, regime, method
        """
        X = features[self.feature_names].copy()
        valid_mask = X.notna().all(axis=1)
        X = X[valid_mask]

        if X.empty:
            return {"signal": 0, "confidence": 0.0, "regime": regime_label,
                    "method": "no_data"}

        # Try the designated specialist
        if regime_label in self.specialists:
            rf = self.specialists[regime_label]
            signal = rf.predict(X)[-1]
            proba = rf.predict_proba(X)[-1]
            confidence = float(max(proba))

            return {
                "signal": int(signal) * 2 - 1,  # Convert 0/1 to -1/+1
                "confidence": confidence,
                "regime": regime_label,
                "method": "specialist",
            }

        # Fallback: ensemble vote weighted by regime probabilities
        if regime_probs:
            return self._ensemble_predict(X, regime_probs)

        # Last resort: use any available specialist
        for available_regime, rf in self.specialists.items():
            signal = rf.predict(X)[-1]
            proba = rf.predict_proba(X)[-1]
            return {
                "signal": int(signal) * 2 - 1,
                "confidence": float(max(proba)),
                "regime": regime_label,
                "method": f"fallback_{available_regime}",
            }

        return {"signal": 0, "confidence": 0.0, "regime": regime_label,
                "method": "no_specialist"}

    def _ensemble_predict(self, X: pd.DataFrame, regime_probs: dict) -> dict:
        """Weighted ensemble prediction across all specialists."""
        weighted_signal = 0.0
        total_weight = 0.0

        for regime, prob in regime_probs.items():
            if regime in self.specialists:
                rf = self.specialists[regime]
                proba = rf.predict_proba(X)[-1]
                signal = proba[1] - proba[0]  # Continuous signal
                weighted_signal += signal * prob
                total_weight += prob

        if total_weight > 0:
            weighted_signal /= total_weight

        return {
            "signal": 1 if weighted_signal > 0 else -1,
            "confidence": abs(float(weighted_signal)),
            "regime": "ensemble",
            "method": "ensemble_weighted",
        }

    def predict_batch(
        self,
        df: pd.DataFrame,
        regime_col: str = "regime_label_stable",
        regime_detector=None,
    ) -> pd.DataFrame:
        """
        Generate signals for the entire dataset in a fast, vectorized manner.

        Uses the HMM transition matrix to predict **tomorrow's** most likely
        regime, then routes to that specialist RF for signal generation.
        This implements the spec: "today's HMM prediction dictates which
        specialist Random Forest is used to generate tomorrow's trading signal."

        Parameters
        ----------
        df : pd.DataFrame
            Feature-engineered DataFrame with regime labels.
        regime_col : str
            Column containing regime labels.
        regime_detector : RegimeDetector, optional
            Fitted HMM detector. If provided, uses transition matrix
            to predict next-day regime for specialist routing.

        Returns DataFrame with columns: signal, confidence, method.
        """
        logger.info("Generating batch signals...")
        
        n_rows = len(df)
        signals = np.zeros(n_rows, dtype=int)
        confidences = np.zeros(n_rows, dtype=float)
        regimes_used = ["unknown"] * n_rows
        predicted_next_regimes = ["unknown"] * n_rows
        methods = ["missing_regime"] * n_rows
        current_regimes = df[regime_col].astype(str).tolist()
        
        # 1. Precompute transition matrix routing mapping
        next_regime_map = {}
        if regime_detector is not None and regime_detector.model is not None:
            reverse_mapping = {
                label: state for state, label in regime_detector.regime_mapping.items()
            }
            for label, current_state in reverse_mapping.items():
                trans_probs = regime_detector.model.transmat_[current_state]
                next_state = int(np.argmax(trans_probs))
                next_label = regime_detector.regime_mapping.get(next_state, label)
                next_regime_map[label] = next_label
                
        # Fill predicted_next_regimes
        for i in range(n_rows):
            regime = df.iloc[i].get(regime_col)
            if pd.notna(regime) and regime != "unknown":
                predicted_next_regimes[i] = next_regime_map.get(regime, regime)
                methods[i] = "specialist"
            else:
                methods[i] = "missing_regime"
                
        # 2. Get features
        X_all = df[self.feature_names].copy()
        
        # Identify valid rows (no NaN in features)
        valid_mask = X_all.notna().all(axis=1).values
        
        # Mark rows with NaN as no_data
        for i in range(n_rows):
            if not valid_mask[i] and methods[i] != "missing_regime":
                methods[i] = "no_data"
                regimes_used[i] = predicted_next_regimes[i]
                
        # For valid rows, group by predicted_next_regime and run batch prediction
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) > 0:
            regime_to_indices = {}
            for idx in valid_indices:
                next_r = predicted_next_regimes[idx]
                if next_r not in regime_to_indices:
                    regime_to_indices[next_r] = []
                regime_to_indices[next_r].append(idx)
                
            for regime_label, indices in regime_to_indices.items():
                indices = np.array(indices)
                X_subset = X_all.iloc[indices].values
                
                if regime_label in self.specialists:
                    rf = self.specialists[regime_label]
                    preds = rf.predict(X_subset)
                    probas = rf.predict_proba(X_subset)
                    
                    signals[indices] = preds * 2 - 1  # Convert 0/1 to -1/+1
                    confidences[indices] = np.max(probas, axis=1)
                    for idx in indices:
                        regimes_used[idx] = regime_label
                        methods[idx] = "specialist"
                else:
                    available_specialist_names = list(self.specialists.keys())
                    if available_specialist_names:
                        fallback_regime = available_specialist_names[0]
                        rf = self.specialists[fallback_regime]
                        preds = rf.predict(X_subset)
                        probas = rf.predict_proba(X_subset)
                        
                        signals[indices] = preds * 2 - 1
                        confidences[indices] = np.max(probas, axis=1)
                        for idx in indices:
                            regimes_used[idx] = regime_label
                            methods[idx] = f"fallback_{fallback_regime}"
                    else:
                        for idx in indices:
                            regimes_used[idx] = regime_label
                            methods[idx] = "no_specialist"
                            
        # Create output DataFrame
        signals_df = pd.DataFrame({
            "signal": signals,
            "confidence": confidences,
            "regime_used": regimes_used,
            "predicted_next_regime": predicted_next_regimes,
            "current_regime": current_regimes,
            "method": methods
        }, index=df.index)
        
        logger.info(f"Signal distribution: {signals_df['signal'].value_counts().to_dict()}")
        logger.info(f"Next-regime routing: {signals_df['predicted_next_regime'].value_counts().to_dict()}")
        return signals_df

    def _predict_next_regime_label(
        self,
        current_regime: str,
        regime_detector=None,
    ) -> str:
        """
        Predict tomorrow's most likely regime using the HMM transition matrix.

        Parameters
        ----------
        current_regime : str
            Today's regime label (e.g., "Bull", "Crisis").
        regime_detector : RegimeDetector, optional
            Fitted HMM detector with transition matrix.

        Returns
        -------
        str : Predicted next-day regime label.
        """
        if regime_detector is None or regime_detector.model is None:
            # Fallback: use current regime if no detector available
            return current_regime

        # Find the HMM state ID for the current regime label
        reverse_mapping = {
            label: state for state, label in regime_detector.regime_mapping.items()
        }
        current_state = reverse_mapping.get(current_regime)
        if current_state is None:
            return current_regime

        # Use HMM transition matrix to predict next state
        trans_probs = regime_detector.model.transmat_[current_state]
        next_state = int(np.argmax(trans_probs))
        next_label = regime_detector.regime_mapping.get(next_state, current_regime)

        return next_label

    def save(self, path: Path = None):
        """Save all specialists and metadata."""
        if path is None:
            path = OUTPUT_MODELS
        path.mkdir(parents=True, exist_ok=True)

        for regime, rf in self.specialists.items():
            safe_name = regime.lower().replace(" ", "_")
            joblib.dump(rf, path / f"rf_specialist_{safe_name}.pkl")

        joblib.dump(self.feature_names, path / "rf_feature_names.pkl")
        joblib.dump(self.feature_importances, path / "rf_feature_importances.pkl")
        joblib.dump(self.regime_metrics, path / "rf_metrics.pkl")
        joblib.dump(list(self.specialists.keys()), path / "rf_regime_list.pkl")
        logger.info(f"Saved {len(self.specialists)} specialists to {path}")

    @classmethod
    def load(cls, path: Path = None) -> "RegimeSpecialistEnsemble":
        """Load previously trained specialists."""
        if path is None:
            path = OUTPUT_MODELS

        ensemble = cls()
        ensemble.feature_names = joblib.load(path / "rf_feature_names.pkl")
        ensemble.feature_importances = joblib.load(path / "rf_feature_importances.pkl")
        ensemble.regime_metrics = joblib.load(path / "rf_metrics.pkl")

        regime_list = joblib.load(path / "rf_regime_list.pkl")
        for regime in regime_list:
            safe_name = regime.lower().replace(" ", "_")
            ensemble.specialists[regime] = joblib.load(
                path / f"rf_specialist_{safe_name}.pkl"
            )

        logger.info(f"Loaded {len(ensemble.specialists)} specialists from {path}")
        return ensemble


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def run_specialist_training(
    features_df: pd.DataFrame = None,
    regime_detector=None,
) -> tuple[pd.DataFrame, RegimeSpecialistEnsemble]:
    """
    Run the specialist model training pipeline.

    Parameters
    ----------
    features_df : pd.DataFrame, optional
        Feature DataFrame with regime labels.
    regime_detector : RegimeDetector, optional
        Fitted HMM detector for next-day regime prediction routing.

    Returns
    -------
    signals_df : DataFrame with signals for each day
    ensemble : fitted RegimeSpecialistEnsemble
    """
    if features_df is None:
        features_df = pd.read_parquet(DATA_PROCESSED / "features_with_regimes.parquet")
        logger.info(f"Loaded features with regimes: {features_df.shape}")

    ensemble = RegimeSpecialistEnsemble()
    ensemble.fit(features_df)

    signals_df = ensemble.predict_batch(
        features_df, regime_detector=regime_detector
    )
    ensemble.save()

    # Merge signals back
    full_df = features_df.join(signals_df, how="left")
    full_df.to_parquet(DATA_PROCESSED / "features_with_signals.parquet")

    return signals_df, ensemble


if __name__ == "__main__":
    signals, ens = run_specialist_training()
    print(f"\nSignals: {signals.shape}")
    print(signals["signal"].value_counts())
    print(f"\nRegime metrics:")
    for regime, metrics in ens.regime_metrics.items():
        print(f"  {regime}: {metrics}")
