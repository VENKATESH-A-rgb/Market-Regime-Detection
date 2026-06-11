"""
explainability.py — Step 8: SEBI 2026 Compliance & Explainability
SHAP-based model interpretability and audit logging.
"""

import json
import logging
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

# Ensure project root is on path for sibling imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUT_PLOTS = PROJECT_ROOT / "output" / "plots"
OUTPUT_LOGS = PROJECT_ROOT / "output" / "logs"
OUTPUT_MODELS = PROJECT_ROOT / "output" / "models"


class SHAPExplainer:
    """
    SHAP-based explainability for the regime-specialist RF models.
    Generates both global and per-decision explanations for SEBI 2026 compliance.
    """

    def __init__(self):
        self.explainers = {}  # regime → shap.TreeExplainer
        self.shap_values = {}  # regime → shap_values array
        self.feature_names = []

    def compute_explanations(
        self,
        ensemble,
        features_df: pd.DataFrame,
        regime_col: str = "regime_label_stable",
        max_samples: int = 1000,
    ):
        """
        Compute SHAP values for each regime specialist.

        Parameters
        ----------
        ensemble : RegimeSpecialistEnsemble
            Fitted specialist ensemble.
        features_df : pd.DataFrame
            Full feature dataset with regime labels.
        max_samples : int
            Max samples per regime for SHAP computation (for speed).
        """
        import shap

        logger.info("Computing SHAP explanations...")
        self.feature_names = ensemble.feature_names

        for regime, rf_model in ensemble.specialists.items():
            logger.info(f"  Computing SHAP for regime: {regime}")

            # Get data for this regime
            mask = features_df[regime_col] == regime
            regime_data = features_df.loc[mask, self.feature_names].dropna()

            if len(regime_data) == 0:
                logger.warning(f"  No data for regime {regime}")
                continue

            # Subsample if too large
            if len(regime_data) > max_samples:
                regime_data = regime_data.sample(max_samples, random_state=42)

            # Create TreeExplainer (exact SHAP values for tree models)
            explainer = shap.TreeExplainer(rf_model)
            self.explainers[regime] = explainer

            # Compute SHAP values
            sv = explainer.shap_values(regime_data.values)
            # For binary classification, take class 1 SHAP values
            if isinstance(sv, list) and len(sv) == 2:
                self.shap_values[regime] = sv[1]
            elif isinstance(sv, np.ndarray) and len(sv.shape) == 3 and sv.shape[2] == 2:
                self.shap_values[regime] = sv[:, :, 1]
            else:
                self.shap_values[regime] = sv

            logger.info(f"  {regime}: SHAP computed for {len(regime_data)} samples")

    def generate_plots(self):
        """Generate and save SHAP summary plots."""
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        OUTPUT_PLOTS.mkdir(parents=True, exist_ok=True)

        for regime, sv in self.shap_values.items():
            safe_name = regime.lower().replace(" ", "_")

            # 1. Summary plot (beeswarm)
            fig, ax = plt.subplots(figsize=(12, 8))
            shap.summary_plot(
                sv,
                feature_names=self.feature_names,
                show=False,
                max_display=20,
            )
            plt.title(f"SHAP Feature Importance — {regime} Regime", fontsize=14)
            plt.tight_layout()
            plt.savefig(OUTPUT_PLOTS / f"shap_summary_{safe_name}.png", dpi=150)
            plt.close()

            # 2. Bar plot (mean |SHAP|)
            fig, ax = plt.subplots(figsize=(12, 8))
            shap.summary_plot(
                sv,
                feature_names=self.feature_names,
                plot_type="bar",
                show=False,
                max_display=20,
            )
            plt.title(f"SHAP Mean |Impact| — {regime} Regime", fontsize=14)
            plt.tight_layout()
            plt.savefig(OUTPUT_PLOTS / f"shap_bar_{safe_name}.png", dpi=150)
            plt.close()

            logger.info(f"  Saved SHAP plots for {regime}")

        # 3. Combined importance across all regimes
        self._generate_combined_importance_plot()

    def _generate_combined_importance_plot(self):
        """Generate a combined feature importance plot across all regimes."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        importance_data = {}
        for regime, sv in self.shap_values.items():
            mean_abs = np.abs(sv).mean(axis=0)
            importance_data[regime] = pd.Series(mean_abs, index=self.feature_names)

        if not importance_data:
            return

        imp_df = pd.DataFrame(importance_data)
        imp_df["total"] = imp_df.sum(axis=1)
        imp_df = imp_df.sort_values("total", ascending=True).tail(20)

        fig, ax = plt.subplots(figsize=(14, 10))
        colors = {"Bull": "#2ecc71", "Recovery": "#3498db", "Bear": "#e67e22", "Crisis": "#e74c3c"}

        bottom = np.zeros(len(imp_df))
        for regime in imp_df.columns:
            if regime == "total":
                continue
            color = colors.get(regime, "#95a5a6")
            ax.barh(imp_df.index, imp_df[regime], left=bottom, label=regime, color=color)
            bottom += imp_df[regime].values

        ax.set_xlabel("Mean |SHAP value|", fontsize=12)
        ax.set_title("Feature Importance Across All Regimes (SHAP)", fontsize=14)
        ax.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(OUTPUT_PLOTS / "shap_combined_importance.png", dpi=150)
        plt.close()
        logger.info("Saved combined SHAP importance plot")

    def generate_audit_log(
        self,
        features_df: pd.DataFrame,
        signals_df: pd.DataFrame,
        regime_col: str = "regime_label_stable",
    ):
        """
        Generate SEBI-compliant audit log (JSONL format).
        Each line documents: date, regime, signal, top SHAP features.
        """
        OUTPUT_LOGS.mkdir(parents=True, exist_ok=True)
        log_path = OUTPUT_LOGS / "shap_audit.jsonl"

        logger.info(f"Generating SHAP audit log → {log_path}")
        n_entries = 0

        with open(log_path, "w") as f:
            for date in signals_df.index:
                if date not in features_df.index:
                    continue

                row = features_df.loc[date]
                regime = row.get(regime_col, "unknown")
                if pd.isna(regime):
                    regime = "unknown"

                signal_info = signals_df.loc[date] if date in signals_df.index else {}

                # Get SHAP values for this prediction if available
                top_features = self._get_top_shap_features(
                    features_df.loc[[date]], regime
                )

                entry = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "date": str(date.date()) if hasattr(date, 'date') else str(date),
                    "regime": str(regime),
                    "signal": int(signal_info.get("signal", 0)) if isinstance(signal_info, (dict, pd.Series)) else 0,
                    "confidence": float(signal_info.get("confidence", 0)) if isinstance(signal_info, (dict, pd.Series)) else 0.0,
                    "method": str(signal_info.get("method", "unknown")) if isinstance(signal_info, (dict, pd.Series)) else "unknown",
                    "top_features": top_features,
                    "model_version": "hmm_rf_v1",
                    "compliance_framework": "SEBI_2026",
                }

                f.write(json.dumps(entry) + "\n")
                n_entries += 1

        logger.info(f"Audit log: {n_entries} entries written to {log_path}")

    def _get_top_shap_features(
        self, features_row: pd.DataFrame, regime: str, top_n: int = 5
    ) -> list[dict]:
        """Get top SHAP features for a single prediction."""
        if regime not in self.explainers:
            return []

        try:
            feature_vals = features_row[self.feature_names].values
            if np.isnan(feature_vals).any():
                return []

            sv = self.explainers[regime].shap_values(feature_vals)
            if isinstance(sv, list) and len(sv) == 2:
                sv = sv[1]
            elif isinstance(sv, np.ndarray) and len(sv.shape) == 3 and sv.shape[2] == 2:
                sv = sv[:, :, 1]

            sv_flat = sv.flatten()
            top_idx = np.argsort(np.abs(sv_flat))[-top_n:][::-1]

            return [
                {
                    "feature": self.feature_names[i],
                    "shap_value": round(float(sv_flat[i]), 6),
                    "feature_value": round(float(feature_vals[0, i]), 6),
                }
                for i in top_idx
            ]
        except Exception:
            return []


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def run_explainability(
    features_df: pd.DataFrame = None,
    signals_df: pd.DataFrame = None,
) -> SHAPExplainer:
    """
    Run the full explainability pipeline.
    """
    from src.specialist_models import RegimeSpecialistEnsemble

    if features_df is None:
        features_df = pd.read_parquet(DATA_PROCESSED / "features_with_regimes.parquet")

    if signals_df is None:
        try:
            full = pd.read_parquet(DATA_PROCESSED / "features_with_signals.parquet")
            sig_cols = ["signal", "confidence", "method"]
            available = [c for c in sig_cols if c in full.columns]
            signals_df = full[available] if available else pd.DataFrame(index=full.index)
        except FileNotFoundError:
            signals_df = pd.DataFrame(index=features_df.index)

    # Load specialist models
    ensemble = RegimeSpecialistEnsemble.load()

    # Compute SHAP
    explainer = SHAPExplainer()
    explainer.compute_explanations(ensemble, features_df)

    # Generate plots
    explainer.generate_plots()

    # Generate audit log
    explainer.generate_audit_log(features_df, signals_df)

    logger.info("Explainability pipeline complete")
    return explainer


if __name__ == "__main__":
    exp = run_explainability()
    print("SHAP explainability complete.")
    print(f"Plots saved to: {OUTPUT_PLOTS}")
    print(f"Audit log saved to: {OUTPUT_LOGS}")
