"""
run_pipeline.py — End-to-End Pipeline Runner
Orchestrates all 10 steps: data → features → HMM → RF → backtest → SHAP.
"""

import sys
import time
import logging
from pathlib import Path

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pipeline")


def main():
    total_start = time.time()

    # ── Step 2: Data Acquisition ──────────────────────────────
    logger.info("=" * 70)
    logger.info("STEP 2: DATA ACQUISITION")
    logger.info("=" * 70)
    from src.data_loader import run_data_pipeline
    master = run_data_pipeline()
    logger.info(f"  → Master dataset: {master.shape}")

    # ── Step 3: Feature Engineering + FFD ─────────────────────
    logger.info("=" * 70)
    logger.info("STEP 3: FEATURE ENGINEERING & FFD")
    logger.info("=" * 70)
    from src.feature_engineering import run_feature_pipeline
    features_df, ffd_params = run_feature_pipeline(master)
    logger.info(f"  → Features: {features_df.shape}, FFD params: {ffd_params}")

    # ── Step 4: Regime Detection (HMM) ────────────────────────
    logger.info("=" * 70)
    logger.info("STEP 4: REGIME DETECTION (HMM)")
    logger.info("=" * 70)
    from src.regime_model import run_regime_detection
    features_with_regimes, detector = run_regime_detection(features_df)
    logger.info(f"  → Regimes assigned: {features_with_regimes['regime_label_stable'].value_counts().to_dict()}")

    # ── Step 5: Specialist Models (RF per regime) ─────────────
    logger.info("=" * 70)
    logger.info("STEP 5: SPECIALIST RANDOM FORESTS")
    logger.info("=" * 70)
    from src.specialist_models import run_specialist_training
    signals_df, ensemble = run_specialist_training(
        features_with_regimes, regime_detector=detector
    )
    logger.info(f"  → Signals: {signals_df['signal'].value_counts().to_dict()}")

    # ── Step 6+7: Portfolio Optimization + Backtest ───────────
    logger.info("=" * 70)
    logger.info("STEP 6+7: BACKTEST (WALK-FORWARD OPTIMIZATION)")
    logger.info("=" * 70)
    from src.backtest import run_backtest
    oos_df, metrics = run_backtest(features_df)

    # ── Step 8: SHAP Explainability ───────────────────────────
    logger.info("=" * 70)
    logger.info("STEP 8: SHAP EXPLAINABILITY")
    logger.info("=" * 70)
    from src.explainability import run_explainability
    explainer = run_explainability(features_with_regimes, signals_df)

    # ── Summary ───────────────────────────────────────────────
    elapsed = time.time() - total_start
    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Total time:         {elapsed/60:.1f} minutes")
    logger.info(f"  Data shape:         {master.shape}")
    logger.info(f"  Features:           {features_df.shape[1]} columns")
    logger.info(f"  Regimes:            {features_with_regimes['regime_label_stable'].nunique()}")
    logger.info(f"  Specialists:        {len(ensemble.specialists)}")
    logger.info(f"  OOS days:           {len(oos_df)}")
    logger.info(f"  Sharpe Ratio:       {metrics['sharpe_ratio']}")
    logger.info(f"  Max Drawdown:       {metrics['max_drawdown']:.2%}")
    logger.info(f"  Annual Return:      {metrics['annual_return']:.2%}")
    logger.info(f"  Win Rate:           {metrics['win_rate']:.2%}")
    logger.info("=" * 70)

    return metrics


if __name__ == "__main__":
    main()
