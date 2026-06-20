# Implementation Plan — Debugging the Market Regime Detection Project

This plan documents the critical fixes required to resolve errors, warnings, and missing data displays across the Streamlit app and machine learning pipeline.

## User Review Required

No breaking configuration or API changes are introduced. The modifications strictly fix code bugs, handle file cache invalidation dynamically, improve numeric stability for portfolio optimization, and restore the Random Forest specialist training.

---

## Proposed Changes

### 1. Feature Engineering: Eliminate NaN Features
#### [MODIFY] [feature_engineering.py](file:///d:/Market%20regime%20detection/src/feature_engineering.py)
* Identify and drop any feature columns that are entirely NaN (such as `BTC_mom_30d` which is initialized to `np.nan` because Bitcoin prices are not downloaded).
* This prevents `specialist_models.py` from dropping all rows due to `dropna()`, which currently results in `0 specialists` trained.

### 2. Specialist Models: Fix Sklearn Feature Name Warnings
#### [MODIFY] [specialist_models.py](file:///d:/Market%20regime%20detection/src/specialist_models.py)
* In `predict_batch`, avoid converting the feature slice `X_subset` to raw numpy `.values`. Keep it as a pandas DataFrame so that `RandomForestClassifier.predict` has matching feature names and runs without throwing warnings.

### 3. Portfolio Optimizer: Clean Return Matrix & Dynamic Asset Filtering
#### [MODIFY] [portfolio_optimizer.py](file:///d:/Market%20regime%20detection/src/portfolio_optimizer.py)
* Add a `lookback` check to `_get_asset_columns` so we only include assets that have sufficient history (e.g. >= 126 trading days) in the *active window* rather than the whole history. This prevents new or inactive assets from corrupting the covariance calculations.
* Update `_get_returns` to replace `inf` and `-inf` returns with `NaN`, and apply a robust `.ffill().fillna(0.0)` clean up. This ensures a clean numeric input matrix is passed to PyPortfolioOpt, preventing the "Input contains NaN" warnings and fallback triggers.

### 4. Streamlit App: Fix Slider Crash & File Cache Invalidation
#### [MODIFY] [app.py](file:///d:/Market%20regime%20detection/src/app.py)
* In `load_data` and `load_audit_logs`, wrap the `@st.cache_data` decorated loaders in helper functions that fetch the file modification time (`st_mtime`). Pass these times as arguments to the cached loaders. When the files are newly created or updated by running the pipeline, the cache is automatically invalidated and reloaded.
* In the audit log slider filter, handle the case where `min_conf` or `max_conf` are `NaN` (due to empty/NaN columns in the audit log) using `pd.isna`.

---

## Verification Plan

### Automated Tests
1. Execute the pipeline:
   `venv\Scripts\python run_pipeline.py`
   * Confirm that Random Forest specialists are successfully trained (`Trained 4 specialists`).
   * Confirm that SHAP explainability runs and saves summary plots for all regimes.
   * Verify that there are no "Input contains NaN" warnings from PyPortfolioOpt.
   * Verify that there are no sklearn feature name warnings.

### Manual Verification
1. Run the Streamlit dashboard:
   `venv\Scripts\streamlit run src/app.py`
   * Verify that the dashboard launches without any slider exceptions.
   * Verify that the Price & Regimes tab displays the plotly chart and timeline correctly without the "No price data available" warning.
   * Verify that the audit log explorer works and filters correctly.
