# Market Regime Detection & Adaptive Portfolio Allocation

> An autonomous quantitative finance pipeline that detects latent market states using unsupervised learning to dynamically optimize asset allocation and execute regime-specific trading models.

## 📖 Overview
Financial markets do not exhibit stationary behavior; they transition between distinct behavioral regimes (e.g., low-volatility bull markets and high-volatility crash states). This project implements an MVP for an adaptive algorithmic trading architecture. By leveraging a Gaussian Hidden Markov Model (HMM), the system classifies market environments in real-time. This regime prediction then dictates which specialized Machine Learning model generates trading signals and determines the mathematical objective function for portfolio optimization.

To ensure compliance with emerging algorithmic trading frameworks (such as the SEBI 2026 regulations), the system includes full SHAP-based interpretability to explain automated capital rotation decisions.

## ✨ Core Pipeline & Architecture

### 1. Data Pipeline & Feature Engineering
* **Multi-Asset Universe:** Ingests daily price data via `yfinance` (SPY, ^VIX, TLT, BTC-USD, GLD) alongside macroeconomic indicators from `fredapi` (10Y Yield, 2Y Yield, Fed Funds Rate).
* **Fractional Differentiation:** Transforms financial time series into stationary datasets for machine learning while preserving historical market memory (fixed-width window method).
* **Technical Indicators:** Computes rolling 20-day and 60-day volatility profiles and yield curve slopes.

### 2. Unsupervised Regime Detection
* **Gaussian HMM Engine:** Trains an unsupervised Hidden Markov Model (`hmmlearn`) on volatility and macro features to classify the market into discrete latent states (e.g., Regime 0: Bull/Low-Vol, Regime 1: Crisis/High-Vol).

### 3. Adaptive Machine Learning
* **Specialist Classifiers:** Deploys regime-specific Random Forest models. 
  * *Model 0* is trained exclusively on data from Regime 0.
  * *Model 1* is trained exclusively on data from Regime 1.
* Today's HMM state prediction determines which specialist model is activated for tomorrow's signal generation.

### 4. Dynamic Portfolio Optimization
Utilizes `PyPortfolioOpt` (backed by `cvxpy`) for regime-aware capital allocation:
* **Bull Regime:** Optimizes for the Maximum Sharpe Ratio ($S = \frac{R_p - R_f}{\sigma_p}$) with relaxed equity bounds.
* **Bear/Crisis Regime:** Enforces strict risk limits (max 40% per asset) and optimizes for Minimum Volatility to automatically rotate capital into safe-haven assets (TLT, GLD).

### 5. High-Fidelity Backtesting & Explainability
* **Walk-Forward Optimization (WFO):** Uses rolling training windows to prevent look-ahead bias and overfitting.
* **VectorBT Execution:** Simulates real-world market friction by accounting for explicit transaction fees and slippage using `vectorbt.Portfolio.from_signals()`.
* **Regulatory Transparency (SHAP):** Generates SHapley Additive exPlanations to visually mathematically demonstrate which feature spikes (e.g., VIX) triggered a transition into a defensive portfolio allocation.

## 🛠️ Tech Stack
* **Language:** Python 3.10+
* **Data Gathering:** `yfinance`, `fredapi`
* **Data Processing & Math:** `pandas`, `numpy`
* **Quantitative Modeling:** `hmmlearn`, `scikit-learn`
* **Portfolio Optimization:** `PyPortfolioOpt`, `cvxpy`
* **Backtesting Engine:** `vectorbt`
* **Interpretability:** `shap`
* **Visualization/UI:** `streamlit`

## 🚀 Getting Started

### Prerequisites
* Python 3.10 or higher.
* An active internet connection for real-time Yahoo Finance and FRED API calls.

### Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/adaptive-regime-portfolio.git](https://github.com/yourusername/adaptive-regime-portfolio.git)
   cd adaptive-regime-portfolio