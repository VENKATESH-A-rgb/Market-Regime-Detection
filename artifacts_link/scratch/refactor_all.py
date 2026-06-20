import os

replacements = [
    ('SPY_Close', 'NSEI_Close'),
    ('SPY_returns', 'NSEI_returns'),
    ('SPY_mom_', 'NSEI_mom_'),
    ('SPY_SMA', 'NSEI_SMA'),
    ('SPY_TLT_ratio', 'NSEI_LIQUID_ratio'),
    ('SPY_GLD_ratio', 'NSEI_GOLD_ratio'),
    ('Market (SPY)', 'Market (NIFTY)'),
    ('SPY benchmark', 'NIFTY benchmark'),
    ('spy_close', 'nsei_close'),
    ('spy_ret', 'nsei_ret'),
    ('TLT_Close', 'LIQUIDBEES.NS_Close'),
    ('GLD_Close', 'GOLDBEES.NS_Close'),
    ('QQQ_Close', 'JUNIORBEES.NS_Close'),
    ('DIA_Close', 'BANKBEES.NS_Close'),
    ('VIX_Close', 'INDIAVIX_Close'),
]

files_to_update = [
    r'd:\Market regime detection\src\backtest.py',
    r'd:\Market regime detection\src\regime_model.py',
    r'd:\Market regime detection\src\specialist_models.py',
    r'd:\Market regime detection\src\explainability.py',
    r'd:\Market regime detection\src\app.py',
    r'd:\Market regime detection\src\portfolio_optimizer.py'
]

for file_path in files_to_update:
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        for old, new in replacements:
            content = content.replace(old, new)
            
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

print("All files updated with NSE/BSE tickers")
