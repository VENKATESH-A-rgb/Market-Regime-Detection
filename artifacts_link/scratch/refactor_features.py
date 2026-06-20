import re
import os

path = r'd:\Market regime detection\src\feature_engineering.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace variables and column prefixes
replacements = [
    ('SPY_Close', 'NSEI_Close'),
    ('SPY_Volume', 'NSEI_Volume'),
    ('SPY_SMA', 'NSEI_SMA'),
    ('SPY_mom_', 'NSEI_mom_'),
    ('SPY_returns', 'NSEI_returns'),
    ('spy_close', 'nsei_close'),
    ('spy_returns', 'nsei_returns'),
    ('spy_vol', 'nsei_vol'),
    ('spy_ret', 'nsei_ret'),
    ('TLT_Close', 'LIQUIDBEES.NS_Close'),
    ('tlt_ret', 'liquid_ret'),
    ('GLD_Close', 'GOLDBEES.NS_Close'),
    ('gld_ret', 'gold_ret'),
    ('QQQ_Close', 'JUNIORBEES.NS_Close'),
    ('qqq_ret', 'junior_ret'),
    ('DIA_Close', 'BANKBEES.NS_Close'),
    ('dia_ret', 'bank_ret'),
    ('SPY_TLT_ratio', 'NSEI_LIQUID_ratio'),
    ('SPY_GLD_ratio', 'NSEI_GOLD_ratio'),
    ('corr_SPY_TLT_60d', 'corr_NSEI_LIQUID_60d'),
    ('corr_SPY_GLD_60d', 'corr_NSEI_GOLD_60d'),
    ('VIX_Close', 'INDIAVIX_Close'),
]

for old, new in replacements:
    content = content.replace(old, new)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print("feature_engineering.py updated")
