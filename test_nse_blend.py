import pandas as pd
import yfinance as yf
from nsepython import index_history

def fetch_nifty_full():
    # 1. Fetch yfinance
    yf_df = yf.download("^NSEI", start="1990-01-01", auto_adjust=True, progress=False)
    yf_df.index = pd.to_datetime(yf_df.index)
    
    # 2. Fetch nsepython
    nse_df = index_history("NIFTY 50", "01-Jan-1990", "31-Dec-2007")
    if nse_df is not None and not nse_df.empty:
        nse_df["Date"] = pd.to_datetime(nse_df["HistoricalDate"])
        nse_df = nse_df.set_index("Date")
        
        # Format columns to match yfinance
        # Convert strings to numeric, coerce errors to NaN
        for col in ["OPEN", "HIGH", "LOW", "CLOSE"]:
            nse_df[col] = pd.to_numeric(nse_df[col], errors="coerce")
            
        nse_mapped = pd.DataFrame({
            "Open": nse_df["OPEN"],
            "High": nse_df["HIGH"],
            "Low": nse_df["LOW"],
            "Close": nse_df["CLOSE"],
            "Volume": 0
        })
        
        # 3. Combine
        combined = pd.concat([nse_mapped, yf_df])
        # drop duplicates based on index
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.sort_index(inplace=True)
        return combined
    return yf_df

df = fetch_nifty_full()
print(df.head())
print(df.shape)
print("Min date:", df.index.min())
