from nsepython import index_history
import traceback
try:
    df = index_history('NIFTY 50', '01-Jan-1993', '31-Dec-1994')
    print(df.head())
except Exception as e:
    traceback.print_exc()
