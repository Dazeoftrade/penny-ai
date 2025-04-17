import yfinance as yf
import pandas as pd
import ta
from sklearn.ensemble import RandomForestClassifier
import joblib

df = yf.download("SNDL", start="2022-01-01", end="2024-12-31")
df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
df['macd'] = ta.trend.MACD(df['Close']).macd()
df['ema_10'] = ta.trend.EMAIndicator(df['Close'], window=10).ema_indicator()
df['sma_20'] = ta.trend.SMAIndicator(df['Close'], window=20).sma_indicator()
bb = ta.volatility.BollingerBands(df['Close'])
df['bb_bbm'] = bb.bollinger_mavg()
df['bb_bbh'] = bb.bollinger_hband()
df['bb_bbl'] = bb.bollinger_lband()
df['volume_ema'] = ta.trend.EMAIndicator(df['Volume'], window=10).ema_indicator()
df.dropna(inplace=True)

df['high_flag'] = (df['High'] == df['High'].rolling(window=5, center=True).max()).astype(int)
df['low_flag'] = (df['Low'] == df['Low'].rolling(window=5, center=True).min()).astype(int)

features = ['Open', 'High', 'Low', 'Close', 'Volume', 'rsi', 'macd', 'ema_10', 'sma_20', 'bb_bbm', 'bb_bbh', 'bb_bbl', 'volume_ema']
X = df[features]
y_high = df['high_flag']
y_low = df['low_flag']

high_model = RandomForestClassifier(n_estimators=100).fit(X, y_high)
low_model = RandomForestClassifier(n_estimators=100).fit(X, y_low)

joblib.dump(high_model, "high_model.pkl")
joblib.dump(low_model, "low_model.pkl")

print("✅ Models trained and saved!")