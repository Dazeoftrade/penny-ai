import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import matplotlib.pyplot as plt
import joblib

low_model = joblib.load("low_model.pkl")
high_model = joblib.load("high_model.pkl")

def get_data(ticker):
    df = yf.download(ticker, period='3mo', interval='1h')
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
    return df

def predict(df):
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'rsi', 'macd',
                'ema_10', 'sma_20', 'bb_bbm', 'bb_bbh', 'bb_bbl', 'volume_ema']
    df['low_pred'] = low_model.predict(df[features])
    df['high_pred'] = high_model.predict(df[features])
    return df

st.title("📈 Penny Stock High/Low Predictor")
ticker = st.text_input("Enter Stock Ticker:", "SNDL")

if st.button("Predict Now"):
    df = get_data(ticker.upper())
    df = predict(df)

    st.success("✅ Prediction complete!")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['Close'], label='Close', alpha=0.7)
    ax.scatter(df[df['low_pred'] == 1].index, df[df['low_pred'] == 1]['Close'], label='Buy Signal', marker='^', color='green')
    ax.scatter(df[df['high_pred'] == 1].index, df[df['high_pred'] == 1]['Close'], label='Sell Signal', marker='v', color='red')
    ax.legend()
    st.pyplot(fig)