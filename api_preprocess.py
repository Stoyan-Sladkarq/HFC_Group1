import requests
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def fetch_bitcoin_data():
    """Fetch Bitcoin OHLCV data from Binance API and preprocess it."""
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": "BTCUSDT",
        "interval": "1m",  # 1-minute intervals
        "limit": 500  # Fetch last 500 minutes of data
    }

    response = requests.get(url, params=params)
    data = response.json()

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', '_1', '_2', '_3', '_4', '_5', '_6'])
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]  # Keep only relevant columns

    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    # Convert numerical columns to float
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    df[['open', 'high', 'low', 'close', 'volume']] = scaler.fit_transform(df[['open', 'high', 'low', 'close', 'volume']])

    # Save to CSV for reuse
    df.to_csv("bitcoin_ohlcv.csv", index=False)

    print("Data fetched and saved successfully.")
    return df

# Uncomment to run:
# fetch_bitcoin_data()
