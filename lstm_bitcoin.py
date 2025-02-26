import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load preprocessed data
df = pd.read_csv("bitcoin_ohlcv.csv")

# Select features
data = df[['open', 'high', 'low', 'close', 'volume']].values

# Define function to create LSTM sequences
def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])  # 60-minute sequence
        y.append(data[i+seq_length, 3])  # Predict next minute's closing price
    return np.array(X), np.array(y)

SEQ_LENGTH = 60  # Use the past 60 minutes to predict next minute
X, y = create_sequences(data, SEQ_LENGTH)

# Split data into training and testing sets
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build LSTM Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(SEQ_LENGTH, 5)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Make predictions
predictions = model.predict(X_test)

# Inverse transform predictions to original scale
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(df[['open', 'high', 'low', 'close', 'volume']])  # Re-fit scaler
predictions = scaler.inverse_transform(np.hstack((np.zeros((len(predictions), 4)), predictions.reshape(-1, 1))))[:, 4]

# Plot results
plt.figure(figsize=(10,5))
plt.plot(df.index[-len(y_test):], y_test, label='Actual Price')
plt.plot(df.index[-len(y_test):], predictions, label='Predicted Price')
plt.legend()
plt.show()
