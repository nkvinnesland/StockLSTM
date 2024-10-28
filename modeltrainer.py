import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import joblib

# Set pandas option to opt into future behavior
pd.set_option('future.no_silent_downcasting', True)

# Load the combined CSV file
data = pd.read_csv('technology_stocks_daily_data_sorted.csv')

# Create a dictionary to hold data for each ticker
ticker_data = {ticker: data[data['ticker'] == ticker] for ticker in data['ticker'].unique()}

# Define the date range to cover the entire period of interest
start_date = data['date'].min()
end_date = data['date'].max()
date_range = pd.date_range(start=start_date, end=end_date)

# Reindex each ticker's DataFrame to include all dates in the range
for ticker in ticker_data:
    ticker_data[ticker] = ticker_data[ticker].set_index('date').reindex(date_range).reset_index()
    ticker_data[ticker] = ticker_data[ticker].fillna(0).infer_objects(copy=False)  # Replace NaN values with 0 and infer objects

# Combine the data back into a single DataFrame
data = pd.concat(ticker_data.values())

# Create a scaler
scaler = MinMaxScaler()

# Normalize each ticker's data
for ticker in ticker_data:
    features = ['open', 'high', 'low', 'close', 'adjusted_close', 'volume']
    ticker_data[ticker][features] = scaler.fit_transform(ticker_data[ticker][features])

# Combine the normalized data back into a single DataFrame
normalized_data = pd.concat(ticker_data.values())

def create_sequences(data, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        seq = data.iloc[i:i + seq_length][['open', 'high', 'low', 'close', 'adjusted_close', 'volume']].values
        label = data.iloc[i + seq_length]['close']  # or another column as your target
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)

# Create sequences for each ticker
seq_length = 60  # Example sequence length
all_sequences = []
all_labels = []
for ticker in ticker_data:
    sequences, labels = create_sequences(ticker_data[ticker], seq_length)
    all_sequences.append(sequences)
    all_labels.append(labels)

# Combine all sequences and labels
X = np.concatenate(all_sequences)
y = np.concatenate(all_labels)

# Define the LSTM model with Dropout layers
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))  # Adding dropout layer with dropout rate of 0.2
model.add(LSTM(50))
model.add(Dropout(0.2))  # Adding another dropout layer
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model with early stopping
history = model.fit(X, y, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Save the model
model.save('tech_model.keras')
print("Model saved as tech_model.keras")

# Save the scaler
scaler_filename = 'scaler.save'
joblib.dump(scaler, scaler_filename)
print("Scaler saved.")

