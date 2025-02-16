import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import plotly.graph_objs as go
from sklearn.metrics import mean_squared_error

# Fetch historical stock data for Tesla
stock_data = yf.download('TSLA', start='2016-10-01', end='2025-02-06')

# Use only the 'Close' column for price prediction
close_prices = stock_data['Close'].values

# Normalize the dataset using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices.reshape(-1, 1))

# Split the data into training (80%) and testing (20%) sets
train_size = int(len(scaled_data) * 0.8)
train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

def create_sequences(data, seq_length):
    x, y = [], []
    for i in range(seq_length, len(data)):
        x.append(data[i-seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(x), np.array(y)

seq_length = 60  # Use last 60 days to predict the next day
x_train, y_train = create_sequences(train_data, seq_length)
x_test, y_test = create_sequences(test_data, seq_length)

# Reshape the input data to be compatible with LSTM
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Initialize the LSTM model
model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_shape=(seq_length, 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=100, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=10)

# Predict stock prices on the test data
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Inverse transform the actual test data
y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# Predict future stock prices (next 3 days)
future_days = 3
future_predictions = []
future_input = scaled_data[-seq_length:].copy()

for _ in range(future_days):
    future_input_reshaped = np.reshape(future_input, (1, seq_length, 1))
    next_price = model.predict(future_input_reshaped)
    next_price_original = scaler.inverse_transform(next_price)[0, 0]
    future_predictions.append(next_price_original)
    future_input = np.append(future_input[1:], next_price, axis=0)

# Create a plotly figure
fig = go.Figure()
fig.add_trace(go.Scatter(x=stock_data.index[-len(y_test):], y=y_test_scaled.flatten(), mode='lines', name='Actual Price'))
fig.add_trace(go.Scatter(x=stock_data.index[-len(y_test):], y=predictions.flatten(), mode='lines', name='Predicted Price'))
fig.add_trace(go.Scatter(x=pd.date_range(stock_data.index[-1], periods=future_days + 1)[1:], y=future_predictions, mode='lines', name='Future Predictions', line=dict(dash='dot')))

# Add titles and labels
fig.update_layout(title='Tesla Stock Price Prediction', xaxis_title='Date', yaxis_title='Stock Price (USD)')
fig.show()

# Calculate MSE and RMSE
mse = mean_squared_error(y_test_scaled, predictions)
rmse = np.sqrt(mse)

print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
print(f'Predicted prices for the next {future_days} days: {future_predictions}')
