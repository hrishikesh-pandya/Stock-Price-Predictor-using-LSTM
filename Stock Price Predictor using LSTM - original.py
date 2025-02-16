import yfinance as yf
import pandas as pd

#Fetching historical stock data for Tesla
stock_data=yf.download('F',start='2016-10-01',end='2025-02-05')
# Display the first few rows of the dataset, shows headers like open,high,low etc.
stock_data.head()

import numpy as np
from sklearn.preprocessing import MinMaxScaler

#Use only the 'Close' column for price prediction
close_prices=stock_data['Close'].values

#Normalize the dataset using MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(close_prices.reshape(-1,1))

#Split the data into training (80%) and testing(20%) sets
train_size=int(len(scaled_data)*0.8)
train_data, test_data=scaled_data[:train_size], scaled_data[train_size:]

def create_sequences(data, seq_length):
    x,y=[],[]
    for i in range(seq_length,len(data)):
        x.append(data[i-seq_length:i,0])
        y.append(data[i,0])
    return np.array(x),np.array(y)

#Create sequences from the training and test data
seq_length=60 #meaning use the last 60 days to predict the next day's price
x_train,y_train = create_sequences(train_data, seq_length)
x_test, y_test = create_sequences(test_data, seq_length)

#Reshape the input data to be compatible with LSTM
x_train=np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test=np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM, Dropout

#Initialize the model
model=Sequential()

# Add LSTM layers

model.add(LSTM(units=100, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(Dropout(0.2))

model.add(LSTM(units=100,return_sequences=False))
model.add(Dropout(0.2))

# Add output layer
model.add(Dense(units=1))

#Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

#Train the model
model.fit(x_train,y_train,epochs=20,batch_size=40)

#Predict stock prices on the test data
predictions=model.predict(x_test)

#Inverse transform the predictions back to original price scale
predictions=scaler.inverse_transform(predictions)

#Inverse transform the actual test data
y_test_scaled=scaler.inverse_transform(y_test.reshape(-1,1))

import plotly.graph_objs as go

#Create a plotly figure
fig=go.Figure()

# Add trace for actual prices
fig.add_trace(go.Scatter(x=stock_data.index[-len(y_test):], y=y_test_scaled.flatten(), mode='lines', name='Actual Price'))

# Add trace for predicted prices
fig.add_trace(go.Scatter(x=stock_data.index[-len(y_test):], y=predictions.flatten(), mode='lines', name='Predicted Price'))

# Add titles and labels
fig.update_layout(title='Ford prediction', xaxis_title='Date', yaxis_title='Stock Price (USD)', yaxis=dict(range=[0, max(max(y_test_scaled.flatten()), max(predictions.flatten())) * 1.1]))

# Show the figure
fig.show()

from sklearn.metrics import mean_squared_error

#Calculate MSE and RMSE
mse=mean_squared_error(y_test_scaled,predictions)
rmse=np.sqrt(mse)

print(f'Mean squared error:{mse}')
print(f'Root Mean squared error:{rmse}')
