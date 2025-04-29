import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers, models
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Fetch stock data
def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]

# Preprocess data for time-series prediction
def preprocess_data(data, window_size=10):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i-window_size:i])
        y.append(scaled_data[i, 3])  # Predict 'Close' price
    
    X = np.array(X)
    y = np.array(y)
    
    # Split into train and test sets
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return X_train, X_test, y_train, y_test, scaler

# Build Feed Forward Neural Network
def build_ffnn(input_shape):
    model = models.Sequential([
        layers.Flatten(input_shape=input_shape),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)  # Output: predicted closing price
    ])
    
    model.compile(optimizer='adam', loss='mse')
    return model

# Plot predictions
def plot_predictions(y_test, y_pred, title):
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label='Actual Price')
    plt.plot(y_pred, label='Predicted Price')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Normalized Price')
    plt.legend()
    plt.savefig('stock_prediction.png')
    plt.close()

# Main execution
if __name__ == "__main__":
    # Parameters
    ticker = 'AAPL'  # Example: Apple stock
    start_date = '2020-01-01'
    end_date = '2023-12-31'
    window_size = 10  # Number of past days to use for prediction
    
    # Fetch and preprocess data
    stock_data = fetch_stock_data(ticker, start_date, end_date)
    X_train, X_test, y_train, y_test, scaler = preprocess_data(stock_data, window_size)
    
    # Build and train model
    model = build_ffnn(input_shape=(window_size, stock_data.shape[1]))
    model.summary()
    
    history = model.fit(X_train, y_train,
                        epochs=50,
                        batch_size=32,
                        validation_split=0.2,
                        verbose=1)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"\nRoot Mean Squared Error: {rmse:.4f}")
    
    # Plot results
    plot_predictions(y_test, y_pred, f"{ticker} Stock Price Prediction")
    print("Prediction plot saved as 'stock_prediction.png'")
    
    # Example: Predict next day's closing price
    last_sequence = scaler.transform(stock_data[-window_size:]).reshape(1, window_size, stock_data.shape[1])
    next_pred = model.predict(last_sequence)[0][0]
    
    # Inverse transform to get actual price
    next_pred_array = np.zeros((1, stock_data.shape[1]))
    next_pred_array[0, 3] = next_pred
    next_pred_price = scaler.inverse_transform(next_pred_array)[0, 3]
    print(f"Predicted Next Day Closing Price: ${next_pred_price:.2f}")