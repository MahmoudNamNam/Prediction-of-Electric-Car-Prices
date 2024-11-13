# utils.py

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def calculate_ema(data, window):
    """
    Calculate Exponential Moving Average (EMA) for a given time series.
    
    Parameters:
        data (pd.Series): The time series data (e.g., stock prices).
        window (int): The period over which to apply the EMA.
    
    Returns:
        pd.Series: The EMA values.
    """
    return data.ewm(span=window, adjust=False).mean()

def preprocess_data(data, train_size=0.8, window=10):
    """
    Preprocess data by calculating EMA and scaling.
    """
    data['EMA_10'] = calculate_ema(data['Close'], window)
    data = data[['Date', 'EMA_10']]
    data.set_index('Date', inplace=True)

    data.dropna(inplace=True)
    
    # Split data into train and test sets
    train_len = int(len(data) * train_size)
    train_data = data[:train_len]
    test_data = data[train_len:]
    
    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_data['EMA_10'] = scaler.fit_transform(train_data[['EMA_10']])
    test_data['EMA_10'] = scaler.transform(test_data[['EMA_10']])
    
    return train_data, test_data, scaler
