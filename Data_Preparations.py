import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def calculate_ema(data, window):
    return data.ewm(span=window, adjust=False).mean()

def preprocess_data(data, train_size=0.8, window=10):
    data['EMA_10'] = calculate_ema(data['Close'], window)
    data = data[['Date', 'EMA_10']].dropna()
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    data = data.sort_index()

    train_len = int(len(data) * train_size)
    train_data = data[:train_len].copy()
    test_data = data[train_len:].copy()

    scaler = MinMaxScaler((0, 1))
    train_data['EMA_10'] = scaler.fit_transform(train_data[['EMA_10']])
    test_data['EMA_10'] = scaler.transform(test_data[['EMA_10']])

    return train_data, test_data, scaler
