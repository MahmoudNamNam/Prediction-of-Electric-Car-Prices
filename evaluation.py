from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
from config import *

def evaluate_model(model, x_test, y_test, scaler):
    model.eval()
    with torch.no_grad():
        predictions = model(x_test).detach().numpy()

    predictions_rescaled = scaler.inverse_transform(predictions)
    y_test_rescaled = scaler.inverse_transform(y_test.unsqueeze(1).numpy())

    mse = float(mean_squared_error(y_test_rescaled, predictions_rescaled))
    r2 = float(r2_score(y_test_rescaled, predictions_rescaled))
    return mse, r2, predictions_rescaled, y_test_rescaled

def plot_predictions(train_data, test_data, predictions_rescaled, y_test_rescaled, company, figure_dir,scaler):
    plt.figure(figsize=(14, 7))
    train_data_rescaled = scaler.inverse_transform(train_data.values)
    plt.plot(train_data.index[SEQUENCE_LENGTH:], train_data_rescaled[SEQUENCE_LENGTH:], label='Training Data')
    plt.plot(test_data.index[SEQUENCE_LENGTH:], y_test_rescaled, label='Actual Price')
    plt.plot(test_data.index[SEQUENCE_LENGTH:], predictions_rescaled, label='Predicted Price')
    plt.title(f"Price Prediction for {company}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    os.makedirs(figure_dir, exist_ok=True)
    fig_path = f'{figure_dir}/{company.lower()}_price_prediction.png'
    plt.savefig(fig_path)
    return fig_path
