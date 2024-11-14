import torch
import matplotlib.pyplot as plt
import pandas as pd
from model import StockPriceLSTM
from utils import preprocess_data
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import os
import json

def load_model(model_path, model_class=StockPriceLSTM, hidden_layer_size=64):
    """
    Loads the model from the saved path.

    Parameters:
    - model_path (str): Path to the saved model file.
    - model_class (nn.Module): The model class to instantiate.
    - hidden_layer_size (int): The size of the hidden layer in the model.

    Returns:
    - model (nn.Module): The loaded model.
    """
    model = model_class(input_size=1, hidden_layer_size=hidden_layer_size, output_size=1)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set model to evaluation mode
    return model

def generate_plots_for_model(company, model, scaler, sequence_length=30):
    """
    Generates and saves plots comparing actual vs predicted prices for the given model.

    Parameters:
    - company (str): The name of the company for which to generate the plot.
    - model (nn.Module): The trained model to make predictions.
    - scaler (MinMaxScaler): The scaler used to inverse transform the data.
    - sequence_length (int): The length of the input sequence for the LSTM.
    """
    path = f'./Data/{company}/{company.lower()}.csv'
    data = pd.read_csv(path)
    data['Date'] = pd.to_datetime(data['Date'])

    # Preprocess the data
    train_data, test_data, scaler = preprocess_data(data)

    # Create sequences for training and testing
    x_train, y_train = create_sequences(train_data[['EMA_10']].values, sequence_length)
    x_test, y_test = create_sequences(test_data[['EMA_10']].values, sequence_length)

    # Convert to PyTorch tensors
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32).unsqueeze(-1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # Generate predictions
    with torch.no_grad():
        predictions = model(x_test_tensor).detach().numpy()

    # Inverse transform the predictions and actual values
    predictions_rescaled = scaler.inverse_transform(predictions)
    y_test_rescaled = scaler.inverse_transform(y_test_tensor.unsqueeze(1).numpy())

    # Calculate metrics
    mse = float(mean_squared_error(y_test_rescaled, predictions_rescaled))
    r2 = float(r2_score(y_test_rescaled, predictions_rescaled))

    print(f"Mean Squared Error (MSE) for {company}: {mse:.4f}")
    print(f"R-squared (R²) for {company}: {r2:.4f}")

    # Plot the results
    plt.figure(figsize=(14, 7))
    train_data_rescaled = scaler.inverse_transform(train_data[['EMA_10']].values)
    plt.plot(train_data.index[sequence_length:], train_data_rescaled[sequence_length:], label='Training Data')
    plt.plot(test_data.index[sequence_length:], y_test_rescaled, label='Actual Price')
    plt.plot(test_data.index[sequence_length:], predictions_rescaled, label='Predicted Price')
    plt.title(f"Price Prediction for {company}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()

    # Save the plot
    os.makedirs("figures", exist_ok=True)
    plt.savefig(f'figures/{company.lower()}_price_prediction.png')
    plt.show()

    # Save metrics to JSON
    os.makedirs("reports", exist_ok=True)
    metrics = {"company": company, "mse": mse, "r2": r2}
    with open(f'reports/{company.lower()}_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics for {company} saved to reports/{company.lower()}_metrics.json")

# Example usage:
if __name__ == "__main__":
    companies = ['BMW', 'Honda', 'NIO', 'Nissan', 'Tata', 'Tesla', 'Volkswagen']

    for company in companies:
        # Load the model (adjust path and model type as needed)
        model_path = f"models/{company.lower()}_stock_model.pth"  # path to your saved model
        model = load_model(model_path)
        
        # Generate plots
        generate_plots_for_model(company, model, scaler=None)  # Pass the scaler from preprocessing
