import os
import json
import torch
import pandas as pd
import numpy as np
from model import StockPriceLSTM
from Data_Preparations import preprocess_data
from evaluation import evaluate_model, plot_predictions
from config import *
def append_to_json(file_path, new_data):
    # Check if the file exists and is not empty
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        with open(file_path, 'r') as file:
            try:
                existing_data = json.load(file)  # Try to load existing data
            except json.JSONDecodeError:  # If file is empty or corrupted
                existing_data = []
    else:
        existing_data = []  # Initialize an empty list if the file is empty or doesn't exist

    # Append the new data
    existing_data.append(new_data)

    # Save the updated data back to the file
    with open(file_path, 'w') as file:
        json.dump(existing_data, file, indent=4)
    print(f"Results saved to {file_path}")

def train_model_for_manufacturer(company):
    path = f'./data/{company}/{company.lower()}.csv'
    data = pd.read_csv(path)

    train_data, test_data, scaler = preprocess_data(data, TRAIN_SIZE, EMA_WINDOW)


    # After creating sequences
    x_train, y_train = StockPriceLSTM.create_sequences(train_data.values, SEQUENCE_LENGTH)
    x_test, y_test = StockPriceLSTM.create_sequences(test_data.values, SEQUENCE_LENGTH)

    # Check and enforce numeric type for x_train and y_train
    x_train = np.array(x_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    x_test = np.array(x_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)

    # Convert to tensors
    x_train = torch.tensor(x_train, dtype=torch.float32).unsqueeze(-1)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32).unsqueeze(-1)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Initialize model
    model = StockPriceLSTM(input_size=1, hidden_layer_size=HIDDEN_LAYER_SIZE, output_size=1)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        output = model(x_train)
        loss = criterion(output, y_train.unsqueeze(1))
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {loss.item():.4f}")

    # Evaluate and save results
    mse, r2, predictions_rescaled, y_test_rescaled = evaluate_model(model, x_test, y_test, scaler)

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = f'{MODEL_DIR}/{company.lower()}_stock_model.pth'
    torch.save(model.state_dict(), model_path)

    # Save metrics
    os.makedirs(REPORT_DIR, exist_ok=True)
    metrics_path = f'{REPORT_DIR}/{company.lower()}_metrics.json'
    report_data = {
        "company": company,
        "config": {
            "SEQUENCE_LENGTH": SEQUENCE_LENGTH,
            "EPOCHS": EPOCHS,
            "HIDDEN_LAYER_SIZE": HIDDEN_LAYER_SIZE,
            "LEARNING_RATE": LEARNING_RATE,
            "TRAIN_SIZE": TRAIN_SIZE,
            "EMA_WINDOW": EMA_WINDOW,
        },
        "metrics": {
            "mse": mse,
            "r2": r2
        }
    }
    append_to_json(metrics_path, report_data)

    # Plot and save figure
    fig_path = plot_predictions(train_data, test_data, predictions_rescaled, y_test_rescaled, company, FIGURE_DIR,scaler)
    print(f"Training complete for {company}. Model saved to {model_path}, metrics to {metrics_path}, figure to {fig_path}")

if __name__ == "__main__":
    companies = ['BMW', 'Honda', 'NIO', 'Nissan', 'Rolls Royces', 'Tata', 'Tesla', 'Volkswagen']
    for company in companies:
        train_model_for_manufacturer(company)
