import os
import json
import torch
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from model import StockPriceLSTM_v1, StockPriceLSTM_v2
from Data_Preparations import preprocess_data
from evaluation import evaluate_model, plot_predictions
from config import *


def prepare_model_data(data,company):
    model=StockPriceLSTM_v2(input_size=1, hidden_layer_size=HIDDEN_LAYER_SIZE, output_size=1)
    train_data, test_data, scaler = preprocess_data(data, TRAIN_SIZE, 50)
    if company in {'Tesla','NIO'}:
        model = StockPriceLSTM_v1()
        train_data, test_data, scaler = preprocess_data(data, TRAIN_SIZE, 10)
    # After creating sequences
    x_train, y_train = model.create_sequences(train_data.values, SEQUENCE_LENGTH)
    x_test, y_test = model.create_sequences(test_data.values, SEQUENCE_LENGTH)
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


    model_path = f'models/{company.lower()}_stock_model.pth'
    model.load_state_dict(torch.load(model_path,weights_only=True))
    model.eval()
    with torch.no_grad():
        predictions = model(x_test).detach().numpy()

    predictions_rescaled = scaler.inverse_transform(predictions)
    y_test_rescaled = scaler.inverse_transform(y_test.unsqueeze(1).numpy())
    return train_data, test_data, y_test_rescaled, predictions_rescaled, scaler
