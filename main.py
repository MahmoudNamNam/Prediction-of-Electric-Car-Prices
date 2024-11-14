# main.py
from train import train_model_for_manufacturer
from predict import predict
from config import FIGURE_DIR
import pandas as pd
if __name__ == "__main__":
    companies = ['BMW', 'Honda', 'NIO', 'Nissan', 'Tata', 'Tesla', 'Volkswagen']
    
    # Train models for each company
    for company in companies:
        train_model_for_manufacturer(company)

    # Example Prediction
    company = 'Tesla'
    data = pd.read_csv(f'./data/{company}/{company.lower()}.csv')
    predictions = predict(company, data, scaler=None)  # Provide scaler if needed
    print(f"Predictions for {company}: {predictions}")
