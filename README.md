# Stock Price Prediction Model

This project leverages LSTM neural networks for stock price prediction, tailored to multiple manufacturers. The system is designed to preprocess, train, evaluate, and visualize stock trends with high accuracy, and includes a user-friendly dashboard for tracking predictions and insights over time.

## Features

- **Data Processing and Preparation:** Preprocesses stock data with scaling and sequence generation for LSTM.
- **LSTM Model Training:** Customizable LSTM model with flexible hidden layers and dropout for improved generalization.
- **Performance Evaluation:** Computes MSE and R-squared metrics and saves results in JSON format for further analysis.
- **Visualization and Dashboard:** Streamlit-based dashboard with interactive visualizations using Plotly for easy exploration of stock trends, moving averages, daily returns, and high-low spreads.

## Files Overview

- `train.py`: Prepares data, trains the model, evaluates performance, and saves results.
- `model.py`: Defines `StockPriceLSTM_v1` and `StockPriceLSTM_v2` for modeling time-series data.
- `app.py`: Streamlit app for visualizing stock trends, metrics, and prediction insights.
- `Data_Preparations.py`, `evaluation.py`, `config.py`: Modules for preprocessing, evaluation, and configuration.

## Getting Started

### Prerequisites

- Python 3.7+
- Required libraries: `torch`, `numpy`, `pandas`, `streamlit`, `plotly`

### Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/stock-price-prediction
pip install -r requirements.txt
```

### Usage

#### Model Training

To train the model on multiple companies, run:

```bash
python train.py
```

#### Running the Dashboard

Launch the Streamlit app to visualize predictions:

```bash
streamlit run app.py
```
![screenshot-localhost-8501-1731687323445](https://github.com/user-attachments/assets/19b70eb3-96b9-43f9-9cc0-427597555edf)
![screenshot-localhost-8501-1731687276383](https://github.com/user-attachments/assets/16331b16-6223-4c8f-a22d-c6301fd69925)


## Results and Visualizations

Each trained model is saved with its metrics and visualizations, providing a clear view of performance and prediction accuracy.
