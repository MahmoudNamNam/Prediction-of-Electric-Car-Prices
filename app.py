import numpy as np
import streamlit as st
import os
import plotly.express as px
import pandas as pd
from config import *
from predict import prepare_model_data
# Define companies and their specific themes and logos
companies = {
    "BMW": {"color": "#0066B2", "logo_path": "logos/bmw_logo.png"},
    "Honda": {"color": "#E40521", "logo_path": "logos/honda_logo.png"},
    "NIO": {"color": "#eaaaab", "logo_path": "logos/nio_logo.png"},
    "Nissan": {"color": "#C3002F", "logo_path": "logos/nissan_logo.png"},
    "Tata": {"color": "#1A5586", "logo_path": "logos/tata_logo.png"},
    "Tesla": {"color": "#CC0000", "logo_path": "logos/tesla_logo.png"},
    "Volkswagen": {"color": "#00306A", "logo_path": "logos/volkswagen_logo.png"}
}

# Sidebar for selecting a company
selected_company = st.sidebar.selectbox("Select Company", list(companies.keys()))

# Load logo and theme color for the selected company
company_theme = companies[selected_company]
logo_path = company_theme["logo_path"]
theme_color = company_theme["color"]

st.markdown(
    f"""
    <style>
    .appview-container .block-container {{
        background-color: {theme_color};
        color: white;
    }}
    </style>
    """,
    unsafe_allow_html=True
)


# Display the logo and title for the selected company
if os.path.exists(logo_path):
    st.image(logo_path, width=150)
else:
    st.write("Logo not found.")

st.title(f"{selected_company} Stock Predictions Dashboard")

# Load company-specific data
data_path = f'./data/{selected_company}/{selected_company.lower()}.csv'

if os.path.exists(data_path):
    data = pd.read_csv(data_path)
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.dropna()

    sequence_length=60
    train_size = 0.8
    model_path = f'models/{selected_company.lower()}_stock_model.pth'

    train_data ,test_data,y_test_rescaled,predictions_rescaled,scaler =prepare_model_data(data,selected_company)

    train_data_rescaled = scaler.inverse_transform(train_data.values)

    train_data_rescaled_flat = train_data_rescaled[SEQUENCE_LENGTH:].flatten()
    y_test_rescaled_flat = y_test_rescaled.flatten()
    predictions_rescaled_flat = predictions_rescaled.flatten()
    color_map = {
    'Training Data': theme_color,
    'Actual Price': '#04BF9D',
    'Predicted Price': '#F27457'
}
    results_df = pd.DataFrame({
        'Date': train_data.index[SEQUENCE_LENGTH:].tolist() + test_data.index[SEQUENCE_LENGTH:].tolist() + test_data.index[SEQUENCE_LENGTH:].tolist(),
        'Price': np.concatenate([train_data_rescaled_flat, y_test_rescaled_flat, predictions_rescaled_flat]),
        'Type': ['Training Data'] * len(train_data_rescaled_flat) + 
                ['Actual Price'] * len(y_test_rescaled_flat) + 
                ['Predicted Price'] * len(predictions_rescaled_flat)
    })

    fig = px.line(results_df, x='Date', y='Price', color='Type',
                title=f"Price Prediction for {selected_company}",
                color_discrete_map=color_map)
    fig.update_traces(line=dict(width=3))
    st.plotly_chart(fig)
    fig.update_layout(title_x=0.5)


    # Metric selection
    metric = st.selectbox("Select Metric", ["Open", "High", "Low", "Close", "Adj Close", "Volume"])

    # Plotting with Plotly Express
    fig = px.line(
        data,
        x="Date",
        y=metric,
        title=f"{selected_company} {metric} Price Over Time",
        labels={"Date": "Date", metric: metric}
    )
    fig.update_traces(line=dict(color=theme_color))
    fig.update_layout(title_x=0.5)
    st.plotly_chart(fig, use_container_width=True)

    # st.write(data[["Open", "High", "Low", "Close", "Adj Close", "Volume",'EMA_10']].describe() )
    summary = data[["Open", "High", "Low", "Close", "Adj Close", "Volume", 'EMA_10']].describe()

    # center the dataframe
    st.markdown(
        """
        <style>
        .stDataFrame {
            display: flex;
            justify-content: center;
        }
        </style>
        """, 
        unsafe_allow_html=True
    )
    st.dataframe(summary)

    # Moving Averages
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['Daily_Return'] = data['Close'].pct_change() * 100
    data['High_Low_Spread'] = (data['High'] - data['Low']) / data['Low'] * 100

    # Close Price Over Time with Moving Averages
    fig_close = px.line(
        data, 
        x="Date", 
        y=["Close", "MA20", "MA50"], 
        title=f"{selected_company} Close Price & Moving Averages",
        labels={"value": "Price", "variable": "Type"},
        color_discrete_map={"Close": theme_color, "MA20": "#eae", "MA50": "#ffa"}
    )
    fig_close.update_layout(title_x=0.5)
    st.plotly_chart(fig_close, use_container_width=True)


    # Daily Return Distribution
    fig_return = px.line(
        data, 
        x='Date',
        y="Daily_Return", 
        title="Daily Return Distribution",
        labels={"Daily_Return": "Daily Return (%)"}
    )
    fig_return.update_traces(line=dict(color='#eee'))
    fig_return.update_layout(title_x=0.5)
    st.plotly_chart(fig_return, use_container_width=True)

    fig_return_dist = px.histogram(
        data, 
        x="Daily_Return", 
        nbins=30, 
        title="Daily Return Distribution",
        labels={"Daily_Return": "Daily Return (%)"},
        color_discrete_sequence=['#eac']
    )
    fig_return_dist.update_layout(title_x=0.5)
    st.plotly_chart(fig_return_dist, use_container_width=True)

    # High-Low Spread Over Time
    fig_spread = px.line(
        data, 
        x="Date", 
        y="High_Low_Spread", 
        title=f"{selected_company} High-Low Spread Over Time",
        labels={"High_Low_Spread": "High-Low Spread (%)"}
    )
    fig_spread.update_traces(line=dict(color='#aca'))
    fig_spread.update_layout(title_x=0.5)
    st.plotly_chart(fig_spread, use_container_width=True)
else:
    st.write("Data file not found.")
