# streamlit_app/app.py

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
import yfinance as yf
import yaml

from src.data_engineering import download_data, compute_technical_indicators
from src.preprocessing import normalize_features, create_sequences
from src.transformer_model import TransformerModel
from sklearn.model_selection import train_test_split

# ----------------------
# Configuration Section
# ----------------------
# You can either load these from a config.yaml file or define them here.
ticker = "^NSEBANK"
start_date = "2020-01-01"
end_date = "2023-01-01"
window_size = 10
test_size = 0.2
num_epochs = 50
learning_rate = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model parameters
input_dim = 4  # Using features: Close, MA10, RSI, LogReturn
model_dim = 64
num_heads = 4
num_layers = 2
output_dim = 1

st.title("NiftyVision: AI-Powered Forecasting for Bank Nifty")
st.write("This dashboard displays predictions of the Bank Nifty close price using an improved Transformer model.")

# ---------------------------
# Data Engineering & Loading
# ---------------------------
st.header("Data and Feature Engineering")
st.write("Downloading data from Yahoo Finance and computing technical indicators...")

try:
    # Download and process the data
    data = download_data(ticker, start_date, end_date)
    data = compute_technical_indicators(data)
except ValueError as e:
    st.error(str(e))
    st.stop()

# Display a sample of the data
st.subheader("Enhanced Data Sample")
st.dataframe(data.head())

# Plot technical indicators
st.subheader("Technical Indicators")
fig1, ax1 = plt.subplots(2, 1, figsize=(12, 8))
ax1[0].plot(data.index, data['Close'], label='Close')
ax1[0].plot(data.index, data['MA10'], label='MA10')
ax1[0].plot(data.index, data['MA20'], label='MA20')
ax1[0].plot(data.index, data['MA50'], label='MA50')
ax1[0].set_title("Close Price and Moving Averages")
ax1[0].legend()

ax1[1].plot(data.index, data['RSI'], label='RSI', color='purple')
ax1[1].plot(data.index, data['Volatility'], label='Volatility', color='orange')
ax1[1].set_title("RSI and Volatility")
ax1[1].legend()
st.pyplot(fig1)

# ---------------------------
# Preprocessing
# ---------------------------
st.header("Data Preprocessing")
features = data[['Close', 'MA10', 'RSI', 'LogReturn']].values.astype(np.float32)
features_scaled, scalers = normalize_features(features)
X, y = create_sequences(features_scaled, window_size)
st.write("Shape of input sequences:", X.shape)

# Split data into training and test sets (no shuffling for time series)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
y_train = torch.tensor(y_train, dtype=torch.float32, device=device)
X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
y_test = torch.tensor(y_test, dtype=torch.float32, device=device)

# ---------------------------
# Load Trained Model
# ---------------------------
st.header("Model Predictions")
st.write("Loading the trained Transformer model...")

model = TransformerModel(input_dim, model_dim, num_heads, num_layers, output_dim, dropout=0.1).to(device)
# Assuming the model has been saved as "best_model.pth" in the project root
model_path = "best_model.pth"
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

model.eval()

# ---------------------------
# Generate Predictions
# ---------------------------
with torch.no_grad():
    train_preds = model(X_train).cpu().numpy()
    test_preds = model(X_test).cpu().numpy()

close_scaler = scalers[0]
train_preds_actual = close_scaler.inverse_transform(train_preds)
y_train_actual = close_scaler.inverse_transform(y_train.cpu().numpy().reshape(-1, 1))
test_preds_actual = close_scaler.inverse_transform(test_preds)
y_test_actual = close_scaler.inverse_transform(y_test.cpu().numpy().reshape(-1, 1))

# ---------------------------
# Plot Predictions on Single Graph
# ---------------------------
st.subheader("Actual vs. Predicted Prices (Train and Test)")
train_index = np.arange(len(y_train_actual))
test_index = np.arange(len(y_test_actual)) + len(y_train_actual)

fig2, ax2 = plt.subplots(figsize=(14,6))
ax2.plot(train_index, y_train_actual, label="Train Actual", color="blue")
ax2.plot(train_index, train_preds_actual, label="Train Predicted", color="green", linestyle="--")
ax2.plot(test_index, y_test_actual, label="Test Actual", color="red")
ax2.plot(test_index, test_preds_actual, label="Test Predicted", color="orange", linestyle="--")
ax2.axvline(x=len(y_train_actual)-1, color='black', linestyle='--', label="Train/Test Split")
ax2.set_title("Actual vs. Predicted Prices")
ax2.set_xlabel("Time Step")
ax2.set_ylabel("Price")
ax2.legend()
st.pyplot(fig2)

# ---------------------------
# Display Error Metrics
# ---------------------------
errors = y_test_actual - test_preds_actual
test_mae = np.mean(np.abs(errors))
test_rmse = np.sqrt(np.mean(errors**2))
st.write(f"Test MAE: {test_mae:.2f}")
st.write(f"Test RMSE: {test_rmse:.2f}")

# Optionally, plot error distribution
fig3, ax3 = plt.subplots(figsize=(8,4))
ax3.hist(errors, bins=30, color='skyblue', edgecolor='black')
ax3.set_title("Error Distribution (Test Set)")
ax3.set_xlabel("Error")
ax3.set_ylabel("Frequency")
st.pyplot(fig3)
