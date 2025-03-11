import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from src.data_engineering import download_data, compute_technical_indicators
from src.preprocessing import normalize_features, create_sequences
from src.transformer_model import TransformerModel

# Use same configuration as in train.py
ticker = "^NSEBANK"
start_date = "2020-01-01"
end_date = "2023-01-01"
window_size = 10
test_size = 0.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Download and preprocess data
data = download_data(ticker, start_date, end_date)
data = compute_technical_indicators(data)
features = data[['Close', 'MA10', 'RSI', 'LogReturn']].values.astype(np.float32)
features_scaled, scalers = normalize_features(features)
X, y = create_sequences(features_scaled, window_size)
print("Shape of X:", X.shape, "Shape of y:", y.shape)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
y_train = torch.tensor(y_train, dtype=torch.float32, device=device)
X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
y_test = torch.tensor(y_test, dtype=torch.float32, device=device)

# Define model parameters (same as training)
input_dim = features.shape[1]
model_dim = 64
num_heads = 4
num_layers = 2
output_dim = 1

# Initialize model and load trained weights
model = TransformerModel(input_dim, model_dim, num_heads, num_layers, output_dim, dropout=0.1).to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

with torch.no_grad():
    train_preds = model(X_train).cpu().numpy()
    test_preds = model(X_test).cpu().numpy()

# Inverse transform predictions and actual values for the "Close" feature using the scaler for feature 0
close_scaler = scalers[0]
train_preds_actual = close_scaler.inverse_transform(train_preds)
y_train_actual = close_scaler.inverse_transform(y_train.cpu().numpy().reshape(-1, 1))
test_preds_actual = close_scaler.inverse_transform(test_preds)
y_test_actual = close_scaler.inverse_transform(y_test.cpu().numpy().reshape(-1, 1))

# Create index arrays for plotting in one graph
train_index = np.arange(len(y_train_actual))
test_index = np.arange(len(y_test_actual)) + len(y_train_actual)

plt.figure(figsize=(14,6))
plt.plot(train_index, y_train_actual, label="Train Actual", color="blue")
plt.plot(train_index, train_preds_actual, label="Train Predicted", color="green", linestyle="--")
plt.plot(test_index, y_test_actual, label="Test Actual", color="red")
plt.plot(test_index, test_preds_actual, label="Test Predicted", color="orange", linestyle="--")
plt.axvline(x=len(y_train_actual)-1, color='black', linestyle='--', label="Train/Test Split")
plt.title("Actual vs. Predicted Prices (Train and Test)")
plt.xlabel("Time Step")
plt.ylabel("Price")
plt.legend()
plt.show()

# Compute error metrics for the test set
errors = y_test_actual - test_preds_actual
mae = np.mean(np.abs(errors))
rmse = np.sqrt(np.mean(errors**2))
print(f"Test MAE: {mae:.2f}")
print(f"Test RMSE: {rmse:.2f}")
