import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from src.data_engineering import download_data, compute_technical_indicators
from src.preprocessing import normalize_features, create_sequences
from src.transformer_model import TransformerModel

# Load configuration from config.yaml or hardcode here
ticker = "^NSEBANK"
start_date = "2024-01-01"
end_date = "2025-02-28"
window_size = 10
test_size = 0.2
num_epochs = 50
learning_rate = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Download and preprocess data
data = download_data(ticker, start_date, end_date)
data = compute_technical_indicators(data)
# Uncomment to plot indicators
# from src.data_engineering import plot_indicators
# plot_indicators(data)

# Create feature set: we use Close, MA10, RSI, LogReturn
features = data[['Close', 'MA10', 'RSI', 'LogReturn']].values.astype(np.float32)
features_scaled, scalers = normalize_features(features)
X, y = create_sequences(features_scaled, window_size)
print("Shape of X:", X.shape, "Shape of y:", y.shape)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
y_train = torch.tensor(y_train, dtype=torch.float32, device=device)
X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
y_test = torch.tensor(y_test, dtype=torch.float32, device=device)

# Define model parameters
input_dim = features.shape[1]  # 4 features
model_dim = 64
num_heads = 4
num_layers = 2
output_dim = 1

model = TransformerModel(input_dim, model_dim, num_heads, num_layers, output_dim, dropout=0.1).to(device)
print(model)

criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_losses = []
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs.squeeze(), y_train)
    loss.backward()
    optimizer.step()
    
    train_losses.append(loss.item())
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}")

plt.figure(figsize=(10,4))
plt.plot(train_losses, label="Train Loss")
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Save the trained model
torch.save(model.state_dict(), "best_model.pth")
