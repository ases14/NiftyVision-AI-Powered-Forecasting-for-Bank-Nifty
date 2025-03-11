import numpy as np
from sklearn.preprocessing import MinMaxScaler

def normalize_features(features):
    scalers = {}
    features_scaled = features.copy()
    for i in range(features.shape[1]):
        scaler = MinMaxScaler(feature_range=(0, 1))
        features_scaled[:, i:i+1] = scaler.fit_transform(features[:, i:i+1])
        scalers[i] = scaler
    return features_scaled, scalers

def create_sequences(data, window_size):
    sequences = []
    labels = []
    for i in range(len(data) - window_size):
        sequences.append(data[i:i+window_size])
        # Predict the "Close" price (first feature)
        labels.append(data[i+window_size, 0])
    return np.array(sequences), np.array(labels)
