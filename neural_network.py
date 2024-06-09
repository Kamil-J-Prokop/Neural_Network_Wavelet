import random
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Load data from folder
def load_data_from_folder(folder_path):
    signals = []
    labels = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            data = pd.read_csv(file_path)
            signal = data['Signal'].values
            signal_tensor = torch.tensor(signal, dtype=torch.float32)
            signal_tensor = (signal_tensor - torch.mean(signal_tensor)) / torch.std(signal_tensor)  # Normalize the signal
            signals.append(signal_tensor)
            labels.append(extract_params_from_filename(file_name))
    return signals, labels

def extract_params_from_filename(file_name):
    parts = file_name.replace('.csv', '').split('_')
    wavelet_function = parts[1]
    level_of_decomposition = int(parts[2])
    threshold_value = float(parts[3])
    return [wavelet_function, level_of_decomposition, threshold_value]

# Define paths
data_folder_path = 'data'  # Assuming all data is initially in one folder
file_names = os.listdir(data_folder_path)
random.shuffle(file_names)  # Shuffle to ensure randomness

wavelet_function_map = {
    'bior1.1': 0,
    'bior1.3': 1,
    'bior1.5': 2,
    'bior2.2': 3,
    'bior2.4': 4,
    'bior2.6': 5,
    'bior2.8': 6,
    'bior3.1': 7,
    'bior3.3': 8,
    'bior3.5': 9,
    'bior3.7': 10,
    'bior3.9': 11,
    'bior4.4': 12,
    'bior5.5': 13,
    'bior6.8': 14,
    'coif1': 15,
    'coif2': 16,
    'coif3': 17,
    'coif4': 18,
    'coif5': 19,
    'coif6': 20,
    'coif7': 21,
    'coif8': 22,
    'coif9': 23,
    'coif10': 24,
    'coif11': 25,
    'coif12': 26,
    'coif13': 27,
    'coif14': 28,
    'coif15': 29,
    'coif16': 30,
    'coif17': 31,
    'db1': 32,
    'db2': 33,
    'db3': 34,
    'db4': 35,
    'db5': 36,
    'db6': 37,
    'db7': 38,
    'db8': 39,
    'db9': 40,
    'db10': 41,
    'db11': 42,
    'db12': 43,
    'db13': 44,
    'db14': 45,
    'db15': 46,
    'db16': 47,
    'db17': 48,
    'db18': 49,
    'db19': 50,
    'db20': 51,
    'db21': 52,
    'db22': 53,
    'db23': 54,
    'db24': 55,
    'db25': 56,
    'db26': 57,
    'db27': 58,
    'db28': 59,
    'db29': 60,
    'db30': 61,
    'db31': 62,
    'db32': 63,
    'db33': 64,
    'db34': 65,
    'db35': 66,
    'db36': 67,
    'db37': 68,
    'db38': 69,
    'dmey': 70,
    'haar': 71,
    'rbio1.1': 72,
    'rbio1.3': 73,
    'rbio1.5': 74,
    'rbio2.2': 75,
    'rbio2.4': 76,
    'rbio2.6': 77,
    'rbio2.8': 78,
    'rbio3.1': 79,
    'rbio3.3': 80,
    'rbio3.5': 81,
    'rbio3.7': 82,
    'rbio3.9': 83,
    'rbio4.4': 84,
    'rbio5.5': 85,
    'rbio6.8': 86,
    'sym2': 87,
    'sym3': 88,
    'sym4': 89,
    'sym5': 90,
    'sym6': 91,
    'sym7': 92,
    'sym8': 93,
    'sym9': 94,
    'sym10': 95,
    'sym11': 96,
    'sym12': 97,
    'sym13': 98,
    'sym14': 99,
    'sym15': 100,
    'sym16': 101,
    'sym17': 102,
    'sym18': 103,
    'sym19': 104,
    'sym20': 105
}

# Split data
split_ratio = 0.8
train_files = file_names[:int(len(file_names) * split_ratio)]
val_files = file_names[int(len(file_names) * split_ratio):]

# Function to load data from a list of files
def load_data_from_files(folder_path, file_list):
    signals = []
    labels = []
    for file_name in file_list:
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            data = pd.read_csv(file_path)
            signal = data['Signal'].values
            signal_tensor = torch.tensor(signal, dtype=torch.float32)
            signal_tensor = (signal_tensor - torch.mean(signal_tensor)) / torch.std(signal_tensor)  # Normalize the signal
            signals.append(signal_tensor)
            labels.append(extract_params_from_filename(file_name))
    return signals, labels

# Load training data
train_signals, train_labels = load_data_from_files(data_folder_path, train_files)

# Load validation data
val_signals, val_labels = load_data_from_files(data_folder_path, val_files)

# Convert wavelet function names to integers and separate other labels
def preprocess_labels(labels):
    wavelet_indices = [wavelet_function_map[l[0]] for l in labels]
    other_labels = [[l[1], l[2]] for l in labels]
    wavelet_indices = np.array(wavelet_indices).reshape(-1, 1)
    other_labels = np.array(other_labels)
    return np.hstack((wavelet_indices, other_labels))

# Preprocess training and validation labels
processed_train_labels = preprocess_labels(train_labels)
processed_val_labels = preprocess_labels(val_labels)

# Normalize labels
def normalize_labels(labels):
    labels_mean = np.mean(labels, axis=0)
    labels_std = np.std(labels, axis=0)
    normalized_labels = (labels - labels_mean) / labels_std
    return normalized_labels, labels_mean, labels_std

# Normalize training and validation labels
normalized_train_labels, train_labels_mean, train_labels_std = normalize_labels(processed_train_labels)
normalized_val_labels = (processed_val_labels - train_labels_mean) / train_labels_std

# Convert lists to tensors for training data
train_signals_tensor = torch.stack(train_signals)
train_labels_tensor = torch.tensor(normalized_train_labels, dtype=torch.float32)

# Convert lists to tensors for validation data
val_signals_tensor = torch.stack(val_signals)
val_labels_tensor = torch.tensor(normalized_val_labels, dtype=torch.float32)

# Create the Dataset and DataLoader for both training and validation data
train_dataset = TensorDataset(train_signals_tensor, train_labels_tensor)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = TensorDataset(val_signals_tensor, val_labels_tensor)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define the Neural Network with an additional layer and more neurons
class WaveletParamNet(nn.Module):
    def __init__(self):
        super(WaveletParamNet, self).__init__()
        self.fc1 = nn.Linear(train_signals_tensor.shape[1], 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.dropout3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(32, 3)  # Output layer: 3 units for wavelet function, level, and threshold

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        x = self.fc4(x)
        return x

# Initialize the network
model = WaveletParamNet()

# Loss and optimizer with L2 regularization
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.005)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Early stopping parameters
early_stopping_patience = 100
early_stopping_counter = 0
best_val_loss = float('inf')

# Lists to store the training and validation loss
train_losses = []
val_losses = []

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for signals, labels in train_dataloader:
        optimizer.zero_grad()
        outputs = model(signals)
        loss = criterion(outputs, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item() * signals.size(0)

    train_loss = train_loss / len(train_dataloader.dataset)
    train_losses.append(train_loss)

    # Validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for signals, labels in val_dataloader:
            outputs = model(signals)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * signals.size(0)

    val_loss = val_loss / len(val_dataloader.dataset)
    val_losses.append(val_loss)

    print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    # Check for early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stopping_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        early_stopping_counter += 1

    if early_stopping_counter >= early_stopping_patience:
        print('Early stopping')
        break

    # Step the scheduler
    scheduler.step()

# Load the best model
model.load_state_dict(torch.load('best_model.pth'))

# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
