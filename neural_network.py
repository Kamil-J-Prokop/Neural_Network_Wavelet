import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

# Step 1: Data Preparation

def load_data_from_folder(folder_path):
    signals = []
    params = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            data = pd.read_csv(file_path)
            signal = data['Signal'].values
            signal_tensor = torch.tensor(signal, dtype=torch.float32)
            signal_tensor = (signal_tensor - torch.mean(signal_tensor)) / torch.std(signal_tensor)  # Normalize the signal
            signals.append(signal_tensor)
            # Extract parameters from the file name
            params.append(extract_params_from_filename(file_name))
    return signals, params

def extract_params_from_filename(file_name):
    # Implement your logic to extract parameters from the file name or elsewhere
    # This is just a placeholder example
    # Let's assume the filename is like 'signal_100_10_10_1_32.csv'
    parts = file_name.replace('.csv', '').split('_')
    return [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]

folder_path = 'training_data'  # Update this to your folder path
signals, params = load_data_from_folder(folder_path)

# Convert lists to tensors
signals_tensor = torch.stack(signals)
params_tensor = torch.tensor(params, dtype=torch.float32)

# Step 2: Create the Dataset and DataLoader
dataset = TensorDataset(signals_tensor, params_tensor)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)  # Adjust batch_size as needed

# Step 3: Define the Neural Network

class WaveletParamNet(nn.Module):
    def __init__(self):
        super(WaveletParamNet, self).__init__()
        self.fc1 = nn.Linear(signals_tensor.shape[1], 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 4)  # Assuming we predict 4 parameters

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)  # No activation on output
        return x

# Step 4: Custom Loss Function with MSE Threshold
class CustomLoss(nn.Module):
    def __init__(self, mse_threshold):
        super(CustomLoss, self).__init__()
        self.mse_threshold = mse_threshold
        self.mse_loss = nn.MSELoss()

    def forward(self, outputs, targets):
        mse = self.mse_loss(outputs, targets)
        penalty = torch.where(mse > self.mse_threshold, mse - self.mse_threshold, torch.tensor(0.0, device=mse.device))
        return mse + penalty

# Define your MSE threshold
mse_threshold = 1e-5
criterion = CustomLoss(mse_threshold)

# Step 5: Early Stopping
class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0

    def step(self, loss):
        if self.best_loss is None:
            self.best_loss = loss
        elif loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

# Initialize early stopping
early_stopping = EarlyStopping(patience=10, min_delta=1e-6)

# Initialize model, optimizer
model = WaveletParamNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Step 6: Training the Neural Network with Early Stopping
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    for signals, params in dataloader:
        outputs = model(signals)
        loss = criterion(outputs, params)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation step
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for val_signals, val_params in dataloader:  # Using the same dataloader for simplicity
            val_outputs = model(val_signals)
            val_loss += criterion(val_outputs, val_params).item()
        val_loss /= len(dataloader)

    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}')

    if early_stopping.step(val_loss):
        print(f'Early stopping at epoch {epoch + 1}')
        break

# Step 7: Example Prediction
model.eval()
with torch.no_grad():
    example_signal = signals_tensor[0].unsqueeze(0)  # Use the first signal as an example
    predicted_params = model(example_signal).numpy()
    print(predicted_params)
    print(predicted_params)  # Use these parameters in your C++ code

# Step 8: Convert PyTorch Model to ONNX (Optional)
# dummy_input = torch.randn(1, signals_tensor.shape[1])
# torch.onnx.export(model, dummy_input, "wavelet_param_net.onnx", input_names=['input'], output_names=['output'])
