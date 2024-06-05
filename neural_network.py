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

folder_path = 'path_to_your_folder'  # Update this to your folder path
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
        self.fc1 = nn.Linear(1000, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 4)  # Assuming we predict 4 parameters

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)  # No activation on output
        return x

# Step 4: Training the Neural Network
model = WaveletParamNet()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    for signals, params in dataloader:
        outputs = model(signals)
        loss = criterion(outputs, params)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Step 5: Example Prediction
model.eval()
with torch.no_grad():
    example_signal = signals_tensor[0].unsqueeze(0)  # Use the first signal as an example
    predicted_params = model(example_signal).numpy()
    print(predicted_params)  # Use these parameters in your C++ code

# Step 6: Convert PyTorch Model to ONNX (Optional)
# dummy_input = torch.randn(1, 1000)
# torch.onnx.export(model, dummy_input, "wavelet_param_net.onnx", input_names=['input'], output_names=['output'])
