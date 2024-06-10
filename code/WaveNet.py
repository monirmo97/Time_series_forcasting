import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Model Definition
class WaveNetModel(nn.Module):
    def __init__(self, layers=10, blocks=4, dilation_channels=32, residual_channels=32,
                 skip_channels=256, end_channels=256, input_features=6, output_features=1,
                 kernel_size=2, bias=False):
        super(WaveNetModel, self).__init__()
        self.layers = layers
        self.blocks = blocks
        self.dilation_channels = dilation_channels
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.input_features = input_features
        self.output_features = output_features
        self.kernel_size = kernel_size

        self.dilations = []
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()

        self.start_conv = nn.Conv1d(in_channels=self.input_features,
                                    out_channels=residual_channels,
                                    kernel_size=1,
                                    bias=bias)

        for b in range(blocks):
            new_dilation = 1
            for i in range(layers):
                self.dilations.append(new_dilation)
                self.filter_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=kernel_size,
                                                   dilation=new_dilation,
                                                   padding=(kernel_size - 1) * new_dilation,
                                                   bias=bias))
                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=kernel_size,
                                                 dilation=new_dilation,
                                                 padding=(kernel_size - 1) * new_dilation,
                                                 bias=bias))
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=1,
                                                     bias=bias))
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=1,
                                                 bias=bias))
                new_dilation *= 2

        self.end_conv_1 = nn.Conv1d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=1,
                                    bias=True)
        self.end_conv_2 = nn.Conv1d(in_channels=end_channels,
                                    out_channels=self.output_features,
                                    kernel_size=1,
                                    bias=True)

    def forward(self, x):
        x = self.start_conv(x)
        skip_connections = []

        for i in range(len(self.filter_convs)):
            filter = torch.tanh(self.filter_convs[i](x))
            gate = torch.sigmoid(self.gate_convs[i](x))
            x = filter * gate
            skip_connections.append(self.skip_convs[i](x))
            x = self.residual_convs[i](x) + x

        min_length = min([s.size(2) for s in skip_connections])
        skip_connections = [s[:, :, -min_length:] for s in skip_connections]

        x = sum(skip_connections)
        x = torch.relu(x)
        x = torch.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x[:, :, -1]

def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath)
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)
    scaler = StandardScaler()
    data[['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']] = scaler.fit_transform(data[['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']])
    return data, scaler

def create_sequences(data, window_size, prediction_length=1):
    X, y = [], []
    data_values = data[['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']].values
    target_values = data['OT'].values
    for i in range(len(data) - window_size - prediction_length + 1):
        X.append(data_values[i:(i + window_size)])
        y.append(target_values[i + window_size:i + window_size + prediction_length])
    return np.array(X), np.array(y).reshape(-1, 1)

def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=50, save_path="best_model.pth"):
    best_val_loss = float('inf')
    model.to(device)
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch.transpose(1, 2))
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_loss = evaluate_model(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader)}, Val Loss: {val_loss}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print("Saved Best Model")

def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch.transpose(1, 2))
            loss = criterion(output, y_batch)
            val_loss += loss.item()
    return val_loss / len(data_loader)

def test_model(model, test_loader, device):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch.transpose(1, 2))
            predictions.append(output.cpu())
            actuals.append(y_batch.cpu())
    return torch.cat(predictions), torch.cat(actuals)

def calculate_mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def calculate_r2_scores(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    print(f"R2 Score: {r2:.4f}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_filepath = '/content/drive/MyDrive/ETTh1.csv'
    data, scaler = load_and_preprocess_data(data_filepath)

    window_size = 24
    prediction_length = 1
    X, y = create_sequences(data, window_size, prediction_length)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Convert data to PyTorch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize model
    model = WaveNetModel(input_features=6, output_features=1)
    criterion = nn.L1Loss()  # MAE Loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=50)

    # Load the best model
    model.load_state_dict(torch.load("best_model.pth"))

    # Test the model
    predictions, actuals = test_model(model, test_loader, device)

    # Calculate and print MAE for train, validation, and test data
    train_mae = evaluate_model(model, train_loader, criterion, device)
    val_mae = evaluate_model(model, val_loader, criterion, device)
    test_mae = evaluate_model(model, test_loader, criterion, device)
    print(f"Train MAE: {train_mae:.4f}, Validation MAE: {val_mae:.4f}, Test MAE: {test_mae:.4f}")

    # Calculate R2 score for test data
    predictions = predictions.numpy()
    actuals = actuals.numpy()
    calculate_r2_scores(actuals, predictions)

    # Visualization
    start_index = 100
    end_index = 200
    plt.figure(figsize=(12, 6))
    plt.plot(actuals[start_index:end_index], label='True Value', linestyle='dotted')
    plt.plot(predictions[start_index:end_index], label='Predicted Value', linestyle='dotted')
    plt.title('Oil Temperature Prediction')
    plt.xlabel('Time Step')
    plt.ylabel('Oil Temperature')
    plt.legend()
    plt.savefig('predictions_plot_WaveNet.png')
    plt.show()

if __name__ == "__main__":
    main()
