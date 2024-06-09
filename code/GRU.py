import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

def main():
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    data = pd.read_csv('/content/drive/MyDrive/ETTh1.csv')
    
    # Preprocess the data
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)
    
    features = data[['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']].values
    target = data['OT'].values.reshape(-1, 1)

    scaler_features = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler_features.fit_transform(features)

    scaler_target = MinMaxScaler(feature_range=(0, 1))
    scaled_target = scaler_target.fit_transform(target)

    # Create sequences
    def create_sequences(data, target, seq_length):
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            x = data[i:(i + seq_length)]
            y = target[i + seq_length]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    seq_length = 24 # Adjusted sequence length for more context
    X, y = create_sequences(scaled_features, scaled_target, seq_length)

    # Split the data
    train_ratio, val_ratio, test_ratio = 0.6, 0.2, 0.2
    total_size = len(X)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    X_train, X_val, X_test = X[:train_size], X[train_size:train_size + val_size], X[-test_size:]
    y_train, y_val, y_test = y[:train_size], y[train_size:train_size + val_size], y[-test_size:]

    # Convert data to PyTorch tensors
    X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
    X_val, y_val = torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)
    X_test, y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # Adjusted batch size
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Define the GRU model
    class GRUModel(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
            super(GRUModel, self).__init__()
            self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.3)
            self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
            self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.3)

        def forward(self, x):
            h0 = torch.zeros(2, x.size(0), 50).to(device)
            out, _ = self.gru(x, h0)
            out = self.fc1(out[:, -1, :])
            out = self.relu(out)
            out = self.dropout(out)
            out = self.fc2(out)
            return out

    model = GRUModel(input_dim=6, hidden_dim=50, output_dim=1, num_layers=2).to(device)

    # criterion = nn.MSELoss()  # Mean Squared Error (MSE)
    criterion = nn.L1Loss()  # Mean Absolute Error (MAE)
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)  # Adjusted learning rate

    # Train the model
    best_loss = float('inf')
    for epoch in range(100):  # Increased number of epochs
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        val_losses = []
        model.eval()
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

    # Evaluate the model
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    y_preds = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            y_preds.append(outputs.cpu().numpy())

    y_pred = np.concatenate(y_preds, axis=0)
    y_test = y_test.numpy()

    # Inverse transform the predictions and test data
    # y_pred = scaler_target.inverse_transform(y_pred)
    # y_test = scaler_target.inverse_transform(y_test)

    # Calculate MAE and R2 score
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Test MAE: {mae:.4f}')
    print(f'R2 Score: {r2:.4f}')

    # Plot predictions
    start_index = 0
    end_index = 100

    plt.figure(figsize=(12, 6))
    plt.plot(y_test[start_index:end_index], label='True Value', linestyle='dotted')
    plt.plot(y_pred[start_index:end_index], label='Predicted Value', linestyle='dotted')
    plt.title('Oil Temperature Prediction')
    plt.xlabel('Time Step')
    plt.ylabel('Oil Temperature')
    plt.legend()
    plt.savefig('predictions_plot_GRU.png')  # Save the plot
    plt.show()

if __name__ == "__main__":
    main()