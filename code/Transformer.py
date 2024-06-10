import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

class ETTH1Dataset(Dataset):
    def __init__(self, data, seq_len, label_len, pred_len, transform=None):
        self.data = data
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.transform = transform

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        seq_x = self.data[idx:idx + self.seq_len, :-1]
        seq_y = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_len, -1]

        if self.transform:
            seq_x = self.transform(seq_x)
            seq_y = self.transform(seq_y)

        return seq_x, seq_y

def load_data(file_path):
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df

def preprocess_data(df):
    scaler_features = StandardScaler()
    scaler_target = StandardScaler()
    df[['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']] = scaler_features.fit_transform(df[['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']])
    df['OT'] = scaler_target.fit_transform(df[['OT']])
    return df, scaler_features, scaler_target

def split_data(df, seq_len, label_len, pred_len):
    data = df[['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']].values
    train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)
    train_data, val_data = train_test_split(train_data, test_size=0.25, shuffle=False)

    train_dataset = ETTH1Dataset(train_data, seq_len, label_len, pred_len)
    val_dataset = ETTH1Dataset(val_data, seq_len, label_len, pred_len)
    test_dataset = ETTH1Dataset(test_data, seq_len, label_len, pred_len)

    return train_dataset, val_dataset, test_dataset

class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, output_dim, num_heads, num_encoder_layers, num_decoder_layers, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.model_dim = model_dim

        self.encoder_embedding = nn.Linear(input_dim, model_dim)
        self.decoder_embedding = nn.Linear(output_dim, model_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 1000, model_dim))

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True, dropout=dropout)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)

        self.decoder_layer = nn.TransformerDecoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True, dropout=dropout)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_decoder_layers)

        self.output_layer = nn.Linear(model_dim, output_dim)

    def forward(self, src, tgt):
        src = self.encoder_embedding(src) + self.positional_encoding[:, :src.size(1), :]
        tgt = self.decoder_embedding(tgt) + self.positional_encoding[:, :tgt.size(1), :]

        memory = self.encoder(src)
        output = self.decoder(tgt, memory)

        return self.output_layer(output)

def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            src, tgt = batch
            src, tgt = src.to(device).float(), tgt.to(device).float().unsqueeze(-1)
            optimizer.zero_grad()
            output = model(src, tgt)
            loss = criterion(output, tgt)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * src.size(0)

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                src, tgt = batch
                src, tgt = src.to(device).float(), tgt.to(device).float().unsqueeze(-1)
                output = model(src, tgt)
                loss = criterion(output, tgt)
                val_loss += loss.item() * src.size(0)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader.dataset)}, Val Loss: {val_loss/len(val_loader.dataset)}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_transformer_model.pth')

def evaluate_model(model, criterion, test_loader, scaler_target, start_idx=0, end_idx=100, save_path="predictions_transformer.png"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    test_loss = 0.0
    predictions = []
    actuals = []
    with torch.no_grad():
        for batch in test_loader:
            src, tgt = batch
            src, tgt = src.to(device).float(), tgt.to(device).float().unsqueeze(-1)
            output = model(src, tgt)
            loss = criterion(output, tgt)
            test_loss += loss.item() * src.size(0)
            predictions.append(output.cpu().numpy())
            actuals.append(tgt.cpu().numpy())

    predictions = np.concatenate(predictions)
    actuals = np.concatenate(actuals)
    print(f"Test MAE: {test_loss/len(test_loader.dataset)}")

    r2_ot = r2_score(actuals.reshape(-1, 1), predictions.reshape(-1, 1))
    print(f"R2 OT: {r2_ot}")

    predictions = scaler_target.inverse_transform(predictions.reshape(-1, 1)).reshape(predictions.shape)
    actuals = scaler_target.inverse_transform(actuals.reshape(-1, 1)).reshape(actuals.shape)

    plt.figure(figsize=(12, 6))
    plt.plot(actuals[start_idx:end_idx].flatten(), label='Actual')
    plt.plot(predictions[start_idx:end_idx].flatten(), label='Predicted')
    plt.title('Oil Temperature Prediction')
    plt.xlabel('Time Step')
    plt.ylabel('Oil Temperature')
    plt.legend()
    plt.savefig(save_path)
    plt.show()

def main():
    file_path = '/content/drive/MyDrive/ETTh1.csv'
    seq_len = 96
    label_len = 48
    pred_len = 96

    df = load_data(file_path)
    df, scaler_features, scaler_target = preprocess_data(df)
    train_dataset, val_dataset, test_dataset = split_data(df, seq_len, label_len, pred_len)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    input_dim = 6  # 6 features
    model_dim = 512
    output_dim = 1
    num_heads = 4
    num_encoder_layers = 1
    num_decoder_layers = 1
    dropout = 0.1

    model = TransformerModel(input_dim, model_dim, output_dim, num_heads, num_encoder_layers, num_decoder_layers, dropout)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=1)
    evaluate_model(model, criterion, test_loader, scaler_target, start_idx=100, end_idx=120, save_path="predictions_transformer.png")

if __name__ == "__main__":
    main()
