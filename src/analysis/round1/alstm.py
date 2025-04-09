from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

def create_sequences(df, feature_cols, label_col='target', window_size=20):
    X, y = [], []
    features = df[feature_cols].fillna(0).values
    labels = df[label_col].values

    for i in range(len(df) - window_size):
        X.append(features[i:i+window_size])
        y.append(labels[i+window_size])
    
    return np.array(X), np.array(y)

class ALSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, lstm_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=lstm_layers, 
                            batch_first=True, dropout=dropout if lstm_layers > 1 else 0)
        self.attn_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3)
        )

    def forward(self, x):
        h, _ = self.lstm(x)
        e = self.attn_layer(h).squeeze(-1)  
        alpha = torch.softmax(e, dim=1) 
        context = torch.sum(h * alpha.unsqueeze(-1), dim=1)
        out = self.fc(context)
        return out, alpha

def analyze_attention_patterns(model, X, feature_cols, num_samples=10):
    """
    对模型预测样本提取 attention 权重并可视化，打印关键时间步的特征值。
    """
    device = next(model.parameters()).device
    model.eval()
    X_tensor = torch.tensor(X[:num_samples]).float().to(device)

    with torch.no_grad():
        _, alphas = model(X_tensor)  # [batch, seq_len]

    alphas = alphas.cpu().numpy()

    # 可视化 attention heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(alphas, cmap="viridis", cbar=True)
    plt.title("Attention Weights Heatmap (each row = one sample)")
    plt.xlabel("Time Step")
    plt.ylabel("Sample Index")
    plt.tight_layout()
    plt.show()

    # 打印最关注时间步的特征值（每个样本）
    for i, attn in enumerate(alphas):
        most_attn_step = np.argmax(attn)
        print(f"Sample {i}: most attended step = {most_attn_step}")
        print("Feature values at that step:")
        for j, feature in enumerate(feature_cols):
            print(f"  {feature}: {X[i, most_attn_step, j]:.4f}")
        print("---")

def train_model(X_train, y_train, X_val, y_val, input_dim, hidden_dim=128, num_epochs=100, lr=0.01):
    model = ALSTM(input_dim, hidden_dim, lstm_layers=2, dropout=0.3)
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_train_tensor = torch.tensor(X_train).float().to(device)
    y_train_tensor = torch.tensor(y_train).long().to(device)
    X_val_tensor = torch.tensor(X_val).float().to(device)
    y_val_tensor = torch.tensor(y_val).long().to(device)

    for epoch in range(num_epochs):
        model.train()
        y_pred, alpha = model(X_train_tensor)
        loss = criterion(y_pred, y_train_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            y_pred, alpha = model(X_val_tensor)
            val_loss = criterion(y_pred, y_val_tensor)
            print(f"Epoch: {epoch+1}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

    return model

if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv('data/round1/round1_ink.csv', index_col=0)
    
    scaler = StandardScaler()
    feature_cols = [col for col in df.columns if col not in ['day', 'timestamp', 'product', 'profit_and_loss', 'log_return5']]
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    split_idx = int(len(df)*0.8)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    
    window_size = 20
    X_train, y_train = create_sequences(train_df, feature_cols, window_size=window_size)
    X_val, y_val = create_sequences(val_df, feature_cols, window_size=window_size)
    
    input_dim = len(feature_cols)
    hidden_dim = 64 
    
    trained_model = train_model(
        X_train, y_train,
        X_val, y_val,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_epochs=200,
        lr=0.001
    )
    analyze_attention_patterns(trained_model, X_val, feature_cols)
    torch.save(trained_model.state_dict(), 'alstm_model.pth')
