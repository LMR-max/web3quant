from __future__ import annotations

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


class LSTMRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out.squeeze(-1)


class TransformerRegressor(nn.Module):
    def __init__(self, input_dim: int, nhead: int = 4, num_layers: int = 2, dim_ff: int = 128):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, dim_ff)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_ff, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(dim_ff, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.encoder(x)
        x = x[:, -1, :]
        return self.fc(x).squeeze(-1)


def get_model(name: str, input_dim: int):
    if torch is None:
        raise ImportError("torch is required for deep models")
    if name == "lstm":
        return LSTMRegressor(input_dim=input_dim)
    if name == "transformer":
        return TransformerRegressor(input_dim=input_dim)
    raise ValueError(f"Unsupported deep model: {name}")
