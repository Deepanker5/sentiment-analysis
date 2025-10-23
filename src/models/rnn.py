# src/models/rnn.py
import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Linear(d, 1, bias=False)
    def forward(self, h):                 # h: (B, T, H)
        a = torch.softmax(self.w(h).squeeze(-1), dim=-1)  # (B, T)
        z = (h * a.unsqueeze(-1)).sum(dim=1)              # (B, H)
        return z, a

class BiLSTMAttn(nn.Module):
    """Bidirectional LSTM + simple additive-style attention head."""
    def __init__(self, vocab_size, emb_dim, hidden, num_classes, pad_idx=0, dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(emb_dim, hidden, num_layers=1, bidirectional=True, batch_first=True)
        self.attn = Attention(hidden * 2)
        self.fc = nn.Linear(hidden * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):                 # x: (B, T)
        e = self.dropout(self.emb(x))     # (B, T, E)
        h, _ = self.rnn(e)                # (B, T, 2H)
        z, _ = self.attn(h)               # (B, 2H)
        return self.fc(self.dropout(z))
