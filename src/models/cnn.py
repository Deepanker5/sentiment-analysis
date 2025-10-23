# src/models/cnn.py
import torch, torch.nn as nn

class KimCNN(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_classes, kernels=(3,4,5), n_filters=100, dropout=0.5, pad_idx=0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList([nn.Conv1d(emb_dim, n_filters, k) for k in kernels])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(kernels), num_classes)

    def forward(self, x):                 # x: (B, T)
        e = self.emb(x).transpose(1, 2)   # (B, E, T)
        c = [torch.relu(conv(e)).max(dim=-1).values for conv in self.convs]
        z = self.dropout(torch.cat(c, dim=1))
        return self.fc(z)
