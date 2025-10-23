# src/models/__init__.py
from .rnn import BiLSTMAttn
from .cnn import KimCNN  # if you have the CNN file too
__all__ = ["BiLSTMAttn", "KimCNN"]
