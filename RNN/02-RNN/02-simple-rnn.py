import torch
from torch import nn


class RNN(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
