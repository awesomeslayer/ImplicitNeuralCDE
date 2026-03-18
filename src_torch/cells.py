# FILE: src_torch/cells.py
import torch
import torch.nn as nn

class RNNCell(nn.Module):
    def __init__(self, in_c, hidden_c):
        super().__init__()
        self.wx = nn.Linear(in_c, hidden_c)
        self.wh = nn.Linear(hidden_c, hidden_c, bias=False)
        self.wout = nn.Linear(hidden_c, hidden_c)

    def forward(self, x, h):
        return torch.tanh(self.wout(torch.relu(self.wx(x) + self.wh(h))))

class GRUCell(nn.Module):
    def __init__(self, in_c, hidden_c):
        super().__init__()
        self.gru = nn.GRUCell(in_c, hidden_c)
        self.wout = nn.Linear(hidden_c, hidden_c)

    def forward(self, x, h):
        # Adding Tanh to the output for CDE stability
        return torch.tanh(self.wout(self.gru(x, h)))

class LSTMCell(nn.Module):
    def __init__(self, in_c, hidden_c):
        super().__init__()
        assert hidden_c % 2 == 0, "For LSTM, hidden_dim must be an even number (for h and c)"
        self.lstm = nn.LSTMCell(in_c, hidden_c // 2)
        self.wout = nn.Linear(hidden_c, hidden_c)

    def forward(self, x, hc):
        # hc is a concatenated state [h, c]
        h, c = hc.chunk(2, dim=-1)
        h_new, c_new = self.lstm(x, (h, c))
        out = torch.cat([h_new, c_new], dim=-1)
        return torch.tanh(self.wout(out))