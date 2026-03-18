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
        # Explicit Linear layers instead of nn.GRUCell to support PyTorch JVP (forward AD)
        self.x2h = nn.Linear(in_c, 3 * hidden_c)
        self.h2h = nn.Linear(hidden_c, 3 * hidden_c)
        self.wout = nn.Linear(hidden_c, hidden_c)

    def forward(self, x, h):
        gate_x = self.x2h(x)
        gate_h = self.h2h(h)
        
        i_r, i_z, i_n = gate_x.chunk(3, 1)
        h_r, h_z, h_n = gate_h.chunk(3, 1)
        
        resetgate = torch.sigmoid(i_r + h_r)
        updategate = torch.sigmoid(i_z + h_z)
        newgate = torch.tanh(i_n + resetgate * h_n)
        
        gru_h = newgate + updategate * (h - newgate)
        return torch.tanh(self.wout(gru_h))

class LSTMCell(nn.Module):
    def __init__(self, in_c, hidden_c):
        super().__init__()
        assert hidden_c % 2 == 0, "For LSTM, hidden_dim must be an even number"
        self.hidden_c = hidden_c // 2
        # Explicit Linear layers instead of nn.LSTMCell
        self.x2h = nn.Linear(in_c, 4 * self.hidden_c)
        self.h2h = nn.Linear(self.hidden_c, 4 * self.hidden_c)
        self.wout = nn.Linear(hidden_c, hidden_c)

    def forward(self, x, hc):
        # hc is a concatenated state [h, c]
        h, c = hc.chunk(2, dim=-1)
        gates = self.x2h(x) + self.h2h(h)
        
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)
        
        c_new = (forgetgate * c) + (ingate * cellgate)
        h_new = outgate * torch.tanh(c_new)
        
        out = torch.cat([h_new, c_new], dim=-1)
        return torch.tanh(self.wout(out))