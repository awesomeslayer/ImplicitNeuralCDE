import torch
from torch import Tensor, nn
from src_torch.nat_cub_spline import eval_cubic_spline

class BaselineCDE(nn.Module):
    def __init__(self, input_channels, hidden_channels, cell_type):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.cell_type = cell_type

        if cell_type in ["mlp", "none"]:
            self.cell = nn.Linear(hidden_channels, hidden_channels)
        elif cell_type == "rnn":
            self.cell = nn.RNNCell(input_channels, hidden_channels)
        elif cell_type == "gru":
            self.cell = nn.GRUCell(input_channels, hidden_channels)
        elif cell_type == "lstm":
            assert hidden_channels % 2 == 0
            self.cell = nn.LSTMCell(input_channels, hidden_channels // 2)
        else:
            raise ValueError(f"Unknown baseline cell: {cell_type}")

        self.proj = nn.Linear(hidden_channels, input_channels * hidden_channels)

    def forward(self, t: Tensor, h: Tensor, args: tuple[Tensor, Tensor, Tensor]):
        coeffs, dcoeffs, tobs = args
        x = eval_cubic_spline(coeffs, tobs, t)
        xdot = eval_cubic_spline(dcoeffs, tobs, t)
        
        if self.cell_type in ["mlp", "none"]:
            out = self.cell(h).relu()
        elif self.cell_type == "lstm":
            h_state, c_state = h.chunk(2, dim=-1)
            h_new, c_new = self.cell(x, (h_state, c_state))
            out = torch.cat([h_new, c_new], dim=-1)
        else:
            out = self.cell(x, h)
        
        matrix = self.proj(out).tanh()
        matrix = matrix.view(matrix.size(0), self.hidden_channels, self.input_channels)
        return (matrix @ xdot.unsqueeze(-1)).squeeze(-1)