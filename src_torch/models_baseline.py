from torch import Tensor, nn
from src_torch.nat_cub_spline import eval_cubic_spline

class MatCDE(nn.Module):
    def __init__(self, input_channels, hidden_channels):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.linear1 = nn.Linear(hidden_channels, hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, input_channels * hidden_channels)

    def forward(self, t: Tensor, h: Tensor, args: tuple[Tensor, Tensor, Tensor]):
        _, dcoeffs, tobs = args
        xdot = eval_cubic_spline(dcoeffs, tobs, t)
        
        h = self.linear1(h)
        h = h.relu()
        h = self.linear2(h)
        h = h.tanh()
        
        h = h.view(h.size(0), self.hidden_channels, self.input_channels)
        return (h @ xdot.unsqueeze(-1)).squeeze(-1)