import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn import init

from .nat_cub_spline import eval_cubic_spline


class JaCDE(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(JaCDE, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.wx = nn.Parameter(torch.empty(hidden_channels, in_channels))
        self.wh = nn.Parameter(torch.empty(hidden_channels, hidden_channels))
        self.wout = nn.Parameter(torch.empty(hidden_channels, hidden_channels))
        self.b0 = nn.Parameter(torch.empty(hidden_channels))
        self.b1 = nn.Parameter(torch.empty(hidden_channels))

        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.wx)
        init.xavier_uniform_(self.wh)
        init.xavier_uniform_(self.wout)
        init.zeros_(self.b0)
        init.zeros_(self.b1)

    def forward(self, t: Tensor, h: Tensor, args: tuple[Tensor, Tensor]):
        """Forward pass of the interpolator-based VF."""
        coeffs, dcoeffs, tobs = args
        x: Tensor = eval_cubic_spline(coeffs, tobs, t)
        xdot: Tensor = eval_cubic_spline(dcoeffs, tobs, t)

        # Forward pass
        l1 = F.linear(x, self.wx) + F.linear(h, self.wh) + self.b0
        relu = l1.relu()
        lout = F.linear(relu, self.wout) + self.b1
        tanh = lout.tanh()

        # Compute jacobians
        dtanh = 1 - tanh**2  # tanh & relu have eye Jacobians
        drelu = l1.sigmoid()  # smoother relu jacobian for better grads
        d_outer = dtanh[:, :, None] * self.wout * drelu[:, None, :]
        Jx = d_outer @ self.wx
        Jh = d_outer @ self.wh

        jx = Jx @ xdot.unsqueeze(-1)
        jxh = Jh @ jx
        jxhh = Jh @ jxh

        return (jx + jxh + jxhh).squeeze(-1)
