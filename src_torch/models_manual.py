import torch
import torch.nn as nn
import torch.nn.functional as F
from src_torch.nat_cub_spline import eval_cubic_spline

class JaCDEManual(nn.Module):
    def __init__(self, in_channels, hidden_channels, cell_type, k_terms):
        super().__init__()
        self.cell_type = cell_type
        self.k_terms = k_terms
        
        if cell_type != "rnn":
            raise NotImplementedError(f"Manual Jacobian for {cell_type} is too complex. Use autograd.")
            
        self.wx = nn.Parameter(torch.empty(hidden_channels, in_channels))
        self.wh = nn.Parameter(torch.empty(hidden_channels, hidden_channels))
        self.wout = nn.Parameter(torch.empty(hidden_channels, hidden_channels))
        self.b0 = nn.Parameter(torch.empty(hidden_channels))
        self.b1 = nn.Parameter(torch.empty(hidden_channels))
        
        nn.init.xavier_uniform_(self.wx)
        nn.init.xavier_uniform_(self.wh)
        nn.init.xavier_uniform_(self.wout)
        nn.init.zeros_(self.b0)
        nn.init.zeros_(self.b1)

    def forward(self, t, h, args):
        coeffs, dcoeffs, tobs = args
        x = eval_cubic_spline(coeffs, tobs, t)
        xdot = eval_cubic_spline(dcoeffs, tobs, t)

        l1 = F.linear(x, self.wx) + F.linear(h, self.wh) + self.b0
        relu = l1.relu()
        tanh = (F.linear(relu, self.wout) + self.b1).tanh()

        # Compute jacobians
        dtanh = 1 - tanh**2
        drelu = l1.sigmoid() # surrogate gradient
        d_outer = dtanh[:, :, None] * self.wout * drelu[:, None, :]
        
        Jx = d_outer @ self.wx
        Jh = d_outer @ self.wh

        jx = Jx @ xdot.unsqueeze(-1)
        h_dot = jx
        curr_term = jx
        
        for _ in range(self.k_terms):
            curr_term = Jh @ curr_term
            h_dot = h_dot + curr_term
            
        return h_dot.squeeze(-1)