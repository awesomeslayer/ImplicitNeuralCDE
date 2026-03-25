import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from src_torch.nat_cub_spline import eval_cubic_spline

class JaCDEManual(nn.Module):
    def __init__(self, in_channels, hidden_channels, cell_type, k_terms, activation, track_radius=False):
        super().__init__()
        self.cell_type = cell_type
        self.k_terms = k_terms
        self.activation = activation 
        self.track_radius = track_radius
        
        if cell_type != "rnn":
            raise NotImplementedError("Manual Jacobian for this cell is too complex.")
            
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

        self.train_spec_rad_sum = 0.0
        self.train_spec_rad_count = 0
        self.val_spec_rad_sum = 0.0
        self.val_spec_rad_count = 0

    def forward(self, t, h, args):
        coeffs, dcoeffs, tobs = args
        x = eval_cubic_spline(coeffs, tobs, t)
        xdot = eval_cubic_spline(dcoeffs, tobs, t)

        l1 = F.linear(x, self.wx) + F.linear(h, self.wh) + self.b0
        relu = l1.relu()
        tanh = (F.linear(relu, self.wout) + self.b1).tanh()

        dtanh = 1 - tanh**2
        
        if self.activation == "surrogate_relu":
            drelu = l1.sigmoid()
        else:
            drelu = (l1 > 0).float()
            
        d_outer = dtanh[:, :, None] * self.wout * drelu[:, None, :]
        
        Jx = d_outer @ self.wx
        Jh = d_outer @ self.wh

        if self.track_radius:
            with torch.no_grad():
                if random.random() < 0.20:
                    Jh_sub = Jh[:4].detach().cpu()
                    eigvals = torch.linalg.eigvals(Jh_sub)
                    spec_rad = torch.abs(eigvals).max(dim=-1).values.mean().item()
                    
                    if self.training:
                        self.train_spec_rad_sum += spec_rad
                        self.train_spec_rad_count += 1
                    else:
                        self.val_spec_rad_sum += spec_rad
                        self.val_spec_rad_count += 1

        jx = Jx @ xdot.unsqueeze(-1)
        h_dot = jx
        curr_term = jx
        
        for _ in range(self.k_terms):
            curr_term = Jh @ curr_term
            h_dot = h_dot + curr_term
            
        return h_dot.squeeze(-1)