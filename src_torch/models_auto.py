import torch
import torch.nn as nn
import random
from torch.func import functional_call, jvp, jacrev, vmap
from src_torch.nat_cub_spline import eval_cubic_spline

class JaCDEAutograd(nn.Module):
    def __init__(self, cell: nn.Module, k_terms: int, track_radius: bool = False):
        super().__init__()
        self.cell = cell
        self.k_terms = k_terms
        self.track_radius = track_radius
        
        self.train_spec_rad_sum = 0.0
        self.train_spec_rad_count = 0
        self.val_spec_rad_sum = 0.0
        self.val_spec_rad_count = 0

    def forward(self, t, h, args):
        coeffs, dcoeffs, tobs = args
        x = eval_cubic_spline(coeffs, tobs, t)
        xdot = eval_cubic_spline(dcoeffs, tobs, t)

        params = dict(self.cell.named_parameters())
        
        def func_x(x_val): return functional_call(self.cell, params, (x_val, h))
        def func_h(h_val): return functional_call(self.cell, params, (x, h_val))

        _, v = jvp(func_x, (x,), (xdot,), strict=False)
        h_dot = v 

        for _ in range(self.k_terms):
            _, v = jvp(func_h, (h,), (v,), strict=False)
            h_dot = h_dot + v

        if self.track_radius:
            with torch.no_grad():
                if random.random() < 0.20:
                    h_sub = h[:4].detach()
                    x_sub = x[:4].detach()

                    def get_jac(h_s, x_s):
                        def f(hidden):
                            return functional_call(self.cell, params, (x_s, hidden))
                        return jacrev(f)(h_s)

                    Jh_sub = vmap(get_jac)(h_sub, x_sub).cpu()
                    
                    eigvals = torch.linalg.eigvals(Jh_sub)
                    spec_rad = torch.abs(eigvals).max(dim=-1).values.mean().item()
                    
                    if self.training:
                        self.train_spec_rad_sum += spec_rad
                        self.train_spec_rad_count += 1
                    else:
                        self.val_spec_rad_sum += spec_rad
                        self.val_spec_rad_count += 1

        return h_dot