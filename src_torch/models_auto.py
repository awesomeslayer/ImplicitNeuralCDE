# FILE: src_torch/models_auto.py
import torch
import torch.nn as nn
from torch.func import functional_call, jvp
from src_torch.nat_cub_spline import eval_cubic_spline

class JaCDEAutograd(nn.Module):
    def __init__(self, cell: nn.Module, k_terms: int):
        super().__init__()
        self.cell = cell
        self.k_terms = k_terms

    def forward(self, t, h, args):
        coeffs, dcoeffs, tobs = args
        x = eval_cubic_spline(coeffs, tobs, t)
        xdot = eval_cubic_spline(dcoeffs, tobs, t)

        params = dict(self.cell.named_parameters())
        
        def func_x(x_val): return functional_call(self.cell, params, (x_val, h))
        def func_h(h_val): return functional_call(self.cell, params, (x, h_val))

        _, v = jvp(func_x, (x,), (xdot,), strict=False)
        h_dot = v 

        # Taylor seirs (k > 0)
        for _ in range(self.k_terms):
            _, v = jvp(func_h, (h,), (v,), strict=False)
            h_dot = h_dot + v

        return h_dot