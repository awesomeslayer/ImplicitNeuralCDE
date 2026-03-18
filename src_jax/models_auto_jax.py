import jax
import equinox as eqx

class JaCDEAutoJax(eqx.Module):
    cell: eqx.Module
    k_terms: int = eqx.field(static=True) 
    
    def __init__(self, cell, k_terms):
        self.cell = cell
        self.k_terms = k_terms

    def __call__(self, t, h, args):
        interp = args
        x = interp.evaluate(t)
        xdot = interp.derivative(t)
        
        f_x = lambda x_v: self.cell(x_v, h)
        f_h = lambda h_v: self.cell(x, h_v)
        
        _, v = jax.jvp(f_x, (x,), (xdot,))
        h_dot = v
        
        for _ in range(self.k_terms):
            _, v = jax.jvp(f_h, (h,), (v,))
            h_dot = h_dot + v
            
        return h_dot