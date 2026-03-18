import jax
import equinox as eqx

class JaCDEAutoJax(eqx.Module):
    cell: eqx.Module
    k_terms: int
    
    def __init__(self, cell, k_terms):
        self.cell = cell
        self.k_terms = k_terms

    def __call__(self, t, h, args):
        interp = args # diffrax.CubicInterpolation passes itself in args
        x = interp.evaluate(t)
        xdot = interp.derivative(t)
        
        f_x = lambda x_v: self.cell(x_v, h)
        f_h = lambda h_v: self.cell(x, h_v)
        
        _, v = jax.jvp(f_x, (x,), (xdot,))
        h_dot = v
        
        # Taylor Expansion (k>0)
        def body(i, val):
            v_i, sum_v = val
            _, v_next = jax.jvp(f_h, (h,), (v_i,))
            return (v_next, sum_v + v_next)
            
        _, h_dot = jax.lax.fori_loop(0, self.k_terms, body, (v, h_dot))
        return h_dot