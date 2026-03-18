import jax
import jax.numpy as jnp
import equinox as eqx

class JaCDEManualJax(eqx.Module):
    wx: jax.Array
    wh: jax.Array
    wout: jax.Array
    b0: jax.Array
    b1: jax.Array
    k_terms: int

    def __init__(self, in_channels, hidden_channels, cell_type, k_terms, key):
        self.k_terms = k_terms
        if cell_type != "rnn":
            raise NotImplementedError(f"Manual Jacobian for {cell_type} is too complex in JAX. Use autograd.")
        
        k1, k2, k3 = jax.random.split(key, 3)
        limit_x = jnp.sqrt(6.0 / (in_channels + hidden_channels))
        self.wx = jax.random.uniform(k1, (hidden_channels, in_channels), minval=-limit_x, maxval=limit_x)
        
        limit_h = jnp.sqrt(6.0 / (hidden_channels + hidden_channels))
        self.wh = jax.random.uniform(k2, (hidden_channels, hidden_channels), minval=-limit_h, maxval=limit_h)
        self.wout = jax.random.uniform(k3, (hidden_channels, hidden_channels), minval=-limit_h, maxval=limit_h)
        
        self.b0 = jnp.zeros(hidden_channels)
        self.b1 = jnp.zeros(hidden_channels)

    def __call__(self, t, h, args):
        interp = args # diffrax.CubicInterpolation
        x = interp.evaluate(t)
        xdot = interp.derivative(t)

        l1 = self.wx @ x + self.wh @ h + self.b0
        relu = jax.nn.relu(l1)
        lout = self.wout @ relu + self.b1
        tanh = jnp.tanh(lout)

        dtanh = 1 - tanh**2
        drelu = jax.nn.sigmoid(l1) # surrogate gradient
        
        d_outer = dtanh[:, None] * self.wout * drelu[None, :]
        Jx = d_outer @ self.wx
        Jh = d_outer @ self.wh

        # Taylor expansion with lax.fori_loop
        jx = Jx @ xdot
        h_dot = jx
        
        def body_fun(i, val):
            v_i, sum_v = val
            v_next = Jh @ v_i
            return (v_next, sum_v + v_next)
        
        _, h_dot = jax.lax.fori_loop(0, self.k_terms, body_fun, (jx, h_dot))
        return h_dot