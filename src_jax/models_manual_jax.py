import jax
import jax.numpy as jnp
import equinox as eqx

class JaCDEManualJax(eqx.Module):
    wx: jax.Array
    wh: jax.Array
    wout: jax.Array
    b0: jax.Array
    b1: jax.Array
    k_terms: int = eqx.field(static=True)
    activation: str = eqx.field(static=True)

    def __init__(self, in_channels, hidden_channels, cell_type, k_terms, activation, key):
        self.k_terms = k_terms
        self.activation = activation
        
        if cell_type != "rnn":
            raise NotImplementedError("Manual Jacobian for this cell is too complex.")
        
        k1, k2, k3 = jax.random.split(key, 3)
        limit_x = jnp.sqrt(6.0 / (in_channels + hidden_channels))
        self.wx = jax.random.uniform(k1, (hidden_channels, in_channels), minval=-limit_x, maxval=limit_x)
        
        limit_h = jnp.sqrt(6.0 / (hidden_channels + hidden_channels))
        self.wh = jax.random.uniform(k2, (hidden_channels, hidden_channels), minval=-limit_h, maxval=limit_h)
        self.wout = jax.random.uniform(k3, (hidden_channels, hidden_channels), minval=-limit_h, maxval=limit_h)
        
        self.b0 = jnp.zeros(hidden_channels)
        self.b1 = jnp.zeros(hidden_channels)

    def __call__(self, t, h, args):
        interp = args
        x = interp.evaluate(t)
        xdot = interp.derivative(t)

        l1 = self.wx @ x + self.wh @ h + self.b0
        relu_val = jax.nn.relu(l1)
        lout = self.wout @ relu_val + self.b1
        tanh_val = jnp.tanh(lout)

        dtanh = 1 - tanh_val**2
        
        if self.activation == "surrogate_relu":
            drelu = jax.nn.sigmoid(l1)
        else:
            drelu = (l1 > 0).astype(jnp.float32)
        
        d_outer = dtanh[:, None] * self.wout * drelu[None, :]
        
        Jx = d_outer @ self.wx
        Jh = d_outer @ self.wh

        jx = Jx @ xdot
        h_dot = jx
        curr_term = jx
        
        for _ in range(self.k_terms):
            curr_term = Jh @ curr_term
            h_dot = h_dot + curr_term
            
        return h_dot