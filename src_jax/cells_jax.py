# FILE: src_jax/cells_jax.py
import jax 
import jax.numpy as jnp
import jax.nn as jnn
import equinox as eqx

class RNNCellJax(eqx.Module):
    wx: eqx.nn.Linear
    wh: eqx.nn.Linear
    wout: eqx.nn.Linear
    def __init__(self, in_c, hidden_c, key):
        k1, k2, k3 = jax.random.split(key, 3)
        self.wx = eqx.nn.Linear(in_c, hidden_c, key=k1)
        self.wh = eqx.nn.Linear(hidden_c, hidden_c, use_bias=False, key=k2)
        self.wout = eqx.nn.Linear(hidden_c, hidden_c, key=k3)
    def __call__(self, x, h):
        return jnp.tanh(self.wout(jnn.relu(self.wx(x) + self.wh(h))))

class GRUCellJax(eqx.Module):
    gru: eqx.nn.GRUCell
    wout: eqx.nn.Linear
    def __init__(self, in_c, hidden_c, key):
        k1, k2 = jax.random.split(key, 2)
        self.gru = eqx.nn.GRUCell(in_c, hidden_c, key=k1)
        self.wout = eqx.nn.Linear(hidden_c, hidden_c, key=k2)
    def __call__(self, x, h):
        return jnp.tanh(self.wout(self.gru(x, h)))

class LSTMCellJax(eqx.Module):
    lstm: eqx.nn.LSTMCell
    wout: eqx.nn.Linear
    hidden_c: int
    def __init__(self, in_c, hidden_c, key):
        assert hidden_c % 2 == 0
        k1, k2 = jax.random.split(key, 2)
        self.lstm = eqx.nn.LSTMCell(in_c, hidden_c // 2, key=k1)
        self.wout = eqx.nn.Linear(hidden_c, hidden_c, key=k2)
        self.hidden_c = hidden_c
    def __call__(self, x, hc):
        h, c = hc[:self.hidden_c//2], hc[self.hidden_c//2:]
        h_new, c_new = self.lstm(x, (h, c))
        return jnp.tanh(self.wout(jnp.concatenate([h_new, c_new])))