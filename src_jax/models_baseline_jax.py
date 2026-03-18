import jax
import jax.numpy as jnp
import equinox as eqx

class BaselineCDEJax(eqx.Module):
    cell: eqx.Module
    proj: eqx.nn.Linear
    cell_type: str
    hidden_channels: int
    input_channels: int

    def __init__(self, in_c, hid_c, cell_type, key):
        self.cell_type = cell_type
        self.hidden_channels = hid_c
        self.input_channels = in_c
        k1, k2, k3 = jax.random.split(key, 3)

        if cell_type in ["mlp", "none"]:
            self.cell = eqx.nn.Linear(hid_c, hid_c, key=k1)
        elif cell_type == "rnn":
            # Простая RNN ячейка
            class SimpleRNN(eqx.Module):
                wx: eqx.nn.Linear
                wh: eqx.nn.Linear
                def __init__(self, i, h, k_a, k_b):
                    self.wx = eqx.nn.Linear(i, h, key=k_a)
                    self.wh = eqx.nn.Linear(h, h, use_bias=False, key=k_b)
                def __call__(self, x, h):
                    return jnp.tanh(self.wx(x) + self.wh(h))
            self.cell = SimpleRNN(in_c, hid_c, k1, k2)
        elif cell_type == "gru":
            self.cell = eqx.nn.GRUCell(in_c, hid_c, key=k1)
        elif cell_type == "lstm":
            self.cell = eqx.nn.LSTMCell(in_c, hid_c // 2, key=k1)
        else:
            raise ValueError(f"Unknown cell type: {cell_type}")

        self.proj = eqx.nn.Linear(hid_c, in_c * hid_c, key=k3)

    def __call__(self, t, h, args):
        interp = args
        x = interp.evaluate(t)
        xdot = interp.derivative(t)

        if self.cell_type in ["mlp", "none"]:
            out = jax.nn.relu(self.cell(h))
        elif self.cell_type == "lstm":
            h_state, c_state = h[:self.hidden_channels//2], h[self.hidden_channels//2:]
            h_new, c_new = self.cell(x, (h_state, c_state))
            out = jnp.concatenate([h_new, c_new])
        else:
            out = self.cell(x, h)
        
        matrix = jnp.tanh(self.proj(out))
        matrix = matrix.reshape((self.hidden_channels, self.input_channels))
        return matrix @ xdot