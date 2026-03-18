from torch import Tensor, nn

from .nat_cub_spline import eval_cubic_spline


class MatCDE(nn.Module):
    def __init__(self, input_channels, hidden_channels):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.linear1 = nn.Linear(hidden_channels, hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, input_channels * hidden_channels)

    def forward(self, t: Tensor, h: Tensor, args: tuple[Tensor, Tensor]):
        """Forward pass of the interpolator-based VF."""
        _, dcoeffs, tobs = args
        xdot = eval_cubic_spline(dcoeffs, tobs, t)
        # z has shape (batch, hidden_channels)
        h = self.linear1(h)
        h = h.relu()
        h = self.linear2(h)
        ######################
        # Easy-to-forget gotcha: Best results tend to be obtained by adding a final tanh nonlinearity.
        ######################
        h = h.tanh()
        ######################
        # Ignoring the batch dimension, the shape of the output tensor must be a matrix,
        # because we need it to represent a linear map from R^input_channels to R^hidden_channels.
        ######################
        h = h.view(h.size(0), self.hidden_channels, self.input_channels)
        return (h @ xdot.unsqueeze(-1)).squeeze(-1)
